import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple
import time

try:
    import av  # PyAV
except Exception as _e:
    av = None  # type: ignore

try:
    from turbojpeg import TurboJPEG  # type: ignore
    _jpeg = TurboJPEG()
except Exception:
    _jpeg = None

try:
    from PIL import Image  # type: ignore
    from io import BytesIO
except Exception:
    Image = None  # type: ignore
    BytesIO = None  # type: ignore


def _encode_jpeg_rgb(rgb_frame, quality: int = 80) -> Optional[bytes]:
    # rgb_frame: ndarray HxWx3 uint8 in RGB order
    try:
        if _jpeg is not None:
            # TurboJPEG expects BGR or RGB depending on flags; use RGB
            return _jpeg.encode(rgb_frame, quality=quality, pixel_format=TurboJPEG.TJPF_RGB)  # type: ignore
    except Exception:
        pass
    try:
        if Image is not None and BytesIO is not None:
            im = Image.fromarray(rgb_frame, 'RGB')
            bio = BytesIO()
            im.save(bio, format='JPEG', quality=int(quality), optimize=True)
            return bio.getvalue()
    except Exception:
        pass
    return None


class _Worker(mp.Process):
    def __init__(self, conn: Connection, jpeg_quality: int, target_fps: float):
        super().__init__(daemon=True)
        self.conn = conn
        self.jpeg_quality = int(jpeg_quality)
        self.target_dt = 1.0 / max(1e-3, float(target_fps))
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _open(self, kind: str, sid: str, src: str) -> Tuple[bool, Dict[str, Any]]:
        if av is None:
            return False, {"error": "PyAV not installed"}
        try:
            options = {}
            if kind == 'rtsp':
                options = {
                    'rtsp_transport': 'tcp',
                    'fflags': 'nobuffer',
                    'flags': 'low_delay',
                    'max_delay': '0',
                }
            container = av.open(src, mode='r', options=options)
            vstream = next((s for s in container.streams if s.type == 'video'), None)
            if vstream is None:
                try: container.close()
                except Exception: pass
                return False, {"error": "no video stream"}
            vstream.thread_type = 'AUTO'
            fps = float(vstream.average_rate) if vstream.average_rate else 25.0
            frame_count = int(getattr(vstream, 'frames', 0) or 0)
            duration = float(container.duration / av.time_base) if container.duration else 0.0  # seconds
            self.sessions[sid] = {
                'kind': kind,
                'src': src,
                'container': container,
                'vstream': vstream,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
            }
            return True, {'fps': fps, 'frame_count': frame_count, 'duration': duration, 'type': kind}
        except Exception as e:
            return False, {"error": str(e)}

    def _close(self, sid: str) -> bool:
        sess = self.sessions.pop(sid, None)
        if not sess:
            return True
        try:
            if sess.get('container') is not None:
                sess['container'].close()
        except Exception:
            pass
        return True

    def _read_one(self, sid: str) -> Tuple[bool, Optional[bytes]]:
        sess = self.sessions.get(sid)
        if not sess:
            return False, None
        container = sess.get('container')
        vstream = sess.get('vstream')
        try:
            for packet in container.demux(vstream):
                for frame in packet.decode():
                    rgb = frame.to_ndarray(format='rgb24')
                    img = _encode_jpeg_rgb(rgb, self.jpeg_quality)
                    if img:
                        return True, img
            return False, None
        except Exception:
            return False, None

    def _seek_and_read(self, sid: str, t: float) -> Tuple[bool, Optional[bytes]]:
        sess = self.sessions.get(sid)
        if not sess:
            return False, None
        container = sess.get('container')
        vstream = sess.get('vstream')
        try:
            container.seek(int(max(0.0, t) / av.time_base), any_frame=False, backward=True, stream=vstream)  # type: ignore
        except Exception:
            # fallback rough seek by seconds
            try:
                container.seek(int(max(0.0, t) * 1e6))
            except Exception:
                pass
        return self._read_one(sid)

    def run(self):
        while True:
            try:
                if not self.conn.poll(0.5):
                    continue
                cmd, payload = self.conn.recv()
                if cmd == 'open_rtsp':
                    ok, meta = self._open('rtsp', payload['id'], payload['url'])
                    self.conn.send((ok, meta))
                elif cmd == 'open_file':
                    ok, meta = self._open('file', payload['id'], payload['path'])
                    self.conn.send((ok, meta))
                elif cmd == 'close':
                    ok = self._close(payload['id'])
                    self.conn.send((ok, {}))
                elif cmd == 'read_jpeg':
                    ok, img = self._read_one(payload['id'])
                    self.conn.send((ok, img))
                elif cmd == 'seek_read_jpeg':
                    ok, img = self._seek_and_read(payload['id'], float(payload.get('t', 0.0)))
                    self.conn.send((ok, img))
                elif cmd == 'meta':
                    sess = self.sessions.get(payload['id'], {})
                    meta = {
                        'fps': float(sess.get('fps', 0.0) or 0.0),
                        'frame_count': int(sess.get('frame_count', 0) or 0),
                        'duration': float(sess.get('duration', 0.0) or 0.0),
                        'type': sess.get('kind', '')
                    }
                    self.conn.send((True, meta))
                else:
                    self.conn.send((False, {"error": f"unknown cmd {cmd}"}))
            except EOFError:
                break
            except Exception as e:
                try:
                    self.conn.send((False, {"error": str(e)}))
                except Exception:
                    pass


class AVIsolate:
    def __init__(self, jpeg_quality: int = 80, target_fps: float = 12.0):
        if av is None:
            raise RuntimeError("PyAV is not installed")
        parent_conn, child_conn = mp.Pipe()
        self.conn = parent_conn
        self.proc = _Worker(child_conn, int(jpeg_quality), float(target_fps))
        self.proc.start()

    def _req(self, cmd: str, payload: Dict[str, Any], timeout: float = 1.5):
        t0 = time.time()
        self.conn.send((cmd, payload))
        while not self.conn.poll(0.05):
            if (time.time() - t0) > timeout:
                raise TimeoutError(f"av_worker timeout on {cmd}")
        ok, data = self.conn.recv()
        if not ok:
            raise RuntimeError(str(data))
        return data

    def open_rtsp(self, sid: str, url: str) -> Dict[str, Any]:
        return self._req('open_rtsp', {'id': sid, 'url': url})

    def open_file(self, sid: str, path: str) -> Dict[str, Any]:
        return self._req('open_file', {'id': sid, 'path': path})

    def close(self, sid: str) -> None:
        try:
            self._req('close', {'id': sid})
        except Exception:
            pass

    def read_jpeg(self, sid: str, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self._req('read_jpeg', {'id': sid}, timeout=timeout)
        except Exception:
            return None

    def seek_read_jpeg(self, sid: str, t: float, timeout: float = 1.5) -> Optional[bytes]:
        try:
            return self._req('seek_read_jpeg', {'id': sid, 't': float(t)}, timeout=timeout)
        except Exception:
            return None

    def meta(self, sid: str) -> Dict[str, Any]:
        return self._req('meta', {'id': sid})


