import multiprocessing as mp
from multiprocessing.connection import Connection
import time
import os
import cv2
import traceback
from typing import Any, Dict, Optional, Tuple


def _encode_jpeg(frame, quality: int = 80) -> Optional[bytes]:
    try:
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,  # Включаем оптимизацию
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1  # Прогрессивный JPEG для веба
        ]
        ok, buf = cv2.imencode('.jpg', frame, encode_params)
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


class _Worker(mp.Process):
    def __init__(self, conn: Connection, jpeg_quality: int, target_fps: float):
        super().__init__(daemon=True)
        self.conn = conn
        self.jpeg_quality = int(jpeg_quality)
        self.target_dt = 1.0 / max(1e-3, float(target_fps))
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _safe_set(self, cap, prop, val):
        try:
            cap.set(prop, val)
        except Exception:
            pass

    def _open_cap(self, kind: str, sid: str, src: str) -> Tuple[bool, Dict[str, Any]]:
        try:
            cap = cv2.VideoCapture(src)
            # try to reduce decoder threading/buffers
            self._safe_set(cap, getattr(cv2, 'CAP_PROP_THREADS', 42), 1)
            self._safe_set(cap, getattr(cv2, 'CAP_PROP_BUFFERSIZE', 43), 1)
            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                return False, {"error": "open failed"}
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
            self.sessions[sid] = {
                "type": kind,
                "src": src,
                "cap": cap,
                "fps": float(fps),
                "frame_count": frame_count,
                "duration": float(duration),
                "last_ts": 0.0,
            }
            return True, {"fps": float(fps), "frame_count": frame_count, "duration": float(duration)}
        except Exception as e:
            return False, {"error": str(e)}

    def _close_cap(self, sid: str) -> bool:
        sess = self.sessions.pop(sid, None)
        if not sess:
            return True
        try:
            if sess.get("cap"):
                sess["cap"].release()
        except Exception:
            pass
        return True

    def _read_jpeg(self, sid: str, timeout_sec: float = 1.0) -> Tuple[bool, Optional[bytes]]:
        sess = self.sessions.get(sid)
        if not sess:
            return False, None
        cap = sess.get("cap")
        if cap is None:
            return False, None
        # Простое чтение без петли ожидания - либо есть кадр, либо нет
        ok, frame = cap.read()
        if ok and frame is not None:
            img = _encode_jpeg(frame, self.jpeg_quality)
            return (img is not None), img
        return False, None

    def _seek_and_read_jpeg(self, sid: str, t_sec: float, timeout_sec: float = 2.0) -> Tuple[bool, Optional[bytes]]:
        sess = self.sessions.get(sid)
        if not sess:
            return False, None
        cap = sess.get("cap")
        if cap is None:
            return False, None
        fps = float(sess.get("fps") or 25.0)
        frame_idx = int(max(0, t_sec) * fps)
        try:
            # Быстрый seek без лишних настроек
            self._safe_set(cap, cv2.CAP_PROP_POS_FRAMES, frame_idx)
            # Одно чтение без warm-up
            ok, frame = cap.read()
            if not ok or frame is None:
                return False, None
            img = _encode_jpeg(frame, self.jpeg_quality)
            return (img is not None), img
        except Exception:
            return False, None

    def run(self):
        try:
            while True:
                if not self.conn.poll(0.5):
                    continue
                try:
                    cmd, payload = self.conn.recv()
                except EOFError:
                    break
                try:
                    if cmd == "ping":
                        self.conn.send((True, {"pong": True}))
                    elif cmd == "open_rtsp":
                        ok, meta = self._open_cap("rtsp", payload["id"], payload["url"])
                        self.conn.send((ok, meta))
                    elif cmd == "open_file":
                        ok, meta = self._open_cap("file", payload["id"], payload["path"])
                        self.conn.send((ok, meta))
                    elif cmd == "close":
                        ok = self._close_cap(payload["id"])
                        self.conn.send((ok, {}))
                    elif cmd == "read_jpeg":
                        ok, img = self._read_jpeg(payload["id"], float(payload.get("timeout", 1.0)))
                        self.conn.send((ok, img))
                    elif cmd == "seek_read_jpeg":
                        ok, img = self._seek_and_read_jpeg(payload["id"], float(payload.get("t", 0.0)), float(payload.get("timeout", 1.0)))
                        self.conn.send((ok, img))
                    elif cmd == "meta":
                        sess = self.sessions.get(payload["id"], {})
                        meta = {
                            "fps": float(sess.get("fps", 0.0) or 0.0),
                            "frame_count": int(sess.get("frame_count", 0) or 0),
                            "duration": float(sess.get("duration", 0.0) or 0.0),
                            "type": sess.get("type", "")
                        }
                        self.conn.send((True, meta))
                    else:
                        self.conn.send((False, {"error": f"unknown cmd {cmd}"}))
                except Exception:
                    self.conn.send((False, {"error": traceback.format_exc()}))
        finally:
            # cleanup
            for sid in list(self.sessions.keys()):
                self._close_cap(sid)


class OpenCVIsolate:
    def __init__(self, jpeg_quality: int = 80, target_fps: float = 12.0):
        self.jpeg_quality = int(jpeg_quality)
        self.target_fps = float(target_fps)
        self.parent_conn: Optional[Connection] = None
        self.proc: Optional[_Worker] = None
        self._start()

    def _start(self):
        parent_conn, child_conn = mp.Pipe()
        self.parent_conn = parent_conn
        self.proc = _Worker(child_conn, self.jpeg_quality, self.target_fps)
        self.proc.start()

    def _ensure(self):
        if self.proc is None or not self.proc.is_alive():
            self._start()

    def _req(self, cmd: str, payload: Dict[str, Any], timeout: float = 1.0):
        """
        Send a request to the worker and wait for a response.
        On timeout we proactively terminate the worker process to avoid
        lingering blocked ffmpeg/cv2 calls (reduces 30s hang issues).
        """
        self._ensure()
        # parent_conn may be replaced by _start(); check again
        if self.parent_conn is None:
            raise RuntimeError("OpenCV worker connection not available")
        try:
            self.parent_conn.send((cmd, payload))
        except Exception:
            # If send fails, try to terminate and raise
            try:
                if self.proc and self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout=1.0)
            except Exception:
                pass
            self.parent_conn = None
            self.proc = None
            raise RuntimeError("Failed to send command to OpenCV worker")

        t0 = time.time()
        while not self.parent_conn.poll(0.05):
            if (time.time() - t0) > timeout:
                # Timed out waiting for worker — terminate it to avoid long ffmpeg hangs
                try:
                    if self.proc and self.proc.is_alive():
                        try:
                            self.proc.terminate()
                        except Exception:
                            pass
                        try:
                            self.proc.join(timeout=1.0)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Close parent_conn to avoid further use of a dead pipe
                try:
                    self.parent_conn.close()
                except Exception:
                    pass
                self.parent_conn = None
                self.proc = None
                raise TimeoutError(f"OpenCV worker timeout on {cmd}")

        ok, data = self.parent_conn.recv()
        if not ok:
            raise RuntimeError(str(data))
        return data

    # public API
    def open_rtsp(self, sid: str, url: str, timeout: float = 8.0) -> Dict[str, Any]:
        # Default timeout increased to 8s to allow slower RTSP handshakes while remaining responsive
        return self._req("open_rtsp", {"id": sid, "url": url}, timeout=timeout)

    def open_file(self, sid: str, path: str, timeout: float = 3.0) -> Dict[str, Any]:
        return self._req("open_file", {"id": sid, "path": path}, timeout=timeout)

    def close(self, sid: str) -> None:
        try:
            self._req("close", {"id": sid})
        except Exception:
            pass

    def meta(self, sid: str) -> Dict[str, Any]:
        return self._req("meta", {"id": sid})

    def read_jpeg(self, sid: str, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self._req("read_jpeg", {"id": sid, "timeout": float(timeout)}, timeout=timeout+0.5)
        except Exception:
            return None

    def seek_read_jpeg(self, sid: str, t: float, timeout: float = 2.0) -> Optional[bytes]:
        try:
            return self._req("seek_read_jpeg", {"id": sid, "t": float(t), "timeout": float(timeout)}, timeout=timeout+0.5)
        except Exception:
            return None

    def ping(self) -> bool:
        try:
            self._req("ping", {})
            return True
        except Exception:
            return False