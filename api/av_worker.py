import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple
import time
import random
import logging
from functools import wraps

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 0.1  # Base delay in seconds
MAX_DELAY = 5.0   # Maximum delay in seconds
BACKOFF_MULTIPLIER = 2.0

logger = logging.getLogger(__name__)

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


def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY, 
                      max_delay: float = MAX_DELAY, backoff_multiplier: float = BACKOFF_MULTIPLIER):
    """
    Декоратор для retry с экспоненциальным backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, RuntimeError) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                    total_delay = delay + jitter
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                                 f"Retrying in {total_delay:.2f}s...")
                    time.sleep(total_delay)
                except Exception as e:
                    # For non-retryable exceptions, fail immediately
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
                    raise e
            
            # This should never be reached, but just in case
            raise last_exception or RuntimeError("Unexpected retry loop exit")
        
        return wrapper
    return decorator


def health_check_retry(func):
    """
    Специальный декоратор для health check операций с более агрессивным retry
    """
    return retry_with_backoff(max_retries=5, base_delay=0.05, max_delay=2.0)(func)


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

    def _read_one(self, sid: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        sess = self.sessions.get(sid)
        if not sess:
            return False, None
        container = sess.get('container')
        vstream = sess.get('vstream')
        try:
            for packet in container.demux(vstream):
                for frame in packet.decode():
                    if not hasattr(frame, 'pts') or frame.pts is None:
                        continue
                    rgb = frame.to_ndarray(format='rgb24')
                    img = _encode_jpeg_rgb(rgb, self.jpeg_quality)
                    if img:
                        return True, {
                            "jpeg": img,
                            "pts": frame.pts,
                            "time_base": float(frame.time_base) if frame.time_base else 0.0
                        }
            return False, None
        except Exception:
            return False, None

    def _seek_and_read(self, sid: str, t: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        sess = self.sessions.get(sid)
        if not sess:
            return False, None
        container = sess.get('container')
        vstream = sess.get('vstream')
        try:
            # Compute timestamp in stream time_base units if available
            ts = None
            try:
                if vstream and getattr(vstream, 'time_base', None):
                    ts = int(max(0.0, float(t)) / float(vstream.time_base))
            except Exception:
                ts = None
            if ts is not None and vstream is not None:
                container.seek(ts, any_frame=False, backward=True, stream=vstream)
            else:
                # Fallback: AV_TIME_BASE (microseconds)
                container.seek(int(max(0.0, float(t)) * 1_000_000))
        except Exception:
            try:
                container.seek(int(max(0.0, float(t)) * 1_000_000))
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
                elif cmd == 'ping':
                    # Health check ping
                    self.conn.send((True, {'status': 'alive', 'sessions': len(self.sessions)}))
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
        
        # Health monitoring
        self._last_health_check = time.time()
        self._health_check_interval = 30.0  # Check every 30 seconds
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3
        
        # Connection recovery
        self._jpeg_quality = jpeg_quality
        self._target_fps = target_fps

    def _check_worker_health(self) -> bool:
        """Проверяет здоровье worker процесса"""
        try:
            if not self.proc.is_alive():
                logger.warning("Worker process is not alive")
                return False
            
            # Простая проверка связи
            t0 = time.time()
            self.conn.send(('ping', {}))
            if not self.conn.poll(1.0):  # Короткий timeout для ping
                logger.warning("Worker ping timeout")
                return False
            
            ok, data = self.conn.recv()
            if not ok:
                logger.warning(f"Worker ping failed: {data}")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Worker health check failed: {e}")
            return False
    
    def _restart_worker(self):
        """Перезапускает worker процесс"""
        try:
            logger.info("Restarting av_worker process...")
            
            # Terminate old process
            if hasattr(self, 'proc') and self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5.0)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join()
            
            # Create new process
            parent_conn, child_conn = mp.Pipe()
            self.conn = parent_conn
            self.proc = _Worker(child_conn, self._jpeg_quality, self._target_fps)
            self.proc.start()
            
            # Reset failure counter
            self._consecutive_failures = 0
            self._last_health_check = time.time()
            
            logger.info("av_worker process restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart av_worker: {e}")
            return False
    
    @retry_with_backoff(max_retries=3, base_delay=0.1, max_delay=2.0)
    def _req(self, cmd: str, payload: Dict[str, Any], timeout: float = 5.0):
        # Periodic health check
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            if not self._check_worker_health():
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    if not self._restart_worker():
                        raise RuntimeError("Failed to restart av_worker after health check failure")
            else:
                self._consecutive_failures = 0
            self._last_health_check = current_time
        
        # Send request with timeout
        t0 = time.time()
        try:
            self.conn.send((cmd, payload))
        except Exception as e:
            raise ConnectionError(f"Failed to send command {cmd}: {e}")
        
        # Wait for response
        while not self.conn.poll(0.05):
            if (time.time() - t0) > timeout:
                self._consecutive_failures += 1
                raise TimeoutError(f"av_worker timeout on {cmd} after {timeout}s")
        
        try:
            ok, data = self.conn.recv()
        except Exception as e:
            self._consecutive_failures += 1
            raise ConnectionError(f"Failed to receive response for {cmd}: {e}")
        
        if not ok:
            self._consecutive_failures += 1
            raise RuntimeError(str(data))
        
        # Reset failure counter on success
        self._consecutive_failures = 0
        return data

    @retry_with_backoff(max_retries=2, base_delay=0.2, max_delay=3.0)
    def open_rtsp(self, sid: str, url: str) -> Dict[str, Any]:
        return self._req('open_rtsp', {'id': sid, 'url': url})

    @retry_with_backoff(max_retries=2, base_delay=0.2, max_delay=3.0)
    def open_file(self, sid: str, path: str) -> Dict[str, Any]:
        # Увеличиваем timeout для открытия файлов, так как это может занять больше времени
        return self._req('open_file', {'id': sid, 'path': path}, timeout=5.0)

    def close(self, sid: str) -> None:
        try:
            self._req('close', {'id': sid})
        except Exception:
            pass

    def read_jpeg(self, sid: str, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        try:
            return self._req('read_jpeg', {'id': sid}, timeout=timeout)
        except Exception:
            return None

    def seek_read_jpeg(self, sid: str, t: float, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        try:
            return self._req('seek_read_jpeg', {'id': sid, 't': float(t)}, timeout=timeout)
        except Exception:
            return None

    def meta(self, sid: str) -> Dict[str, Any]:
        return self._req('meta', {'id': sid})
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Получает статистику здоровья av_worker"""
        return {
            'process_alive': self.proc.is_alive() if hasattr(self, 'proc') else False,
            'consecutive_failures': self._consecutive_failures,
            'last_health_check': self._last_health_check,
            'max_consecutive_failures': self._max_consecutive_failures,
            'health_check_interval': self._health_check_interval
        }
    
    @health_check_retry
    def ping(self) -> Dict[str, Any]:
        """Проверка связи с worker процессом"""
        return self._req('ping', {}, timeout=2.0)


