import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple
import time
import logging
from datetime import datetime

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


class RTSPDiagnosticsCollector:
    """Сборщик диагностических данных для RTSP подключений"""
    
    def __init__(self):
        self.diagnostics = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'timeouts': 0,
            'avg_connection_time': 0.0,
            'last_error': None,
            'last_error_time': None,
            'connection_history': [],
            'stages': {}
        }
    
    def log_stage(self, stage_name: str, status: str, duration: float = 0.0, error: str = None):
        """Логирует стадию подключения"""
        timestamp = datetime.now().isoformat()
        self.diagnostics['stages'][stage_name] = {
            'timestamp': timestamp,
            'status': status,
            'duration_ms': duration * 1000,
            'error': error
        }
        
        if status == 'failed':
            self.diagnostics['failed_connections'] += 1
            self.diagnostics['last_error'] = error
            self.diagnostics['last_error_time'] = timestamp
        elif status == 'success':
            self.diagnostics['successful_connections'] += 1
        
        logger.info(f"📊 RTSP диагностика [{stage_name}]: {status} ({duration*1000:.1f}ms)" + 
                   (f" - {error}" if error else ""))
    
    def log_attempt(self):
        """Логирует попытку подключения"""
        self.diagnostics['connection_attempts'] += 1
    
    def log_timeout(self):
        """Логирует таймаут"""
        self.diagnostics['timeouts'] += 1
    
    def get_diagnostics(self) -> Dict:
        """Возвращает диагностические данные"""
        return self.diagnostics.copy()


_rtsp_diagnostics = RTSPDiagnosticsCollector()


def get_rtsp_diagnostics() -> Dict:
    """Возвращает текущую диагностику RTSP"""
    return _rtsp_diagnostics.get_diagnostics()


# Декораторы retry удалены - retry логика теперь встроена в методы где это необходимо


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
        
        # Для RTSP пробуем сначала TCP, потом UDP
        transports = ['tcp', 'udp'] if kind == 'rtsp' else [None]
        last_error = None
        
        for transport in transports:
            try:
                options = {}
                if kind == 'rtsp':
                    options = {
                        'rtsp_transport': transport,
                        'fflags': 'nobuffer',
                        'flags': 'low_delay',
                        'max_delay': '0',
                        'timeout': '30000000',  # 30 секунд в микросекундах
                        'stimeout': '30000000',  # socket timeout
                    }
                    logger.info(f"Попытка подключения RTSP через {transport.upper()}: {sid}")
                
                container = av.open(src, mode='r', options=options, timeout=30.0)
                vstream = next((s for s in container.streams if s.type == 'video'), None)
                if vstream is None:
                    try: container.close()
                    except Exception: pass
                    last_error = "no video stream"
                    continue
                    
                vstream.thread_type = 'AUTO'
                fps = float(vstream.average_rate) if vstream.average_rate else 25.0
                frame_count = int(getattr(vstream, 'frames', 0) or 0)
                duration = float(container.duration / av.time_base) if container.duration else 0.0
                
                self.sessions[sid] = {
                    'kind': kind,
                    'src': src,
                    'container': container,
                    'vstream': vstream,
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': duration,
                    'transport': transport if kind == 'rtsp' else None,
                }
                
                if kind == 'rtsp':
                    logger.info(f"✅ RTSP подключение успешно через {transport.upper()}: {sid}")
                
                return True, {
                    'fps': fps, 
                    'frame_count': frame_count, 
                    'duration': duration, 
                    'type': kind,
                    'transport': transport if kind == 'rtsp' else None
                }
                
            except Exception as e:
                last_error = str(e)
                if kind == 'rtsp' and transport == 'tcp':
                    logger.warning(f"RTSP TCP не удалось для {sid}: {e}, пробуем UDP...")
                continue
        
        # Все попытки провалились
        return False, {"error": last_error or "connection failed"}

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
            return False, {"error": f"Session {sid} not found"}
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
            return False, {"error": "No frames available"}
        except Exception as e:
            return False, {"error": f"Read frame error: {str(e)}"}

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
        self._health_check_interval = 60.0  # Check every 60 seconds (реже проверяем)
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5  # Больше терпимости к временным проблемам
        
        # Connection recovery
        self._jpeg_quality = jpeg_quality
        self._target_fps = target_fps

    def _check_worker_health(self) -> bool:
        """Проверяет здоровье worker процесса"""
        try:
            if not self.proc.is_alive():
                logger.error("Worker process is not alive")
                return False
            
            # Простая проверка связи
            self.conn.send(('ping', {}))
            if not self.conn.poll(2.0):  # Увеличен timeout до 2 секунд
                # Worker занят - это нормально, не логируем как warning
                logger.debug("Worker ping timeout (worker may be busy)")
                return False
            
            ok, data = self.conn.recv()
            if not ok:
                logger.debug(f"Worker ping failed: {data}")
                return False
            
            return True
        except Exception as e:
            logger.debug(f"Worker health check failed: {e}")
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
    
    def _req(self, cmd: str, payload: Dict[str, Any], timeout: float = 15.0):
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
        
        # Wait for response with smaller poll intervals for better responsiveness
        poll_interval = 0.1
        while not self.conn.poll(poll_interval):
            elapsed = time.time() - t0
            if elapsed > timeout:
                self._consecutive_failures += 1
                # Логируем таймауты на разных уровнях в зависимости от команды
                if cmd in ['read_jpeg', 'seek_read_jpeg']:
                    # Таймауты чтения - это нормально, логируем DEBUG
                    logger.debug(f"av_worker timeout on {cmd} after {elapsed:.1f}s (worker busy)")
                elif cmd == 'close':
                    # Таймауты закрытия - WARNING
                    logger.warning(f"av_worker timeout on {cmd} after {elapsed:.1f}s")
                else:
                    # Остальные таймауты - ERROR
                    logger.error(f"av_worker timeout on {cmd} after {elapsed:.1f}s (timeout={timeout}s)")
                raise TimeoutError(f"av_worker timeout on {cmd} after {timeout}s")
        
        try:
            ok, data = self.conn.recv()
        except Exception as e:
            self._consecutive_failures += 1
            raise ConnectionError(f"Failed to receive response for {cmd}: {e}")
        
        if not ok:
            # Обработка различных типов ошибок
            if data is None:
                error_msg = f"Worker returned error without details for command {cmd}"
            elif isinstance(data, dict):
                error_msg = data.get('error', str(data))
            else:
                error_msg = str(data) if data else f"Unknown error for command {cmd}"
            
            logger.warning(f"av_worker command {cmd} failed: {error_msg}")
            self._consecutive_failures += 1
            raise RuntimeError(error_msg)
        
        # Reset failure counter on success
        self._consecutive_failures = 0
        return data

    def open_rtsp(self, sid: str, url: str) -> Dict[str, Any]:
        """Открывает RTSP поток с retry логикой"""
        logger.info(f"Попытка подключения к RTSP: {sid}")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = self._req('open_rtsp', {'id': sid, 'url': url}, timeout=30.0)
                logger.info(f"✅ RTSP подключение успешно: {sid}")
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"❌ Не удалось подключиться к RTSP {sid} после {max_attempts} попыток: {e}")
                    raise
                else:
                    delay = 1.0 * (attempt + 1)  # 1s, 2s
                    logger.warning(f"⚠️ RTSP попытка {attempt + 1}/{max_attempts} не удалась: {e}. Повтор через {delay}s...")
                    time.sleep(delay)

    def open_file(self, sid: str, path: str) -> Dict[str, Any]:
        """Открывает локальный файл"""
        logger.info(f"Открытие файла: {sid}")
        try:
            result = self._req('open_file', {'id': sid, 'path': path}, timeout=10.0)
            logger.info(f"✅ Файл открыт: {sid}")
            return result
        except Exception as e:
            logger.error(f"❌ Не удалось открыть файл {sid}: {e}")
            raise

    def close(self, sid: str) -> None:
        """Закрывает поток. Использует короткий таймаут чтобы не зависать."""
        try:
            self._req('close', {'id': sid}, timeout=5.0)  # Короткий таймаут
        except TimeoutError:
            logger.warning(f"Timeout closing {sid}, forcing cleanup")
            # Принудительно удаляем из сессий если есть доступ
        except Exception as e:
            logger.debug(f"Error closing {sid}: {e}")

    def read_jpeg(self, sid: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Читает JPEG кадр из потока. Увеличен таймаут для RTSP."""
        try:
            return self._req('read_jpeg', {'id': sid}, timeout=timeout)
        except TimeoutError:
            # Таймауты - это нормально при чтении кадров (worker занят)
            # Не логируем чтобы не засорять логи
            return None
        except ConnectionError:
            # Ошибки соединения - логируем как warning раз в N секунд
            logger.warning(f"⚠️ Connection error reading from {sid}, stream may need reconnection")
            return None
        except RuntimeError as e:
            # RuntimeError означает что worker вернул ошибку
            error_msg = str(e)
            if "10054" in error_msg or "Connection reset" in error_msg or "Errno -10054" in error_msg:
                # Камера разорвала соединение - логируем раз, не спамим
                if not hasattr(self, '_connection_reset_logged'):
                    self._connection_reset_logged = {}
                
                import time
                current_time = time.time()
                last_log = self._connection_reset_logged.get(sid, 0)
                
                if current_time - last_log > 10.0:  # Логируем раз в 10 секунд
                    logger.warning(f"⚠️ RTSP соединение разорвано камерой {sid}. Поток будет переподключен автоматически.")
                    self._connection_reset_logged[sid] = current_time
            else:
                # Другие ошибки (нет кадров и т.д.) - это нормально, не логируем
                pass
            return None
        except Exception as e:
            logger.debug(f"Unexpected error reading jpeg from {sid}: {e}")
            return None

    def seek_read_jpeg(self, sid: str, t: float, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
        try:
            return self._req('seek_read_jpeg', {'id': sid, 't': float(t)}, timeout=timeout)
        except (TimeoutError, ConnectionError):
            return None
        except RuntimeError as e:
            logger.debug(f"Seek read error for {sid}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error seeking in {sid}: {e}")
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
    
    def ping(self) -> Dict[str, Any]:
        """Проверка связи с worker процессом"""
        try:
            return self._req('ping', {}, timeout=2.0)
        except (TimeoutError, ConnectionError, RuntimeError) as e:
            # Ping timeout - это нормально когда worker занят
            logger.debug(f"Worker ping failed: {e}")
            raise


