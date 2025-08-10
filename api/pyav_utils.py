import io
import time
import asyncio
import numpy as np
import logging
import weakref
import threading
import queue
from typing import Optional, Tuple, Dict, Any, Union, List, Callable, TypeVar, Generic, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import contextmanager

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Типы источников видео
class StreamType(Enum):
    FILE = auto()
    RTSP = auto()
    HTTP = auto()
    RTMP = auto()

# Параметры по умолчанию для потоков
DEFAULT_TIMEOUT = 30.0  # секунд
DEFAULT_RECONNECT_ATTEMPTS = 5
DEFAULT_RECONNECT_DELAY = 1.0  # секунды
MAX_RECONNECT_DELAY = 30.0  # секунд

# Глобальные настройки FFmpeg
FFMPEG_OPTIONS = {
    'rtsp_transport': 'tcp',  # Используем TCP для стабильности
    'stimeout': '5000000',    # 5s таймаут в микросекундах
    'max_delay': '5000000',   # 5s максимальная задержка
    'fflags': 'nobuffer',     # Отключаем буферизацию
    'flags': 'low_delay',     # Минимизируем задержку
    'analyzeduration': '1000000',  # 1s для анализа формата
    'probesize': '1000000',   # 1MB для анализа формата
}

try:
    import av
    from av import VideoFrame
    from av.error import FFmpegError
    from av.video import VideoStream
    from av.container import InputContainer, OutputContainer
    from av.codec import CodecContext
    from av.packet import Packet
    from av.frame import Frame
    from av.video.stream import VideoStream as AVVideoStream
    from av.container.input import InputContainer as AVInputContainer
    from av.video.format import VideoFormat
    from av.video.plane import VideoPlane
    from av.video.reformatter import VideoReformatter
    from av.video.codec import Codec
    from av.video.frame import VideoFrame
    
    # Проверяем поддержку аппаратного ускорения
    HW_ACCEL = False
    try:
        import av.ffmpeg
        if hasattr(av.ffmpeg, 'is_hwaccel_available'):
            HW_ACCEL = av.ffmpeg.is_hwaccel_available()
    except (ImportError, AttributeError):
        pass
        
except ImportError:
    av = None
    VideoFrame = None
    FFmpegError = None
    VideoStream = None
    InputContainer = OutputContainer = None
    CodecContext = None
    Packet = None
    Frame = None
    AVVideoStream = None
    AVInputContainer = None
    VideoFormat = None
    VideoPlane = None
    VideoReformatter = None
    Codec = None
    HW_ACCEL = False


class StreamError(Exception):
    """Базовое исключение для ошибок потока"""
    pass


class StreamClosedError(StreamError):
    """Поток был закрыт"""
    pass


class StreamTimeoutError(StreamError):
    """Таймаут операции с потоком"""
    pass


@dataclass
class StreamStats:
    """Статистика потока видео"""
    fps: float = 0.0
    frame_count: int = 0
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    last_frame_time: float = 0.0
    frame_drops: int = 0
    bytes_received: int = 0
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует статистику в словарь"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'last_error': self.last_error,
            'reconnect_attempts': self.reconnect_attempts,
            'uptime': time.time() - self.start_time,
            'frame_drops': self.frame_drops,
            'bytes_received': self.bytes_received
        }


class PyAVStream:
    """Асинхронный обработчик RTSP/HTTP/RTMP потоков с использованием PyAV"""
    
    def __init__(self, url: str, stream_id: str = "stream", 
                 max_queue_size: int = 2, 
                 reconnect_attempts: int = DEFAULT_RECONNECT_ATTEMPTS,
                 reconnect_delay: float = DEFAULT_RECONNECT_DELAY):
        """
        Инициализация видеопотока
        
        Args:
            url: URL потока (rtsp://, http://, rtmp://)
            stream_id: Идентификатор потока для логирования
            max_queue_size: Максимальный размер очереди кадров
            reconnect_attempts: Количество попыток переподключения
            reconnect_delay: Базовая задержка между переподключениями (будет увеличиваться экспоненциально)
        """
        self.url = url
        self.stream_id = stream_id
        self.max_queue_size = max_queue_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # Состояние потока
        self._container = None
        self._video_stream = None
        self._running = False
        self._decode_thread = None
        self._frame_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._stats = StreamStats()
        self._lock = threading.RLock()
        self._last_frame = None
        self._last_frame_time = 0
        self._frame_ready = threading.Event()
        
        # Определяем тип потока по URL
        if url.startswith('rtsp://'):
            self.stream_type = StreamType.RTSP
        elif url.startswith('http'):
            self.stream_type = StreamType.HTTP
        elif url.startswith('rtmp://'):
            self.stream_type = StreamType.RTMP
        else:
            self.stream_type = StreamType.FILE
    
    def start(self) -> bool:
        """Запуск потока"""
        with self._lock:
            if self._running:
                logger.warning(f"[{self.stream_id}] Stream is already running")
                return True
                
            self._stop_event.clear()
            self._decode_thread = threading.Thread(
                target=self._decode_loop,
                name=f"PyAVStream-{self.stream_id}",
                daemon=True
            )
            self._decode_thread.start()
            self._running = True
            logger.info(f"[{self.stream_id}] Stream started")
            return True
    
    def stop(self):
        """Остановка потока"""
        with self._lock:
            if not self._running:
                return
                
            self._stop_event.set()
            
            # Очищаем очередь
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Ожидаем завершения потока
            if self._decode_thread and self._decode_thread.is_alive():
                self._decode_thread.join(timeout=2.0)
                
            # Закрываем контейнер
            self._close_container()
            
            self._running = False
            logger.info(f"[{self.stream_id}] Stream stopped")
    
    def read_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Чтение кадра из потока
        
        Args:
            timeout: Таймаут ожидания кадра в секундах
            
        Returns:
            numpy.ndarray or None: Кадр в формате BGR или None при таймауте/ошибке
        """
        try:
            # Пробуем получить кадр из очереди
            frame = self._frame_queue.get(timeout=timeout)
            self._frame_queue.task_done()
            return frame
        except queue.Empty:
            return None
    
    def get_last_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Получение последнего доступного кадра
        
        Args:
            timeout: Время ожидания нового кадра (если None, возвращает последний доступный)
            
        Returns:
            numpy.ndarray or None: Последний доступный кадр или None
        """
        if timeout is None or timeout <= 0:
            with self._lock:
                return self._last_frame.copy() if self._last_frame is not None else None
                
        # Ждем новый кадр с таймаутом
        if self._frame_ready.wait(timeout=timeout):
            with self._lock:
                return self._last_frame.copy() if self._last_frame is not None else None
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики потока"""
        with self._lock:
            stats = self._stats.to_dict()
            stats.update({
                'running': self._running,
                'queue_size': self._frame_queue.qsize(),
                'last_frame_age': time.time() - self._last_frame_time if self._last_frame_time > 0 else float('inf'),
                'stream_type': self.stream_type.name,
                'url': self.url
            })
            return stats
    
    def is_running(self) -> bool:
        """Проверка, запущен ли поток"""
        with self._lock:
            return self._running and not self._stop_event.is_set()
    
    def _connect(self) -> bool:
        """Подключение к потоку"""
        options = FFMPEG_OPTIONS.copy()
        
        # Опции для разных типов потоков
        if self.stream_type == StreamType.RTSP:
            options.update({
                'rtsp_flags': 'prefer_tcp',
                'rtsp_transport': 'tcp',
            })
        elif self.stream_type == StreamType.RTMP:
            options.update({
                'rtmp_buffer': '1000',
                'rtmp_live': 'live',
            })
        
        try:
            self._container = av.open(
                self.url,
                mode='r',
                format=None,  # Автоопределение
                options=options,
                timeout=(DEFAULT_TIMEOUT * 1_000_000),  # в микросекундах
            )
            
            # Находим видеопоток
            self._video_stream = next((s for s in self._container.streams if s.type == 'video'), None)
            if not self._video_stream:
                raise StreamError("No video stream found")
            
            # Настраиваем декодирование
            self._video_stream.thread_type = 'AUTO'
            self._video_stream.thread_count = 2
            
            logger.info(f"[{self.stream_id}] Connected to {self.url}, stream info: {self._video_stream}")
            return True
            
        except Exception as e:
            self._stats.last_error = str(e)
            logger.error(f"[{self.stream_id}] Connection error: {e}")
            self._close_container()
            return False
    
    def _decode_loop(self):
        """Цикл декодирования кадров"""
        reconnect_delay = self.reconnect_delay
        
        while not self._stop_event.is_set():
            try:
                # Подключаемся к потоку
                if not self._connect():
                    self._stats.reconnect_attempts += 1
                    if self._stats.reconnect_attempts >= self.reconnect_attempts:
                        logger.error(f"[{self.stream_id}] Max reconnection attempts reached")
                        break
                        
                    # Экспоненциальная задержка
                    delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
                    logger.warning(f"[{self.stream_id}] Reconnecting in {delay:.1f}s...")
                    if self._stop_event.wait(delay):
                        break
                    reconnect_delay = min(reconnect_delay * 1.5, MAX_RECONNECT_DELAY)
                    continue
                
                # Сбрасываем задержку при успешном подключении
                reconnect_delay = self.reconnect_delay
                self._stats.reconnect_attempts = 0
                
                # Очищаем очередь от старых кадров
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Читаем пакеты из потока
                for packet in self._container.demux(self._video_stream):
                    if self._stop_event.is_set():
                        break
                        
                    if packet.dts is None or packet.size == 0:
                        continue
                    
                    # Декодируем пакет в кадры
                    for frame in packet.decode():
                        if self._stop_event.is_set():
                            break
                            
                        try:
                            # Конвертируем в numpy массив (BGR)
                            frame = frame.to_ndarray(format='bgr24')
                            if frame is None or frame.size == 0:
                                continue
                            
                            # Обновляем статистику
                            self._stats.frame_count += 1
                            now = time.time()
                            self._last_frame_time = now
                            
                            # Обновляем FPS (скользящее среднее)
                            if hasattr(self, '_last_frame_processed'):
                                dt = now - self._last_frame_processed
                                if dt > 0:
                                    self._stats.fps = 0.9 * self._stats.fps + 0.1 * (1.0 / dt)
                            self._last_frame_processed = now
                            
                            # Сохраняем последний кадр
                            with self._lock:
                                self._last_frame = frame
                                self._frame_ready.set()
                                self._frame_ready.clear()
                            
                            # Добавляем кадр в очередь, если есть место
                            try:
                                self._frame_queue.put_nowait(frame)
                            except queue.Full:
                                self._stats.frame_drops += 1
                                # Пропускаем кадр, если очередь переполнена
                                pass
                                
                        except Exception as e:
                            logger.error(f"[{self.stream_id}] Frame processing error: {e}", exc_info=True)
                            self._stats.last_error = str(e)
                
            except Exception as e:
                self._stats.last_error = str(e)
                logger.error(f"[{self.stream_id}] Decoding error: {e}", exc_info=True)
                
                # Закрываем контейнер при ошибке
                self._close_container()
                
                # Ждем перед повторной попыткой
                if not self._stop_event.is_set():
                    delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
                    logger.warning(f"[{self.stream_id}] Reconnecting in {delay:.1f}s...")
                    if self._stop_event.wait(delay):
                        break
                    reconnect_delay = min(reconnect_delay * 1.5, MAX_RECONNECT_DELAY)
        
        # Очистка при завершении
        self._close_container()
        self._running = False
        logger.info(f"[{self.stream_id}] Decode loop stopped")
    
    def _close_container(self):
        """Безопасное закрытие контейнера"""
        if self._container is not None:
            try:
                self._container.close()
            except Exception as e:
                logger.error(f"[{self.stream_id}] Error closing container: {e}")
            finally:
                self._container = None
                self._video_stream = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def pyav_open_container(video_path: str, timeout: float = 30.0):
    """
    Открывает видеофайл через PyAV с таймаутом, возвращает container и video stream.
    
    Args:
        video_path: Путь к видеофайлу
        timeout: Максимальное время ожидания открытия (секунды)
        
    Returns:
        tuple: (container, stream) или (None, None) при ошибке
    """
    from api.app import perf_logger, logger
    import time
    
    if av is None:
        msg = f"[pyav] PyAV not available for {video_path}"
        perf_logger.info(msg)
        logger.debug(msg)
        return None, None
        
    start_time = time.time()
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        container = None
        try:
            logger.debug(f"[pyav] Opening container (attempt {attempt}/{max_attempts}): {video_path}")
            
            # Пробуем открыть с таймаутом
            container = av.open(video_path, timeout=(timeout * 1000 * 1000))  # в микросекундах
            
            # Получаем видеопоток
            stream = next((s for s in container.streams if s.type == 'video'), None)
            if stream is None:
                msg = f"[pyav] No video stream in {video_path}"
                perf_logger.warning(msg)
                logger.warning(msg)
                container.close()
                return None, None
                
            # Настраиваем поток для оптимального воспроизведения
            stream.thread_type = "AUTO"
            
            # Проверяем, что можем прочитать метаданные
            try:
                codec_name = stream.codec_context.name if hasattr(stream, 'codec_context') else 'unknown'
                logger.debug(
                    f"[pyav] Opened container: {video_path}, "
                    f"stream: {stream} (fps={stream.average_rate}, "
                    f"frames={stream.frames}, codec={codec_name})"
                )
                return container, stream
                
            except Exception as meta_err:
                logger.warning(f"[pyav] Error reading stream metadata: {meta_err}")
                if container:
                    container.close()
                if attempt < max_attempts:
                    time.sleep(0.5)  # Задержка перед повторной попыткой
                continue
                
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                msg = f"[pyav] Timeout opening {video_path} after {elapsed:.1f}s (attempt {attempt})"
                perf_logger.error(msg)
                logger.error(msg)
                if container:
                    try:
                        container.close()
                    except:
                        pass
                return None, None
                
            msg = f"[pyav] Error opening {video_path} (attempt {attempt}): {str(e)}"
            if attempt < max_attempts:
                logger.warning(f"{msg}, retrying...")
                time.sleep(0.5)  # Задержка перед повторной попыткой
            else:
                perf_logger.error(msg)
                logger.exception(msg)
                if container:
                    try:
                        container.close()
                    except:
                        pass
    
    return None, None

def pyav_read_frame(container, stream, seek_time: float = None, max_attempts: int = 3):
    """
    Читает raw-кадр (numpy) из PyAV. Если seek_time задан — делает seek.
    
    Args:
        container: Контейнер PyAV
        stream: Видеопоток PyAV
        seek_time: Временная метка для перехода (в секундах)
        max_attempts: Максимальное количество попыток чтения
        
    Returns:
        numpy.ndarray or None: Кадр в формате BGR или None при ошибке
    """
    from api.app import perf_logger, logger
    import time
    
    if av is None or container is None or stream is None:
        msg = "[pyav] Cannot read frame: av/container/stream not available"
        perf_logger.info(msg)
        logger.debug(msg)
        return None
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Выполняем seek, если нужно
            if seek_time is not None:
                ts = int(seek_time * stream.time_base.denominator / stream.time_base.numerator)
                logger.debug(f"[pyav] Seeking to {seek_time:.3f}s (ts={ts}) in stream {stream}")
                container.seek(ts, any_frame=False, backward=True, stream=stream)
                perf_logger.info(f"[pyav] Seek to {seek_time:.3f}s (ts={ts})")
                
            # Получаем следующий кадр
            for frame in container.decode(stream):
                if frame is None:
                    if attempt < max_attempts:
                        logger.warning(f"[pyav] Got empty frame, retrying... (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(0.1)
                        break
                    logger.warning("[pyav] Got empty frame from decoder")
                    return None
                
                # Преобразуем в numpy массив (BGR формат для OpenCV)
                try:
                    start_convert = time.time()
                    arr = frame.to_ndarray(format="bgr24")
                    convert_time = (time.time() - start_convert) * 1000
                    
                    if arr is None or arr.size == 0:
                        logger.warning(f"[pyav] Empty frame array after conversion")
                        if attempt < max_attempts:
                            time.sleep(0.1)
                            break
                        return None
                        
                    logger.debug(
                        f"[pyav] Decoded frame pts={frame.pts} time={frame.time:.3f}s "
                        f"shape={arr.shape} dtype={arr.dtype} "
                        f"(convert={convert_time:.1f}ms)"
                    )
                    perf_logger.info(
                        f"[pyav] Decoded frame pts={frame.pts} time={frame.time:.3f}s "
                        f"(convert={convert_time:.1f}ms)"
                    )
                    return arr
                    
                except Exception as e:
                    logger.error(f"[pyav] Error converting frame to numpy: {e}")
                    if attempt < max_attempts:
                        time.sleep(0.1)
                        break
                    return None
            
            # Если дошли сюда, значит кадры закончились
            if attempt < max_attempts and seek_time is not None:
                # Пробуем снова с seek, возможно, проблемы с позиционированием
                logger.warning(f"[pyav] No frames after seek, retrying... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(0.1)
                continue
                
            logger.warning("[pyav] No more frames in stream")
            return None
            
        except Exception as e:
            logger.error(f"[pyav] Error reading frame (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(0.1)
                continue
            perf_logger.error(f"[pyav_read_frame] error after {max_attempts} attempts: {e}")
            return None
    
    return None

def pyav_get_meta(container, stream):
    """
    Получает fps, frame_count, duration для PyAV видео.
    """
    if container is None or stream is None:
        return {"fps": 25.0, "frame_count": 0, "duration": 0.0}
    try:
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        frame_count = stream.frames or 0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        return {"fps": fps, "frame_count": frame_count, "duration": duration}
    except Exception:
        return {"fps": 25.0, "frame_count": 0, "duration": 0.0}

def pyav_close_container(container):
    """
    Корректно закрывает PyAV контейнер.
    """
    try:
        if container is not None:
            container.close()
    except Exception:
        pass

def pyav_seek_read_jpeg(video_path: str, t: float, jpeg_quality: int = 80) -> Optional[bytes]:
    """
    Читает jpeg-кадр из видеофайла с помощью PyAV по времени t (секунды).
    Возвращает JPEG-байты или None при ошибке.
    """
    from api.app import perf_logger
    if av is None:
        perf_logger.info(f"[pyav_seek_read_jpeg] av not available for {video_path}")
        return None
    try:
        perf_logger.info(f"[pyav_seek_read_jpeg] open {video_path} at t={t}")
        container = av.open(video_path)
        stream = next(s for s in container.streams if s.type == 'video')
        seek_pts = int(t * stream.average_rate)
        perf_logger.info(f"[pyav_seek_read_jpeg] seek_pts={seek_pts}, stream={stream}")
        container.seek(int(t * av.time_base * stream.time_base.denominator / stream.time_base.numerator), any_frame=False, backward=True, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.time >= t:
                buf = io.BytesIO()
                frame.to_image().save(buf, format='JPEG', quality=jpeg_quality)
                perf_logger.info(f"[pyav_seek_read_jpeg] frame found at pts={frame.pts}, time={frame.time}")
                return buf.getvalue()
        perf_logger.warning(f"[pyav_seek_read_jpeg] no frame found for t={t} in {video_path}")
        return None
    except Exception as e:
        perf_logger.error(f"[pyav_seek_read_jpeg] error for {video_path} at t={t}: {e}")
        return None
