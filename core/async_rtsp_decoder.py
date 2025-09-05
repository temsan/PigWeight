"""
AsyncRTSPDecoder - Асинхронное декодирование RTSP потоков без IPC блокировок
Поддерживает прямое копирование H.264 потока и аппаратное ускорение CUDA
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    av = None  # Используем None вместо заглушки
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Создаем заглушку для numpy
    class MockNumpy:
        class ndarray:
            pass
    np = MockNumpy()
    
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DecoderConfig:
    """Конфигурация декодера"""
    rtsp_url: str = "rtsp://localhost:8554/stream"  # Default placeholder URL
    target_fps: float = 30.0
    buffer_size: int = 3
    use_cuda: bool = True
    h264_direct: bool = True
    connection_timeout: int = 5000  # мс
    read_timeout: int = 1000  # мс
    max_retries: int = 3
    retry_delay: float = 2.0
    
@dataclass
class FrameData:
    """Данные кадра"""
    frame_id: str
    timestamp: float
    pts: int
    h264_data: Optional[bytes] = None  # Сырые H.264 данные
    rgb_frame: Optional[np.ndarray] = None  # Декодированный RGB кадр
    width: int = 0
    height: int = 0
    fps: float = 0.0

class AsyncRTSPDecoder:
    """
    Асинхронный декодер RTSP потоков с оптимизациями:
    - Устранение IPC блокировок
    - Прямое копирование H.264 потока
    - Аппаратное ускорение CUDA
    - Адаптивная настройка качества
    """
    
    def __init__(self, config: DecoderConfig, frame_callback: Callable[[FrameData], None]):
        if not AV_AVAILABLE:
            raise RuntimeError("PyAV не установлен. Используйте: pip install av")
            
        self.config = config
        self.frame_callback = frame_callback
        self.container: Optional[Any] = None
        self.video_stream: Optional[Any] = None
        
        # Состояние декодера
        self._running = False
        self._decode_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
        # Статистика
        self.stats = {
            'frames_decoded': 0,
            'frames_dropped': 0,
            'avg_fps': 0.0,
            'last_error': None,
            'connection_attempts': 0,
            'successful_connections': 0,
            'decode_latency_ms': 0.0
        }
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rtsp_decoder")
        self._frame_counter = 0
        self._last_stats_time = time.time()
        
        # CUDA контекст
        self._cuda_device = None
        if config.use_cuda and self._check_cuda():
            self._setup_cuda()
            
    def _check_cuda(self) -> bool:
        """Проверка доступности CUDA"""
        try:
            # Проверяем доступность CUDA через av
            codecs = av.codec.Codec.create('h264', 'r')
            for codec in codecs:
                if 'cuda' in codec.name.lower():
                    return True
        except Exception as e:
            logger.debug(f"CUDA недоступен: {e}")
        return False
        
    def _setup_cuda(self):
        """Настройка CUDA ускорения"""
        try:
            # Настройка CUDA декодера
            logger.info("Инициализация CUDA декодера")
            self._cuda_device = 0  # Используем первое GPU
        except Exception as e:
            logger.warning(f"Не удалось инициализировать CUDA: {e}")
            self.config.use_cuda = False
            
    async def start(self):
        """Запуск асинхронного декодирования"""
        if self._running:
            logger.warning("Декодер уже запущен")
            return
            
        logger.info(f"Запуск декодера для {self.config.rtsp_url}")
        self._running = True
        
        # Запуск задач
        self._decode_task = asyncio.create_task(self._decode_loop())
        self._stats_task = asyncio.create_task(self._stats_loop())
        
    async def stop(self):
        """Остановка декодера"""
        if not self._running:
            return
            
        logger.info("Остановка декодера")
        self._running = False
        
        # Отмена задач
        if self._decode_task:
            self._decode_task.cancel()
            try:
                await self._decode_task
            except asyncio.CancelledError:
                pass
                
        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
        
        # Закрытие контейнера
        await self._close_container()
        
        # Очистка executor
        self._executor.shutdown(wait=False)
        
    async def _decode_loop(self):
        """Основной цикл декодирования"""
        retry_count = 0
        
        while self._running:
            try:
                await self._connect_to_stream()
                retry_count = 0  # Сброс счетчика при успешном подключении
                
                await self._process_frames()
                
            except Exception as e:
                self.stats['last_error'] = str(e)
                logger.error(f"Ошибка декодирования: {e}")
                
                retry_count += 1
                if retry_count >= self.config.max_retries:
                    logger.error(f"Превышено количество попыток подключения ({self.config.max_retries})")
                    break
                    
                # Задержка перед повторной попыткой
                await asyncio.sleep(self.config.retry_delay)
                
        logger.info("Цикл декодирования завершен")
        
    async def _connect_to_stream(self):
        """Подключение к RTSP потоку"""
        self.stats['connection_attempts'] += 1
        
        await self._close_container()
        
        # Настройка опций подключения
        options = {
            'rtsp_transport': 'tcp',
            'rtsp_flags': 'prefer_tcp',
            'stimeout': str(self.config.connection_timeout * 1000),  # в микросекундах
            'buffer_size': str(self.config.buffer_size * 1024 * 1024),
            'max_delay': '100000',  # 100ms
            'fflags': 'nobuffer+fastseek+flush_packets'
        }
        
        if self.config.use_cuda:
            options['hwaccel'] = 'cuda'
            options['hwaccel_device'] = str(self._cuda_device)
            
        # Подключение в отдельном потоке
        loop = asyncio.get_event_loop()
        self.container = await loop.run_in_executor(
            self._executor, 
            self._open_container, 
            self.config.rtsp_url, 
            options
        )
        
        # Получение видео потока
        video_streams = [s for s in self.container.streams if s.type == 'video']
        if not video_streams:
            raise RuntimeError("Видео поток не найден")
            
        self.video_stream = video_streams[0]
        
        # Настройка декодера
        if self.config.use_cuda and hasattr(self.video_stream, 'codec_context'):
            self.video_stream.codec_context.hw_device_type = 'cuda'
            
        self.stats['successful_connections'] += 1
        logger.info(f"Подключение к потоку успешно: {self.video_stream.codec.name}")
        
    def _open_container(self, url: str, options: Dict[str, str]) -> Any:
        """Открытие контейнера в отдельном потоке"""
        return av.open(url, options=options, timeout=self.config.connection_timeout/1000)
        
    async def _close_container(self):
        """Закрытие контейнера"""
        if self.container:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.container.close)
            self.container = None
            self.video_stream = None
            
    async def _process_frames(self):
        """Обработка кадров из потока"""
        frame_interval = 1.0 / self.config.target_fps
        last_frame_time = 0
        
        try:
            async for packet in self._read_packets():
                if not self._running:
                    break
                    
                current_time = time.time()
                
                # Контроль FPS
                if current_time - last_frame_time < frame_interval:
                    continue
                    
                last_frame_time = current_time
                
                # Обработка пакета
                await self._process_packet(packet, current_time)
                
        except Exception as e:
            logger.error(f"Ошибка обработки кадров: {e}")
            raise
            
    async def _read_packets(self):
        """Асинхронное чтение пакетов"""
        loop = asyncio.get_event_loop()
        
        try:
            for packet in self.container.demux(self.video_stream):
                if not self._running:
                    break
                    
                # Проверка таймаута чтения
                read_start = time.time()
                yield packet
                
                # Периодическая передача управления
                if time.time() - read_start > 0.001:  # 1ms
                    await asyncio.sleep(0)
                    
        except Exception as e:
            logger.error(f"Ошибка чтения пакетов: {e}")
            raise
            
    async def _process_packet(self, packet: Any, timestamp: float):
        """Обработка отдельного пакета"""
        decode_start = time.time()
        
        try:
            # Получение сырых H.264 данных для прямой передачи
            h264_data = None
            if self.config.h264_direct and packet.buffer_size > 0:
                h264_data = bytes(packet)
                
            # Декодирование в RGB (если нужно)
            rgb_frame = None
            if not self.config.h264_direct:
                frames = packet.decode()
                if frames:
                    frame = frames[0]
                    rgb_frame = frame.to_ndarray(format='rgb24')
                    
            # Создание FrameData
            frame_data = FrameData(
                frame_id=f"frame_{self._frame_counter}",
                timestamp=timestamp,
                pts=packet.pts or 0,
                h264_data=h264_data,
                rgb_frame=rgb_frame,
                width=self.video_stream.width or 0,
                height=self.video_stream.height or 0,
                fps=float(self.video_stream.average_rate) if self.video_stream.average_rate else 0.0
            )
            
            # Отправка кадра через callback
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.frame_callback, frame_data)
            
            # Обновление статистики
            self._frame_counter += 1
            self.stats['frames_decoded'] += 1
            self.stats['decode_latency_ms'] = (time.time() - decode_start) * 1000
            
        except Exception as e:
            self.stats['frames_dropped'] += 1
            logger.warning(f"Ошибка обработки пакета: {e}")
            
    async def _stats_loop(self):
        """Цикл обновления статистики"""
        while self._running:
            try:
                current_time = time.time()
                time_diff = current_time - self._last_stats_time
                
                if time_diff >= 1.0:  # Обновление каждую секунду
                    frames_in_period = self.stats['frames_decoded']
                    self.stats['avg_fps'] = frames_in_period / time_diff if time_diff > 0 else 0
                    self._last_stats_time = current_time
                    
                    logger.debug(
                        f"FPS: {self.stats['avg_fps']:.1f}, "
                        f"Decoded: {self.stats['frames_decoded']}, "
                        f"Dropped: {self.stats['frames_dropped']}, "
                        f"Latency: {self.stats['decode_latency_ms']:.1f}ms"
                    )
                    
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Ошибка в статистике: {e}")
                await asyncio.sleep(1.0)
                
    def get_stats(self) -> Dict[str, Any]:
        """Получение текущей статистики"""
        return self.stats.copy()
        
    @property
    def is_running(self) -> bool:
        """Проверка работы декодера"""
        return self._running
        
    @property
    def stream_info(self) -> Optional[Dict[str, Any]]:
        """Информация о потоке"""
        if not self.video_stream:
            return None
            
        return {
            'codec': self.video_stream.codec.name,
            'width': self.video_stream.width,
            'height': self.video_stream.height,
            'fps': float(self.video_stream.average_rate) if self.video_stream.average_rate else 0,
            'duration': float(self.video_stream.duration) if self.video_stream.duration else None,
            'pixel_format': self.video_stream.pix_fmt
        }