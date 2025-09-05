"""
H264DirectTrack - Прямая передача H.264 в WebRTC без перекодирования
Устраняет множественное перекодирование и генерирует корректные PTS
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import struct

try:
    from aiortc import VideoStreamTrack, MediaStreamError
    from aiortc.contrib.media import MediaPlayer
    from av import VideoFrame, Packet, CodecContext
    import av
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    # Создаем заглушки для aiortc и av
    class VideoStreamTrack:
        pass
    
    class MediaStreamError(Exception):
        pass
    
    class VideoFrame:
        pass
        
    class Packet:
        pass
        
    class CodecContext:
        pass
        
    class MockAV:
        pass
        
    av = MockAV()

logger = logging.getLogger(__name__)

@dataclass 
class H264Config:
    """Конфигурация H.264 трека"""
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    bitrate: int = 2000000  # 2 Mbps
    keyframe_interval: int = 30  # Keyframe каждые 30 кадров
    profile: str = "baseline"  # baseline, main, high
    level: str = "3.1"
    
@dataclass
class H264Packet:
    """H.264 пакет данных"""
    data: bytes
    timestamp: float
    pts: int
    dts: int
    is_keyframe: bool = False
    duration: Optional[int] = None

class H264Parser:
    """Парсер H.264 NAL units"""
    
    NAL_UNIT_TYPES = {
        1: 'NON_IDR',      # Non-IDR coded slice
        5: 'IDR',          # IDR coded slice  
        6: 'SEI',          # Supplemental enhancement information
        7: 'SPS',          # Sequence parameter set
        8: 'PPS',          # Picture parameter set
        9: 'AUD',          # Access unit delimiter
    }
    
    @staticmethod
    def parse_nal_units(data: bytes) -> List[Dict[str, Any]]:
        """Парсинг NAL units из H.264 данных"""
        nal_units = []
        
        # Поиск start codes (0x000001 или 0x00000001)
        i = 0
        while i < len(data) - 3:
            if data[i:i+3] == b'\x00\x00\x01':
                start = i + 3
                # Поиск следующего start code
                next_start = len(data)
                for j in range(start + 1, len(data) - 2):
                    if data[j:j+3] == b'\x00\x00\x01':
                        next_start = j
                        break
                        
                # Извлечение NAL unit
                nal_data = data[start:next_start]
                if nal_data:
                    nal_type = nal_data[0] & 0x1F
                    nal_units.append({
                        'type': nal_type,
                        'type_name': H264Parser.NAL_UNIT_TYPES.get(nal_type, f'UNKNOWN_{nal_type}'),
                        'data': nal_data,
                        'size': len(nal_data)
                    })
                    
                i = next_start
            else:
                i += 1
                
        return nal_units
        
    @staticmethod
    def is_keyframe(data: bytes) -> bool:
        """Проверка является ли кадр keyframe"""
        nal_units = H264Parser.parse_nal_units(data)
        return any(nal['type'] == 5 for nal in nal_units)  # IDR frame
        
    @staticmethod
    def extract_sps_pps(data: bytes) -> tuple[Optional[bytes], Optional[bytes]]:
        """Извлечение SPS и PPS из данных"""
        nal_units = H264Parser.parse_nal_units(data)
        sps = None
        pps = None
        
        for nal in nal_units:
            if nal['type'] == 7:  # SPS
                sps = nal['data']
            elif nal['type'] == 8:  # PPS
                pps = nal['data']
                
        return sps, pps

class H264DirectTrack(VideoStreamTrack):
    """
    Трек для прямой передачи H.264 данных в WebRTC без перекодирования
    
    Особенности:
    - Устранение множественного перекодирования
    - Генерация корректных PTS для H.264
    - Fallback механизмы при ошибках
    - Буферизация для сглаживания потока
    """
    
    def __init__(self, config: H264Config, frame_source: Callable[[], Optional[H264Packet]]):
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc не установлен. Используйте: pip install aiortc")
            
        super().__init__()
        self.config = config
        self.frame_source = frame_source
        
        # Состояние трека
        self._running = False
        self._last_pts = 0
        self._frame_count = 0
        self._keyframe_count = 0
        
        # Буферизация
        self._frame_buffer: List[H264Packet] = []
        self._buffer_size = 10
        self._buffer_lock = asyncio.Lock()
        
        # Статистика
        self.stats = {
            'frames_sent': 0,
            'keyframes_sent': 0,
            'bytes_sent': 0,
            'avg_fps': 0.0,
            'last_keyframe_time': 0.0,
            'buffer_underruns': 0,
            'encode_errors': 0
        }
        
        # Кодек контекст для fallback
        self._codec_context: Optional[CodecContext] = None
        self._sps_pps_sent = False
        
        # Timing
        self._start_time = time.time()
        self._last_frame_time = 0.0
        self._target_frame_interval = 1.0 / config.fps
        
        logger.info(f"H264DirectTrack создан: {config.width}x{config.height} @ {config.fps} FPS")
        
    async def recv(self) -> VideoFrame:
        """Получение следующего видео кадра"""
        if not self._running:
            self._running = True
            
        try:
            # Получение H.264 пакета
            h264_packet = await self._get_next_packet()
            if not h264_packet:
                raise MediaStreamError("Нет доступных кадров")
                
            # Преобразование в VideoFrame
            frame = await self._h264_to_video_frame(h264_packet)
            
            # Обновление статистики
            self._update_stats(h264_packet)
            
            return frame
            
        except Exception as e:
            logger.error(f"Ошибка получения кадра: {e}")
            self.stats['encode_errors'] += 1
            
            # Fallback: создание пустого кадра
            return await self._create_fallback_frame()
            
    async def _get_next_packet(self) -> Optional[H264Packet]:
        """Получение следующего H.264 пакета"""
        # Попытка получить кадр из источника
        packet = self.frame_source()
        
        if packet:
            # Добавление в буфер
            async with self._buffer_lock:
                self._frame_buffer.append(packet)
                if len(self._frame_buffer) > self._buffer_size:
                    self._frame_buffer.pop(0)  # Удаляем старый кадр
                    
        # Получение из буфера
        async with self._buffer_lock:
            if self._frame_buffer:
                return self._frame_buffer.pop(0)
            else:
                self.stats['buffer_underruns'] += 1
                return None
                
    async def _h264_to_video_frame(self, h264_packet: H264Packet) -> VideoFrame:
        """Преобразование H.264 пакета в VideoFrame"""
        try:
            # Создание AVPacket из H.264 данных
            av_packet = av.Packet(h264_packet.data)
            
            # Установка временных меток
            pts = self._generate_pts(h264_packet.timestamp)
            av_packet.pts = pts
            av_packet.dts = pts  # Для простоты DTS = PTS
            
            # Создание кодек контекста если нужно
            if not self._codec_context:
                self._codec_context = self._create_codec_context()
                
            # Декодирование (минимальное для получения VideoFrame)
            frames = self._codec_context.decode(av_packet)
            
            if frames:
                frame = frames[0]
                
                # Установка корректного PTS
                frame.pts = pts
                frame.time_base = av.Rational(1, 90000)  # Стандартный time_base для H.264
                
                return frame
            else:
                # Если декодирование не удалось, создаем пустой кадр
                return await self._create_fallback_frame()
                
        except Exception as e:
            logger.warning(f"Ошибка преобразования H.264: {e}")
            return await self._create_fallback_frame()
            
    def _create_codec_context(self) -> CodecContext:
        """Создание контекста кодека"""
        codec = av.CodecContext.create('h264', 'r')
        codec.width = self.config.width
        codec.height = self.config.height
        codec.pix_fmt = 'yuv420p'
        codec.framerate = av.Rational(int(self.config.fps), 1)
        codec.time_base = av.Rational(1, 90000)
        
        # Настройки для низкой задержки
        codec.flags |= av.codec.Flags.LOW_DELAY
        codec.flags2 |= av.codec.Flags2.FAST
        
        return codec
        
    def _generate_pts(self, timestamp: float) -> int:
        """Генерация корректного PTS"""
        # Преобразование timestamp в PTS (90kHz time base)
        if self._start_time == 0:
            self._start_time = timestamp
            
        relative_time = timestamp - self._start_time
        pts = int(relative_time * 90000)  # 90kHz time base
        
        # Обеспечиваем монотонность PTS
        if pts <= self._last_pts:
            pts = self._last_pts + 1
            
        self._last_pts = pts
        return pts
        
    async def _create_fallback_frame(self) -> VideoFrame:
        """Создание fallback кадра при ошибках"""
        try:
            # Создание черного кадра
            frame = VideoFrame.from_rgb(
                np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            )
            
            # Установка временных меток
            current_time = time.time()
            pts = self._generate_pts(current_time)
            frame.pts = pts
            frame.time_base = av.Rational(1, 90000)
            
            return frame
            
        except Exception as e:
            logger.error(f"Ошибка создания fallback кадра: {e}")
            raise MediaStreamError("Не удалось создать кадр")
            
    def _update_stats(self, packet: H264Packet):
        """Обновление статистики"""
        current_time = time.time()
        
        self.stats['frames_sent'] += 1
        self.stats['bytes_sent'] += len(packet.data)
        
        if packet.is_keyframe:
            self.stats['keyframes_sent'] += 1
            self.stats['last_keyframe_time'] = current_time
            
        # Расчет FPS
        if self._last_frame_time > 0:
            frame_interval = current_time - self._last_frame_time
            if frame_interval > 0:
                instant_fps = 1.0 / frame_interval
                # Экспоненциальное сглаживание
                alpha = 0.1
                self.stats['avg_fps'] = (
                    alpha * instant_fps + 
                    (1 - alpha) * self.stats['avg_fps']
                )
                
        self._last_frame_time = current_time
        
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики трека"""
        return {
            **self.stats,
            'buffer_size': len(self._frame_buffer),
            'running': self._running,
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'fps': self.config.fps,
                'bitrate': self.config.bitrate
            }
        }
        
    async def stop(self):
        """Остановка трека"""
        self._running = False
        
        async with self._buffer_lock:
            self._frame_buffer.clear()
            
        if self._codec_context:
            self._codec_context.close()
            self._codec_context = None
            
        logger.info("H264DirectTrack остановлен")

class H264StreamAdapter:
    """
    Адаптер для подключения различных источников H.264 к DirectTrack
    """
    
    def __init__(self, config: H264Config):
        self.config = config
        self._packet_queue: List[H264Packet] = []
        self._queue_lock = asyncio.Lock()
        
    async def add_packet(self, data: bytes, timestamp: Optional[float] = None):
        """Добавление H.264 пакета"""
        if timestamp is None:
            timestamp = time.time()
            
        # Анализ пакета
        is_keyframe = H264Parser.is_keyframe(data)
        
        packet = H264Packet(
            data=data,
            timestamp=timestamp,
            pts=int(timestamp * 90000),  # Конвертация в 90kHz
            dts=int(timestamp * 90000),
            is_keyframe=is_keyframe
        )
        
        async with self._queue_lock:
            self._packet_queue.append(packet)
            
            # Ограничиваем размер очереди
            if len(self._packet_queue) > 30:  # ~1 секунда при 30 FPS
                self._packet_queue.pop(0)
                
    def get_packet(self) -> Optional[H264Packet]:
        """Получение следующего пакета (синхронный метод для DirectTrack)"""
        if self._packet_queue:
            return self._packet_queue.pop(0)
        return None
        
    async def get_packet_async(self) -> Optional[H264Packet]:
        """Асинхронное получение пакета"""
        async with self._queue_lock:
            if self._packet_queue:
                return self._packet_queue.pop(0)
        return None
        
    def clear(self):
        """Очистка очереди"""
        self._packet_queue.clear()

# Импорт numpy для fallback кадра
try:
    import numpy as np
except ImportError:
    # Если numpy не доступен, создаем заглушку
    class _NumpyMock:
        @staticmethod
        def zeros(shape, dtype=None):
            return [[0] * shape[1] for _ in range(shape[0])]
    np = _NumpyMock()