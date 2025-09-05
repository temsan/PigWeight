"""
Конфигурация для оптимизированного асинхронного RTSP декодера
Обеспечивает настройки для максимального FPS и минимальной задержки
"""

from dataclasses import dataclass
from typing import Dict, Optional
import platform


@dataclass
class DecoderConfig:
    """
    Конфигурация асинхронного RTSP декодера для оптимизации производительности
    
    Основные цели:
    - Увеличение FPS с 12-15 до 60+
    - Снижение задержки с 200-500ms до 50-100ms
    - Устранение блокировок IPC
    - Прямая передача H.264 в WebRTC
    """
    
    # Основные параметры декодирования
    buffer_size: int = 32768  # Размер буфера FFmpeg (32KB для минимальной задержки)
    timeout_ms: int = 5000   # Таймаут соединения в миллисекундах
    max_fps: float = 60.0    # Максимальный FPS
    target_latency_ms: int = 50  # Целевая задержка в миллисекундах
    
    # Настройки качества H.264
    h264_direct: bool = True     # Прямое копирование H.264 без перекодирования
    jpeg_quality: int = 75       # Качество JPEG (fallback)
    resolution_width: int = 1280  # Ширина видео
    resolution_height: int = 720  # Высота видео
    
    # Настройки буферизации
    frame_queue_size: int = 200    # Размер очереди кадров
    memory_limit_mb: int = 200     # Лимит памяти для буферов
    drop_frames_when_full: bool = True  # Сбрасывать кадры при переполнении
    
    # Аппаратное ускорение
    hardware_acceleration: bool = True   # Использовать аппаратное ускорение
    cuda_enabled: bool = True           # CUDA для NVIDIA GPU
    prefer_hardware_decode: bool = True  # Предпочитать аппаратное декодирование
    
    # Адаптивные настройки
    adaptive_quality: bool = True        # Адаптивная настройка качества
    min_fps_threshold: float = 10.0     # Минимальный FPS для адаптации
    max_latency_threshold_ms: int = 100  # Максимальная допустимая задержка
    
    # Мониторинг производительности
    enable_stats: bool = True           # Включить сбор статистики
    stats_interval_sec: float = 1.0     # Интервал обновления статистики
    
    # RTSP параметры
    rtsp_transport: str = "tcp"         # TCP для надежности
    rtsp_flags: str = "low_delay"       # Флаги для минимальной задержки
    
    def __post_init__(self):
        """Валидация и автоматическая настройка параметров"""
        self._validate_config()
        self._auto_tune_for_platform()
    
    def _validate_config(self):
        """Валидация конфигурационных параметров"""
        if self.buffer_size < 1024:
            raise ValueError("buffer_size должен быть не менее 1024 байт")
        
        if self.max_fps <= 0 or self.max_fps > 120:
            raise ValueError("max_fps должен быть в диапазоне (0, 120]")
        
        if self.frame_queue_size < 10:
            raise ValueError("frame_queue_size должен быть не менее 10")
        
        if self.memory_limit_mb < 50:
            raise ValueError("memory_limit_mb должен быть не менее 50 MB")
        
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality должен быть в диапазоне [1, 100]")
    
    def _auto_tune_for_platform(self):
        """Автоматическая настройка под платформу"""
        system = platform.system().lower()
        
        if system == "windows":
            # Windows оптимизации
            self.buffer_size = min(self.buffer_size, 65536)
            if not self._has_nvidia_gpu():
                self.cuda_enabled = False
                self.hardware_acceleration = False
        
        elif system == "linux":
            # Linux может обрабатывать большие буферы
            self.buffer_size = max(self.buffer_size, 65536)
        
        # Адаптация под ресурсы системы
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if total_memory_gb < 4:
                # Ограниченная память
                self.memory_limit_mb = min(self.memory_limit_mb, 100)
                self.frame_queue_size = min(self.frame_queue_size, 50)
            elif total_memory_gb > 16:
                # Достаточно памяти
                self.memory_limit_mb = max(self.memory_limit_mb, 500)
                self.frame_queue_size = max(self.frame_queue_size, 500)
        except ImportError:
            pass
    
    def _has_nvidia_gpu(self) -> bool:
        """Проверка наличия NVIDIA GPU"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_ffmpeg_options(self) -> Dict[str, str]:
        """Получение оптимизированных опций FFmpeg"""
        options = {
            'rtsp_transport': self.rtsp_transport,
            'fflags': 'nobuffer+igndts+ignidx+fastseek',
            'flags': self.rtsp_flags,
            'max_delay': '0',
            'buffer_size': str(self.buffer_size),
            'stimeout': str(self.timeout_ms * 1000),  # FFmpeg ожидает микросекунды
        }
        
        if self.hardware_acceleration and self.cuda_enabled:
            options.update({
                'hwaccel': 'cuda',
                'hwaccel_output_format': 'cuda'
            })
        
        return options
    
    def get_quality_settings(self, stream_id: str = "default") -> Dict[str, any]:
        """Получение текущих настроек качества для потока"""
        return {
            'width': self.resolution_width,
            'height': self.resolution_height,
            'fps': self.max_fps,
            'jpeg_quality': self.jpeg_quality,
            'h264_direct': self.h264_direct
        }
    
    def to_dict(self) -> Dict[str, any]:
        """Конвертация в словарь для сериализации"""
        return {
            'buffer_size': self.buffer_size,
            'timeout_ms': self.timeout_ms,
            'max_fps': self.max_fps,
            'target_latency_ms': self.target_latency_ms,
            'h264_direct': self.h264_direct,
            'jpeg_quality': self.jpeg_quality,
            'resolution_width': self.resolution_width,
            'resolution_height': self.resolution_height,
            'frame_queue_size': self.frame_queue_size,
            'memory_limit_mb': self.memory_limit_mb,
            'hardware_acceleration': self.hardware_acceleration,
            'cuda_enabled': self.cuda_enabled,
            'adaptive_quality': self.adaptive_quality,
            'enable_stats': self.enable_stats
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, any]) -> 'DecoderConfig':
        """Создание конфигурации из словаря"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    @classmethod
    def for_high_performance(cls) -> 'DecoderConfig':
        """Предустановка для максимальной производительности"""
        return cls(
            buffer_size=65536,
            max_fps=60.0,
            target_latency_ms=30,
            h264_direct=True,
            frame_queue_size=500,
            memory_limit_mb=500,
            hardware_acceleration=True,
            adaptive_quality=True
        )
    
    @classmethod
    def for_low_latency(cls) -> 'DecoderConfig':
        """Предустановка для минимальной задержки"""
        return cls(
            buffer_size=16384,
            max_fps=30.0,
            target_latency_ms=50,
            h264_direct=True,
            frame_queue_size=100,
            memory_limit_mb=200,
            hardware_acceleration=True,
            adaptive_quality=True,
            drop_frames_when_full=True
        )
    
    @classmethod
    def for_cpu_only(cls) -> 'DecoderConfig':
        """Предустановка для работы без GPU"""
        return cls(
            buffer_size=32768,
            max_fps=15.0,
            target_latency_ms=100,
            h264_direct=False,
            frame_queue_size=50,
            memory_limit_mb=100,
            hardware_acceleration=False,
            cuda_enabled=False,
            adaptive_quality=True
        )


# Глобальные константы
DEFAULT_CONFIG = DecoderConfig()
HIGH_PERFORMANCE_CONFIG = DecoderConfig.for_high_performance()
LOW_LATENCY_CONFIG = DecoderConfig.for_low_latency()
CPU_ONLY_CONFIG = DecoderConfig.for_cpu_only()