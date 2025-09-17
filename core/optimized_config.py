"""
Оптимизированная конфигурация для PigWeight
Интегрирует все новые компоненты производительности
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

from core.adaptive_quality_controller import QualityLevel, AdaptationConfig
from core.dynamic_batcher import BatcherConfig
from core.priority_frame_queue import QueueConfig
from core.performance_monitor import MonitorConfig
# from core.async_rtsp_decoder import DecoderConfig # This component is being deprecated
from core.h264_direct_track import H264Config

logger = logging.getLogger(__name__)

# Автоматическое определение доступности CUDA и ONNX
def _check_gpu_availability() -> tuple[bool, bool]:
    """Проверяет доступность GPU (CUDA) и ONNX Runtime"""
    cuda_available = False
    onnx_available = False
    
    # Проверка CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"[OK] CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("[WARNING] CUDA недоступна, используем CPU")
    except ImportError:
        logger.warning("[WARNING] PyTorch не установлен")
    
    # Проверка ONNX Runtime
    try:
        import onnxruntime as ort
        onnx_available = True
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers and cuda_available:
            logger.info("[OK] ONNX Runtime с CUDA поддержкой доступен")
        elif 'CPUExecutionProvider' in providers:
            logger.info("[OK] ONNX Runtime с CPU доступен")
        else:
            logger.warning("[WARNING] ONNX Runtime недоступен")
            onnx_available = False
    except ImportError:
        logger.warning("[WARNING] ONNX Runtime не установлен")
    
    return cuda_available, onnx_available

@dataclass
class OptimizedConfig:
    """Централизованная конфигурация для оптимизированной системы"""
    
    # Глобальные настройки
    cuda_enabled: bool = True
    cuda_device: int = 0
    use_half_precision: bool = True
    enable_h264_direct: bool = True
    
    # ONNX Runtime настройки
    use_onnx_runtime: bool = False
    onnx_providers: list = field(default_factory=list)
    auto_fallback_to_onnx: bool = True
    
    # Компонентные конфигурации
    # decoder_config: DecoderConfig = field(default_factory=DecoderConfig) # Deprecated
    queue_config: QueueConfig = field(default_factory=QueueConfig)
    batcher_config: BatcherConfig = field(default_factory=BatcherConfig)
    quality_config: AdaptationConfig = field(default_factory=AdaptationConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    h264_config: H264Config = field(default_factory=H264Config)
    
    # Производительность
    target_fps: float = 60.0
    max_concurrent_streams: int = 16
    memory_limit_mb: int = 2048
    
    # Сеть
    network_buffer_size_kb: int = 64
    webrtc_max_bitrate: int = 4000000
    webrtc_min_bitrate: int = 500000
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'OptimizedConfig':
        """Создание конфигурации из переменных окружения"""
        
        # Загрузка .env файла если указан
        if env_file and Path(env_file).exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"Загружена конфигурация из {env_file}")
            
        config = cls()
        
        # Автоматическое определение доступности GPU и ONNX
        cuda_available, onnx_available = _check_gpu_availability()
        
        # Глобальные настройки
        config.cuda_enabled = _get_env_bool('CUDA_ENABLED', cuda_available)
        config.cuda_device = _get_env_int('CUDA_DEVICE', config.cuda_device)
        config.use_half_precision = _get_env_bool('USE_HALF_PRECISION', config.use_half_precision and cuda_available)
        config.enable_h264_direct = _get_env_bool('ENABLE_H264_DIRECT', config.enable_h264_direct)
        config.target_fps = _get_env_float('TARGET_FPS', config.target_fps)
        config.max_concurrent_streams = _get_env_int('MAX_CONCURRENT_STREAMS', config.max_concurrent_streams)
        
        # ONNX Runtime настройки
        config.auto_fallback_to_onnx = _get_env_bool('AUTO_FALLBACK_TO_ONNX', True)
        
        # Автоматическое переключение на ONNX при отсутствии GPU
        if not cuda_available and onnx_available and config.auto_fallback_to_onnx:
            config.use_onnx_runtime = True
            config.cuda_enabled = False
            config.use_half_precision = False
            
            # Настройка ONNX providers
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'CPUExecutionProvider' in available_providers:
                    config.onnx_providers = ['CPUExecutionProvider']
                    logger.info("[OK] Настроен ONNX Runtime с CPU provider")
            except ImportError:
                config.use_onnx_runtime = False
                logger.error("[ERROR] ONNX Runtime недоступен")
        else:
            config.use_onnx_runtime = _get_env_bool('USE_ONNX_RUNTIME', False)
        
        # AsyncRTSPDecoder config is deprecated and removed. Stream settings are now handled by api/app.py
        
        # PriorityFrameQueue
        config.queue_config.max_size = _get_env_int('FRAME_QUEUE_MAX_SIZE', 1000)
        config.queue_config.max_memory_mb = _get_env_int('FRAME_QUEUE_MAX_MEMORY_MB', 200)
        config.queue_config.drop_threshold = _get_env_float('FRAME_QUEUE_DROP_THRESHOLD', 0.8)
        config.queue_config.max_age_seconds = _get_env_float('FRAME_QUEUE_MAX_AGE_SECONDS', 2.0)
        config.queue_config.cleanup_interval = _get_env_float('FRAME_QUEUE_CLEANUP_INTERVAL', 1.0)
        
        # DynamicBatcher
        config.batcher_config.min_batch_size = _get_env_int('BATCH_MIN_SIZE', 1)
        config.batcher_config.max_batch_size = _get_env_int('BATCH_MAX_SIZE', 16)
        config.batcher_config.initial_batch_size = _get_env_int('BATCH_INITIAL_SIZE', 4)
        config.batcher_config.target_latency_ms = _get_env_float('BATCH_TARGET_LATENCY_MS', 50.0)
        config.batcher_config.max_wait_time_ms = _get_env_float('BATCH_MAX_WAIT_MS', 100.0)
        config.batcher_config.adaptation_interval = _get_env_float('BATCH_ADAPTATION_INTERVAL', 2.0)
        config.batcher_config.throughput_weight = _get_env_float('BATCH_THROUGHPUT_WEIGHT', 0.7)
        config.batcher_config.warmup_batches = _get_env_int('BATCH_WARMUP_BATCHES', 10)
        
        # AdaptiveQualityController
        config.quality_config.cpu_threshold_high = _get_env_float('QUALITY_CPU_THRESHOLD_HIGH', 80.0)
        config.quality_config.cpu_threshold_low = _get_env_float('QUALITY_CPU_THRESHOLD_LOW', 60.0)
        config.quality_config.memory_threshold_high = _get_env_float('QUALITY_MEMORY_THRESHOLD_HIGH', 85.0)
        config.quality_config.memory_threshold_low = _get_env_float('QUALITY_MEMORY_THRESHOLD_LOW', 70.0)
        config.quality_config.gpu_threshold_high = _get_env_float('QUALITY_GPU_THRESHOLD_HIGH', 90.0)
        config.quality_config.gpu_threshold_low = _get_env_float('QUALITY_GPU_THRESHOLD_LOW', 70.0)
        config.quality_config.adaptation_interval = _get_env_float('QUALITY_ADAPTATION_INTERVAL', 2.0)
        config.quality_config.cooldown_period = _get_env_float('QUALITY_COOLDOWN_PERIOD', 10.0)
        config.quality_config.latency_threshold_ms = _get_env_float('QUALITY_LATENCY_THRESHOLD_MS', 100.0)
        
        # PerformanceMonitor
        config.monitor_config.metrics_interval = _get_env_float('MONITOR_METRICS_INTERVAL', 1.0)
        config.monitor_config.websocket_port = _get_env_int('MONITOR_WEBSOCKET_PORT', 8765)
        config.monitor_config.max_history_minutes = _get_env_int('MONITOR_MAX_HISTORY_MINUTES', 60)
        config.monitor_config.enable_alerts = _get_env_bool('MONITOR_ENABLE_ALERTS', True)
        
        # H264Config
        quality_level = os.getenv('QUALITY_INITIAL_LEVEL', 'HIGH').upper()  # BALANCED profile default
        if quality_level == 'ULTRA':
            config.h264_config.bitrate = _get_env_int('H264_ULTRA_BITRATE', 4000000)
            config.h264_config.profile = os.getenv('H264_ULTRA_PROFILE', 'high')
        elif quality_level == 'HIGH':
            config.h264_config.bitrate = _get_env_int('H264_HIGH_BITRATE', 3000000)  # BALANCED profile
            config.h264_config.profile = os.getenv('H264_HIGH_PROFILE', 'main')
        elif quality_level == 'MEDIUM':
            config.h264_config.bitrate = _get_env_int('H264_MEDIUM_BITRATE', 1500000)
            config.h264_config.profile = os.getenv('H264_MEDIUM_PROFILE', 'baseline')
        elif quality_level == 'LOW':
            config.h264_config.bitrate = _get_env_int('H264_LOW_BITRATE', 1000000)
            config.h264_config.profile = os.getenv('H264_LOW_PROFILE', 'baseline')
        else:  # MINIMAL
            config.h264_config.bitrate = _get_env_int('H264_MINIMAL_BITRATE', 500000)
            config.h264_config.profile = os.getenv('H264_MINIMAL_PROFILE', 'baseline')
            
        # Сетевые оптимизации
        config.network_buffer_size_kb = _get_env_int('NETWORK_BUFFER_SIZE_KB', 64)
        config.webrtc_max_bitrate = _get_env_int('WEBRTC_MAX_BITRATE', 4000000)
        config.webrtc_min_bitrate = _get_env_int('WEBRTC_MIN_BITRATE', 500000)
        
        return config
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для serialization"""
        return {
            'cuda_enabled': self.cuda_enabled,
            'cuda_device': self.cuda_device,
            'use_half_precision': self.use_half_precision,
            'enable_h264_direct': self.enable_h264_direct,
            'use_onnx_runtime': self.use_onnx_runtime,
            'onnx_providers': self.onnx_providers,
            'auto_fallback_to_onnx': self.auto_fallback_to_onnx,
            'target_fps': self.target_fps,
            'max_concurrent_streams': self.max_concurrent_streams,
            # 'decoder_config' is deprecated
            'queue_config': {
                'max_size': self.queue_config.max_size,
                'max_memory_mb': self.queue_config.max_memory_mb,
                'drop_threshold': self.queue_config.drop_threshold
            },
            'batcher_config': {
                'min_batch_size': self.batcher_config.min_batch_size,
                'max_batch_size': self.batcher_config.max_batch_size,
                'target_latency_ms': self.batcher_config.target_latency_ms
            },
            'quality_config': {
                'cpu_threshold_high': self.quality_config.cpu_threshold_high,
                'memory_threshold_high': self.quality_config.memory_threshold_high,
                'adaptation_interval': self.quality_config.adaptation_interval
            },
            'monitor_config': {
                'metrics_interval': self.monitor_config.metrics_interval,
                'websocket_port': self.monitor_config.websocket_port,
                'enable_alerts': self.monitor_config.enable_alerts
            },
            'h264_config': {
                'width': self.h264_config.width,
                'height': self.h264_config.height,
                'fps': self.h264_config.fps,
                'bitrate': self.h264_config.bitrate,
                'profile': self.h264_config.profile
            }
        }
        
    def validate(self) -> bool:
        """Валидация конфигурации"""
        try:
            # Проверка базовых ограничений
            assert 1 <= self.target_fps <= 120, "FPS должен быть между 1 и 120"
            assert 1 <= self.max_concurrent_streams <= 64, "Concurrent streams должен быть между 1 и 64"
            assert self.batcher_config.min_batch_size <= self.batcher_config.max_batch_size
            assert self.queue_config.max_memory_mb > 0
            assert self.webrtc_min_bitrate <= self.webrtc_max_bitrate
            
            # Проверка CUDA
            if self.cuda_enabled:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA включен в конфигурации, но недоступен в системе")
                except ImportError:
                    logger.warning("CUDA включен, но PyTorch не установлен")
                    
            return True
            
        except AssertionError as e:
            logger.error(f"Ошибка валидации конфигурации: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка валидации: {e}")
            return False

# Вспомогательные функции для чтения переменных окружения
def _get_env_bool(key: str, default: bool) -> bool:
    """Получение boolean из переменной окружения"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def _get_env_int(key: str, default: int) -> int:
    """Получение integer из переменной окружения"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Неверное значение для {key}, используется по умолчанию: {default}")
        return default

def _get_env_float(key: str, default: float) -> float:
    """Получение float из переменной окружения"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Неверное значение для {key}, используется по умолчанию: {default}")
        return default

def _get_env_list(key: str, default: list, separator: str = ',') -> list:
    """Получение списка из переменной окружения"""
    value = os.getenv(key)
    if value:
        return [item.strip() for item in value.split(separator)]
    return default

# Глобальная конфигурация (singleton)
_global_config: Optional[OptimizedConfig] = None

def get_config() -> OptimizedConfig:
    """Получение глобальной конфигурации"""
    global _global_config
    if _global_config is None:
        _global_config = OptimizedConfig.from_env()
        if not _global_config.validate():
            logger.error("Конфигурация не прошла валидацию, используются дефолтные значения")
            _global_config = OptimizedConfig()
    return _global_config

def set_config(config: OptimizedConfig):
    """Установка глобальной конфигурации"""
    global _global_config
    if config.validate():
        _global_config = config
        logger.info("Глобальная конфигурация обновлена")
    else:
        raise ValueError("Конфигурация не прошла валидацию")

def reload_config(env_file: Optional[str] = None):
    """Перезагрузка конфигурации"""
    global _global_config
    _global_config = OptimizedConfig.from_env(env_file)
    logger.info("Конфигурация перезагружена")

# Предустановленные профили конфигурации
PERFORMANCE_PROFILES = {
    'ULTRA_PERFORMANCE': {
        'TARGET_FPS': '120',
        'BATCH_MAX_SIZE': '32',
        'BATCH_TARGET_LATENCY_MS': '25',
        'QUALITY_INITIAL_LEVEL': 'ULTRA',
        'FRAME_QUEUE_MAX_MEMORY_MB': '500',
        'H264_ULTRA_BITRATE': '6000000'
    },
    
    'BALANCED': {
        'TARGET_FPS': '60',
        'BATCH_MAX_SIZE': '16', 
        'BATCH_TARGET_LATENCY_MS': '50',
        'QUALITY_INITIAL_LEVEL': 'HIGH',  # BALANCED uses HIGH quality
        'FRAME_QUEUE_MAX_MEMORY_MB': '200',
        'H264_HIGH_BITRATE': '3000000'  # Enhanced bitrate for BALANCED
    },
    
    'POWER_SAVING': {
        'TARGET_FPS': '30',
        'BATCH_MAX_SIZE': '8',
        'BATCH_TARGET_LATENCY_MS': '100',
        'QUALITY_INITIAL_LEVEL': 'MEDIUM',
        'FRAME_QUEUE_MAX_MEMORY_MB': '100',
        'H264_MEDIUM_BITRATE': '1000000'
    },
    
    'MINIMAL_RESOURCE': {
        'TARGET_FPS': '15',
        'BATCH_MAX_SIZE': '4',
        'BATCH_TARGET_LATENCY_MS': '200',
        'QUALITY_INITIAL_LEVEL': 'LOW',
        'FRAME_QUEUE_MAX_MEMORY_MB': '50',
        'H264_LOW_BITRATE': '500000'
    }
}

def apply_performance_profile(profile_name: str):
    """Применение предустановленного профиля производительности"""
    if profile_name not in PERFORMANCE_PROFILES:
        raise ValueError(f"Неизвестный профиль: {profile_name}")
        
    profile = PERFORMANCE_PROFILES[profile_name]
    
    # Временно устанавливаем переменные окружения
    for key, value in profile.items():
        os.environ[key] = value
        
    # Перезагружаем конфигурацию
    reload_config()
    logger.info(f"Применен профиль производительности: {profile_name}")

# Вспомогательные функции для парсинга переменных окружения
def _get_env_bool(key: str, default: bool) -> bool:
    """Получение boolean значения из переменной окружения"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')

def _get_env_int(key: str, default: int) -> int:
    """Получение int значения из переменной окружения"""
    try:
        value = os.getenv(key)
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def _get_env_float(key: str, default: float) -> float:
    """Получение float значения из переменной окружения"""
    try:
        value = os.getenv(key)
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default
