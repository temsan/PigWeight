"""Configuration constants for the PigWeight application."""

import os
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Performance Profiles
PERFORMANCE_PROFILES = {
    'ULTRA_PERFORMANCE': {
        'TARGET_FPS': '50',  # Оптимально для мощных GPU
        'BATCH_MAX_SIZE': '12',  # Баланс производительности и стабильности
        'BATCH_TARGET_LATENCY_MS': '30',
        'USE_HALF': 'true',
        'IMG_SIZE': '960',  # Максимальное качество
    },
    'RTX_OPTIMIZED': {
        'TARGET_FPS': '20',  # Оптимально для RTX бюджетного класса (2050, 3050)
        'BATCH_MAX_SIZE': '4',  # Безопасно для 3-4GB VRAM
        'BATCH_TARGET_LATENCY_MS': '60',
        'USE_HALF': 'true',
        'IMG_SIZE': '640',  # Оптимально для 3-4GB VRAM
    },
    'BALANCED': {
        'TARGET_FPS': '25',  # Стабильно для большинства GPU
        'BATCH_MAX_SIZE': '6', 
        'BATCH_TARGET_LATENCY_MS': '60',
        'USE_HALF': 'true',
        'IMG_SIZE': '768',  # Компромисс между качеством и производительностью
    },
    'POWER_SAVING': {
        'TARGET_FPS': '20',
        'BATCH_MAX_SIZE': '4',
        'BATCH_TARGET_LATENCY_MS': '100',
        'USE_HALF': 'true',  # Даже слабые современные GPU поддерживают FP16
        'IMG_SIZE': '640',
    },
    'CPU_ONLY': {
        'TARGET_FPS': '15',
        'BATCH_MAX_SIZE': '2',
        'BATCH_TARGET_LATENCY_MS': '200',
        'DEVICE': 'cpu',
        'USE_HALF': 'false',
        'IMG_SIZE': '640',
    }
}

@dataclass
class Config:
    """Centralized configuration for the application."""
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False

    # Model
    MODEL_PATH: str = "models/pig_yolo11-seg.v4.pt"
    PIG_MODEL_PATH: Optional[str] = None
    DETECTION_MODE: str = "pig-only"
    PIG_CLASS_ID: int = 0
    TARGET_CLASS_IDS: str = "20,17,19"
    CONF_THRESHOLD: float = 0.30

    # Device
    DEVICE: str = "auto"
    USE_HALF: bool = True

    # Inference
    IMG_SIZE: int = 960
    BATCH_SIZE: int = 4  # Оптимизировано для стабильности
    MAX_WAIT_MS: int = 50

    # Video
    FPS: int = 25  # Оптимизировано для качества и производительности
    JPEG_QUALITY: int = 90  # Высокое качество
    TARGET_FPS: int = 25  # Оптимизировано для качества и производительности
    CAM_DEFAULT: str = ""  # Отключено - камера недоступна
    CAM_URL: Optional[str] = None

    # Logic
    LINE_LEFT_X: float = 0.25
    LINE_RIGHT_X: float = 0.75
    AVG_WINDOW: int = 20
    FRAME_SKIP: int = 3
    COUNT_WINDOW_SEC: float = 10.0
    COUNT_DECAY_HALFLIFE_SEC: float = 4.0
    COUNT_SOFTMAX_BETA: float = 0.8
    CROSS_COOLDOWN_SEC: float = 1.0

    # Preprocessing
    PREPROCESSING_METHOD: str = "adaptive"
    ANTI_LETTERBOX: bool = False

    def __post_init__(self):
        """Load values from environment variables after initialization."""
        for f in self.__dataclass_fields__:
            env_value = os.getenv(f.upper())
            if env_value is not None:
                # Convert string values to appropriate types
                field_type = self.__annotations__[f]
                if field_type == bool:
                    setattr(self, f, env_value.lower() in ('true', '1', 'yes', 'on'))
                else:
                    try:
                        setattr(self, f, field_type(env_value))
                    except (ValueError, TypeError):
                        pass  # Используем значение по умолчанию
        
        # Auto-detect device if set to "auto"
        if self.DEVICE == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.DEVICE = "cuda:0"
                else:
                    self.DEVICE = "cpu"
            except ImportError:
                self.DEVICE = "cpu"

        # Ensure USE_HALF is False if on CPU
        if self.DEVICE == "cpu":
            self.USE_HALF = False

def detect_optimal_runtime() -> Dict[str, Any]:
    """Определяет оптимальный рантайм на основе доступного оборудования и библиотек."""
    runtime_info = {
        'runtime': 'cpu',
        'device': 'cpu',
        'provider': 'CPUExecutionProvider',
        'use_half': False,
        'profile': 'CPU_ONLY',
        'reasons': []
    }
    
    # Проверяем доступность CUDA/PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else 'Unknown'
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3) if gpu_count > 0 else 0
            
            runtime_info.update({
                'runtime': 'pytorch',
                'device': 'cuda:0',
                'use_half': True,
                'reasons': [f'CUDA GPU доступен: {gpu_name} ({gpu_memory}GB VRAM)']
            })
            
            # Умная логика выбора профиля на основе архитектуры и памяти
            gpu_lower = gpu_name.lower()
            
            # Топовые современные GPU (RTX 4080+, RTX 3080+)
            if any(arch in gpu_lower for arch in ['rtx 4080', 'rtx 4090', 'rtx 3080', 'rtx 3090']) or gpu_memory >= 10:
                runtime_info['profile'] = 'ULTRA_PERFORMANCE'
                runtime_info['reasons'].append('Высокопроизводительная GPU с большим объемом VRAM')
            
            # Современные RTX среднего класса с достаточной памятью
            elif any(arch in gpu_lower for arch in ['rtx 40', 'rtx 30']) and gpu_memory >= 6:
                runtime_info['profile'] = 'ULTRA_PERFORMANCE' 
                runtime_info['reasons'].append('Современная RTX архитектура с достаточной памятью')
            
            # RTX бюджетного сегмента (RTX 2050, 3050, 4050) - 3-4GB VRAM
            elif any(arch in gpu_lower for arch in ['rtx 2050', 'rtx 3050', 'rtx 4050', 'rtx 20']) and gpu_memory <= 4:
                runtime_info['profile'] = 'RTX_OPTIMIZED'
                runtime_info['reasons'].append('RTX бюджетного класса, оптимизация для 3-4GB VRAM')
            
            # RTX среднего класса с хорошей памятью
            elif any(arch in gpu_lower for arch in ['rtx']) and gpu_memory >= 6:
                runtime_info['profile'] = 'ULTRA_PERFORMANCE'
                runtime_info['reasons'].append('RTX архитектура с достаточной памятью')
            
            # Старшие GTX и профессиональные карты
            elif any(name in gpu_lower for name in ['gtx 1070', 'gtx 1080', 'gtx 1660', 'tesla', 'quadro']) or gpu_memory >= 6:
                runtime_info['profile'] = 'BALANCED'
                runtime_info['reasons'].append('Производительная GPU, сбалансированные настройки')
            
            # Средние GPU с достаточной памятью
            elif gpu_memory >= 4:
                runtime_info['profile'] = 'BALANCED'
                runtime_info['reasons'].append('Достаточно VRAM для сбалансированной работы')
            
            # Слабые или старые GPU
            else:
                runtime_info['profile'] = 'POWER_SAVING'
                runtime_info['reasons'].append('Ограниченные ресурсы, энергосберегающий режим')
                    
        else:
            runtime_info['reasons'].append('CUDA недоступен, проверяем ONNX Runtime')
    except ImportError:
        runtime_info['reasons'].append('PyTorch не установлен, проверяем ONNX Runtime')
    
    # Проверяем ONNX Runtime для GPU
    if runtime_info['runtime'] == 'cpu':
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in available_providers:
                runtime_info.update({
                    'runtime': 'onnx-gpu',
                    'device': 'cuda:0',
                    'provider': 'CUDAExecutionProvider',
                    'use_half': True,
                    'profile': 'BALANCED'
                })
                runtime_info['reasons'].append('ONNX Runtime с CUDA поддержкой')
            elif 'DirectMLExecutionProvider' in available_providers and platform.system() == 'Windows':
                runtime_info.update({
                    'runtime': 'onnx-directml',
                    'device': 'dml',
                    'provider': 'DirectMLExecutionProvider',
                    'use_half': False,
                    'profile': 'BALANCED'
                })
                runtime_info['reasons'].append('DirectML поддержка для Windows')
            else:
                runtime_info.update({
                    'runtime': 'onnx-cpu',
                    'provider': 'CPUExecutionProvider',
                    'profile': 'CPU_ONLY'
                })
                runtime_info['reasons'].append(f'ONNX Runtime только с CPU: {available_providers}')
                
        except ImportError:
            runtime_info['reasons'].append('ONNX Runtime недоступен, используем CPU')
    
    # Оптимизация для CPU
    if runtime_info['device'] == 'cpu':
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=False)  # Физические ядра
            memory_gb = psutil.virtual_memory().total // (1024**3)
            
            if cpu_count >= 8 and memory_gb >= 16:
                runtime_info['profile'] = 'BALANCED'
                runtime_info['reasons'].append(f'Мощный CPU: {cpu_count} ядер, {memory_gb}GB RAM')
            elif cpu_count >= 4 and memory_gb >= 8:
                runtime_info['profile'] = 'POWER_SAVING'
                runtime_info['reasons'].append(f'Средний CPU: {cpu_count} ядер, {memory_gb}GB RAM')
            else:
                runtime_info['reasons'].append(f'Слабый CPU: {cpu_count} ядер, {memory_gb}GB RAM')
        except ImportError:
            runtime_info['reasons'].append('psutil недоступен, используем базовые настройки')
    
    return runtime_info

def apply_runtime_optimizations(runtime_info: Dict[str, Any]):
    """Применяет оптимизации на основе выбранного рантайма."""
    # Применяем профиль производительности
    apply_performance_profile(runtime_info['profile'])
    
    # Устанавливаем специфичные настройки рантайма
    os.environ['DEVICE'] = runtime_info['device']
    os.environ['USE_HALF'] = str(runtime_info['use_half']).lower()
    
    if runtime_info['runtime'].startswith('onnx'):
        os.environ['ONNX_PROVIDER'] = runtime_info['provider']
        os.environ['PREFER_ONNX'] = 'true'
    
    logger.info(f"🚀 Выбран оптимальный рантайм: {runtime_info['runtime']}")
    logger.info(f"📱 Устройство: {runtime_info['device']}")
    logger.info(f"⚡ Профиль: {runtime_info['profile']}")
    for reason in runtime_info['reasons']:
        logger.info(f"💡 {reason}")

def apply_performance_profile(profile_name: str):
    """Применяет профиль производительности, устанавливая переменные окружения."""
    profile = PERFORMANCE_PROFILES.get(profile_name.upper())
    if not profile:
        raise ValueError(f"Unknown performance profile: {profile_name}")
    
    for key, value in profile.items():
        os.environ[key] = value
    
    logger.info(f"Applied performance profile: {profile_name.upper()}")
    # The config will be re-read on next instantiation.
    """Applies a performance profile by setting environment variables."""
    profile = PERFORMANCE_PROFILES.get(profile_name.upper())
    if not profile:
        raise ValueError(f"Unknown performance profile: {profile_name}")
    
    for key, value in profile.items():
        os.environ[key] = value
    
    logger.info(f"Applied performance profile: {profile_name.upper()}")
    # The config will be re-read on next instantiation.

# Global config instance
CONFIG = Config()

def get_config() -> Config:
    """Returns the global config instance."""
    return CONFIG

def setup_logging(debug: bool = False) -> logging.Logger:
    """Sets up unified logging for the project."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # В production режиме показываем только WARNING и ERROR
    level = logging.DEBUG if debug else logging.WARNING
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler - в файл записываем все уровни для отладки
    file_level = logging.DEBUG if debug else logging.INFO
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    root_logger.addHandler(file_handler)
    
    # Подавляем внешние библиотеки
    external_loggers = [
        "aiortc", "aioice", "av", "uvicorn", "uvicorn.access", "uvicorn.error", 
        "fastapi", "websockets", "asyncio", "multipart", "httpx", "starlette"
    ]
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR if not debug else logging.WARNING)
    
    # В production режиме подавляем INFO сообщения наших модулей
    if not debug:
        our_loggers = [
            "pigweight", "core", "api", "inference", "stream",
            "core.processor", "core.frame_broker", "core.dynamic_batcher",
            "core.performance_monitor", "core.priority_frame_queue",
            "core.demo_generator", "core.h264_direct_track",
            "api.app_backup", "api.endpoints.system",
            "api.webrtc", "api.av_worker", "api.app_modular",
            "perf.api", "perf.webrtc", "perf.results_store", "perf.frame_broker"
        ]
        for logger_name in our_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # Для отладки оставляем INFO логи в критичных модулях
        critical_loggers = ["services.model_adapter", "api.app"]
        for logger_name in critical_loggers:
            logging.getLogger(logger_name).setLevel(logging.INFO)
    
    return logging.getLogger("pigweight")
