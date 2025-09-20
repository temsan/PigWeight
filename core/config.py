"""Configuration constants for the PigWeight application."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Performance Profiles
PERFORMANCE_PROFILES = {
    'ULTRA_PERFORMANCE': {
        'TARGET_FPS': '120',
        'BATCH_MAX_SIZE': '32',
        'BATCH_TARGET_LATENCY_MS': '25',
        'USE_HALF': 'true',
    },
    'BALANCED': {
        'TARGET_FPS': '60',
        'BATCH_MAX_SIZE': '16',
        'BATCH_TARGET_LATENCY_MS': '50',
        'USE_HALF': 'true',
    },
    'POWER_SAVING': {
        'TARGET_FPS': '30',
        'BATCH_MAX_SIZE': '8',
        'BATCH_TARGET_LATENCY_MS': '100',
        'USE_HALF': 'false',
    },
    'CPU_ONLY': {
        'TARGET_FPS': '20',
        'BATCH_MAX_SIZE': '4',
        'BATCH_TARGET_LATENCY_MS': '150',
        'DEVICE': 'cpu',
        'USE_HALF': 'false',
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
    BATCH_SIZE: int = 4
    MAX_WAIT_MS: int = 50

    # Video
    FPS: int = 12
    JPEG_QUALITY: int = 80
    TARGET_FPS: int = 12
    CAM_DEFAULT: str = "rtsp://admin:Qwerty.123@10.15.6.24/1/1"
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
    USE_OPTIMIZED_PREPROCESSING: bool = True
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
                        logger.warning(f"Could not cast env var {f.upper()} to {field_type}. Using default.")
        
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

def apply_performance_profile(profile_name: str):
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
    
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Silence overly verbose loggers
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("av").setLevel(logging.WARNING)
    
    return logging.getLogger("pigweight")