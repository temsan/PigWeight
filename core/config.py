"""Configuration constants for the PigWeight application."""

import os
from typing import Dict, Any

# Model configuration
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
MODEL_PATH = "models/yolo11n.pt"
ONNX_PATH = "models/yolo11n.onnx"

# Default configuration values
DEFAULT_CONFIG = {
    # Model settings
    "MODEL_PATH": "models/pig_yolo11-seg.v4.pt",
    "DETECTION_MODE": "pig-only",
    "PIG_CLASS_ID": 0,
    "CONF_THRESHOLD": 0.30,

    # Device settings
    "DEVICE": "cuda:0",
    "USE_HALF": True,

    # Inference settings
    "IMG_SIZE": 960,
    "BATCH_SIZE": 4,
    "MAX_WAIT_MS": 50,

    # Broker settings
    "FRAME_BROKER_CACHE": 16,
    "RESULTS_TTL_SECONDS": 30,
    "BROADCAST_MIN_INTERVAL": 0.05,  # 20 FPS max for WebSocket updates

    # Server settings
    "HOST": "0.0.0.0",
    "PORT": 8000,
    "DEBUG": False,
    "RELOAD": False,

    # Video settings
    "FPS": 12,
    "JPEG_QUALITY": 80,
    "TARGET_FPS": 12,

    # Lines for counting
    "LINE_LEFT_X": 0.25,
    "LINE_RIGHT_X": 0.75,
}

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables with defaults."""
    config = DEFAULT_CONFIG.copy()

    # Override with environment variables
    for key in config:
        env_value = os.getenv(key)
        if env_value is not None:
            # Convert string values to appropriate types
            if isinstance(config[key], bool):
                config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(config[key], int):
                try:
                    config[key] = int(env_value)
                except ValueError:
                    pass
            elif isinstance(config[key], float):
                try:
                    config[key] = float(env_value)
                except ValueError:
                    pass
            else:
                config[key] = env_value

    # Auto-detect CUDA availability
    try:
        import torch
        if config["DEVICE"].startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            config["DEVICE"] = "cpu"
            config["USE_HALF"] = False
    except ImportError:
        config["DEVICE"] = "cpu"
        config["USE_HALF"] = False

    return config

# Load configuration on import
CONFIG = load_config()
