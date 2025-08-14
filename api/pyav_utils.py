import logging
import time
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PyAV removed from runtime in this build — provide minimal stubs so callers don't crash.
# These stubs intentionally do not attempt any decoding; they only log and return safe defaults.

DEFAULT_TIMEOUT = 3.0


def pyav_open_container(video_path: str, timeout: float = DEFAULT_TIMEOUT):
    """
    Stub: PyAV not available. Returns (None, None).
    """
    logger.info(f"[pyav_stub] PyAV disabled — cannot open container: {video_path}")
    return None, None


def pyav_read_frame(container, stream, seek_time: float = None, max_attempts: int = 1):
    """
    Stub: PyAV disabled — always return None.
    """
    logger.debug("[pyav_stub] pyav_read_frame called but PyAV is disabled")
    return None


def pyav_get_meta(container, stream):
    """
    Return safe default metadata when PyAV is disabled.
    """
    return {"fps": 25.0, "frame_count": 0, "duration": 0.0}


def pyav_close_container(container):
    """
    Safe no-op close for stub.
    """
    try:
        # nothing to close in stub
        pass
    except Exception:
        pass


def pyav_seek_read_jpeg(video_path: str, t: float, jpeg_quality: int = 80) -> Optional[bytes]:
    """
    Stub: seeking with PyAV unavailable — return None.
    """
    logger.info(f"[pyav_stub] pyav_seek_read_jpeg unavailable for {video_path} at t={t}")
    return None