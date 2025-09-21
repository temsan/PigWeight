"""
Модульные API endpoints для PigWeight
"""

from .health import router as health_router
from .video import router as video_router
from .stream import router as stream_router
from .websocket import router as websocket_router
from .files import router as files_router

__all__ = [
    'health_router',
    'video_router', 
    'stream_router',
    'websocket_router',
    'files_router'
]