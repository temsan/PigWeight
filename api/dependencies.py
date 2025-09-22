"""
Shared dependencies for API endpoints
"""

# This module will be populated with shared dependencies
# to avoid circular imports between app.py and endpoints

# Global variables that will be set by app.py
STREAM_MANAGER = None
TARGET_FPS = None
FileStream = None
perf_logger = None
av_meta = None

def init_dependencies(stream_manager, target_fps, file_stream_class, perf_log, av_meta_func):
    """Initialize shared dependencies"""
    global STREAM_MANAGER, TARGET_FPS, FileStream, perf_logger, av_meta
    STREAM_MANAGER = stream_manager
    TARGET_FPS = target_fps
    FileStream = file_stream_class
    perf_logger = perf_log
    av_meta = av_meta_func