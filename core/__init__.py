"""
Core модуль системы PigWeight
Содержит основную логику обработки видео и анализа
"""

from .processor import (
    UnifiedVideoProcessor, 
    ProcessingOptions, 
    FrameResult, 
    get_processor, 
    remove_processor, 
    reset_processors
)

from .pipeline import (
    VideoPipeline,
    VideoCapture,
    LineAnalyzer,
    ActDetector,
    WeighingAct,
    CrossingEvent,
    create_pipeline
)

__all__ = [
    # Processor
    'UnifiedVideoProcessor',
    'ProcessingOptions', 
    'FrameResult',
    'get_processor',
    'remove_processor',
    'reset_processors',
    # Pipeline
    'VideoPipeline',
    'VideoCapture',
    'LineAnalyzer',
    'ActDetector',
    'WeighingAct',
    'CrossingEvent',
    'create_pipeline'
]