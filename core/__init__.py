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

__all__ = [
    'UnifiedVideoProcessor',
    'ProcessingOptions', 
    'FrameResult',
    'get_processor',
    'remove_processor',
    'reset_processors'
]