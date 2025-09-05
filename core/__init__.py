"""
Core модуль системы PigWeight
Содержит основную логику обработки видео и анализа
"""

from .processor import UnifiedVideoProcessor, ProcessingOptions, FrameResult, get_processor, reset_processor

__all__ = [
    'UnifiedVideoProcessor',
    'ProcessingOptions', 
    'FrameResult',
    'get_processor',
    'reset_processor'
]