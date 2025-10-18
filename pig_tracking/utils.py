"""
Утилиты для системы отслеживания свиней
"""

import cv2
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_video_info(video_path: Path) -> dict:
    """
    Получает информацию о видеофайле
    
    Args:
        video_path: Путь к видеофайлу
        
    Returns:
        Словарь с информацией о видео
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {
                'error': 'Не удалось открыть видео',
                'path': video_path
            }
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration_sec = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_sec': duration_sec,
            'duration_str': format_duration(duration_sec)
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения информации о видео {video_path}: {e}")
        return {
            'error': str(e),
            'path': video_path
        }

def format_duration(seconds: float) -> str:
    """
    Форматирует длительность в читаемый вид
    
    Args:
        seconds: Длительность в секундах
        
    Returns:
        Строка вида "HH:MM:SS" или "MM:SS"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def format_file_size(bytes_size: int) -> str:
    """
    Форматирует размер файла в читаемый вид
    
    Args:
        bytes_size: Размер в байтах
        
    Returns:
        Строка вида "1.5 GB" или "500 MB"
    """
    mb = bytes_size / (1024 * 1024)
    gb = mb / 1024
    
    if gb >= 1:
        return f"{gb:.1f} GB"
    else:
        return f"{mb:.0f} MB"

def estimate_processing_time(frame_count: int, fps_processing: float = 10.0) -> str:
    """
    Оценивает время обработки видео
    
    Args:
        frame_count: Количество кадров
        fps_processing: Скорость обработки (кадров в секунду)
        
    Returns:
        Строка с оценкой времени
    """
    if fps_processing <= 0:
        return "неизвестно"
    
    seconds = frame_count / fps_processing
    return format_duration(seconds)

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Создает текстовый прогресс-бар
    
    Args:
        current: Текущее значение
        total: Максимальное значение
        width: Ширина бара
        
    Returns:
        Строка с прогресс-баром
    """
    if total == 0:
        percent = 0
    else:
        percent = (current / total) * 100
    
    filled = int(width * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (width - filled)
    
    return f"[{bar}] {percent:.1f}%"

def normalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """
    Нормализует координаты к диапазону [0, 1]
    
    Args:
        x, y: Координаты в пикселях
        width, height: Размеры изображения
        
    Returns:
        Нормализованные координаты (0-1)
    """
    norm_x = x / width if width > 0 else 0
    norm_y = y / height if height > 0 else 0
    
    return (
        max(0.0, min(1.0, norm_x)),
        max(0.0, min(1.0, norm_y))
    )

def denormalize_coordinates(norm_x: float, norm_y: float, width: int, height: int) -> Tuple[int, int]:
    """
    Денормализует координаты из диапазона [0, 1] в пиксели
    
    Args:
        norm_x, norm_y: Нормализованные координаты (0-1)
        width, height: Размеры изображения
        
    Returns:
        Координаты в пикселях
    """
    x = int(norm_x * width)
    y = int(norm_y * height)
    
    return (
        max(0, min(width - 1, x)),
        max(0, min(height - 1, y))
    )

class ProgressTracker:
    """Отслеживание прогресса обработки"""
    
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
        self.update_interval = 1.0  # Обновлять не чаще раза в секунду
    
    def update(self, increment: int = 1) -> Optional[str]:
        """
        Обновляет прогресс
        
        Args:
            increment: На сколько увеличить счетчик
            
        Returns:
            Строка с прогрессом или None если еще рано обновлять
        """
        self.current += increment
        now = datetime.now()
        
        # Обновляем не чаще чем раз в секунду
        if (now - self.last_update).total_seconds() < self.update_interval:
            if self.current < self.total:  # Но всегда показываем 100%
                return None
        
        self.last_update = now
        
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed = (now - self.start_time).total_seconds()
        
        if self.current > 0 and elapsed > 0:
            fps = self.current / elapsed
            remaining_frames = self.total - self.current
            eta_sec = remaining_frames / fps if fps > 0 else 0
            eta_str = format_duration(eta_sec)
        else:
            fps = 0
            eta_str = "неизвестно"
        
        progress_bar = create_progress_bar(self.current, self.total)
        
        return f"{progress_bar} | {self.current}/{self.total} кадров | {fps:.1f} fps | ETA: {eta_str}"