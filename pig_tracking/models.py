"""
Модели данных для системы отслеживания свиней
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

@dataclass
class Detection:
    """Детекция свиньи на кадре"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    mask: Optional[List[Tuple[float, float]]] = None  # Полигон маски
    centroid: Optional[Tuple[float, float]] = None  # Центр масс

@dataclass
class TrackedPig:
    """Отслеживаемая свинья"""
    id: int
    bbox: List[float]
    centroid: Tuple[float, float]
    age: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class CrossingEvent:
    """Событие пересечения линии"""
    pig_id: int
    direction: str  # "left" or "right"
    timestamp: datetime
    line_x: float
    line_y: float
    weight_estimate: Optional[float] = None
    act_id: Optional[int] = None
    stream_id: Optional[str] = None

@dataclass
class WeighingAct:
    """Акт взвешивания"""
    started_at: datetime
    ended_at: datetime
    duration_sec: float
    left_count: int
    right_count: int
    peak_count: int
    total_weight: Optional[float] = None
    avg_weight: Optional[float] = None
    stream_id: Optional[str] = None
    video_file: Optional[str] = None
    id: Optional[int] = None
    crossings: List[CrossingEvent] = field(default_factory=list)

@dataclass
class ProcessingStats:
    """Статистика обработки"""
    total_frames: int = 0
    processed_frames: int = 0
    detected_pigs: int = 0
    total_crossings: int = 0
    acts_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        """Процент обработки"""
        if self.total_frames == 0:
            return 0.0
        return (self.processed_frames / self.total_frames) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Прошедшее время в секундах"""
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def fps(self) -> float:
        """Скорость обработки (кадров в секунду)"""
        if self.elapsed_time == 0:
            return 0.0
        return self.processed_frames / self.elapsed_time