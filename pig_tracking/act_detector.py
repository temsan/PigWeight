"""
Модуль обнаружения актов взвешивания.
Определяет начало и конец актов на основе активности пересечений.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WeighingAct:
    """Акт взвешивания"""
    act_id: int
    started_at: float
    ended_at: Optional[float] = None
    left_count: int = 0
    right_count: int = 0
    peak_count: int = 0
    seen_labels: set = field(default_factory=set)
    crossings: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Длительность акта в секундах"""
        if self.ended_at is None:
            return time.time() - self.started_at
        return self.ended_at - self.started_at
    
    @property
    def is_active(self) -> bool:
        """Активен ли акт"""
        return self.ended_at is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует акт в словарь"""
        return {
            'act_id': self.act_id,
            'started_at': self.started_at,
            'started_at_iso': datetime.fromtimestamp(self.started_at).isoformat(),
            'ended_at': self.ended_at,
            'ended_at_iso': datetime.fromtimestamp(self.ended_at).isoformat() if self.ended_at else None,
            'duration': self.duration,
            'left_count': self.left_count,
            'right_count': self.right_count,
            'peak_count': self.peak_count,
            'seen_total': len(self.seen_labels),
            'is_active': self.is_active,
            'crossings_count': len(self.crossings)
        }


class ActDetector:
    """
    Детектор актов взвешивания.
    
    Параметры:
    - min_pigs_for_act: минимальное количество проходов для начала акта
    - max_interval_sec: максимальный интервал без активности для завершения акта
    """
    
    def __init__(
        self,
        min_pigs_for_act: int = 3,
        max_interval_sec: float = 30.0
    ):
        self.min_pigs_for_act = int(min_pigs_for_act)
        self.max_interval_sec = float(max_interval_sec)
        
        self.current_act: Optional[WeighingAct] = None
        self.completed_acts: List[WeighingAct] = []
        self._next_act_id = 1
        
        # Временное окно для подсчета активности
        self._recent_crossings_window: List[float] = []
        self._last_crossing_time: float = 0.0
        
        logger.info(
            f"ActDetector инициализирован: min_pigs={min_pigs_for_act}, "
            f"max_interval={max_interval_sec}s"
        )
    
    def update(
        self,
        crossings: List[Any],
        current_count: int,
        timestamp: Optional[float] = None
    ) -> Optional[WeighingAct]:
        """
        Обновляет состояние детектора на основе новых пересечений.
        
        Args:
            crossings: список событий пересечений
            current_count: текущее количество объектов в зоне
            timestamp: временная метка (по умолчанию time.time())
            
        Returns:
            Завершенный акт, если акт был завершен, иначе None
        """
        now = timestamp or time.time()
        completed_act = None
        
        # Обновляем окно недавних пересечений
        if crossings:
            self._last_crossing_time = now
            for crossing in crossings:
                self._recent_crossings_window.append(now)
        
        # Очищаем старые пересечения из окна (старше 60 секунд)
        cutoff_time = now - 60.0
        self._recent_crossings_window = [
            t for t in self._recent_crossings_window if t > cutoff_time
        ]
        
        # Проверяем условия для начала нового акта
        if self.current_act is None:
            recent_count = len(self._recent_crossings_window)
            if recent_count >= self.min_pigs_for_act:
                self._start_new_act(now)
                logger.info(
                    f"🎬 Начат новый акт #{self._next_act_id - 1}: "
                    f"{recent_count} проходов за последнюю минуту"
                )
        
        # Обновляем текущий акт
        if self.current_act is not None:
            # Добавляем новые пересечения
            for crossing in crossings:
                crossing_data = {
                    'track_id': crossing.track_id,
                    'side': crossing.side,
                    'mode': crossing.mode,
                    'x': crossing.x,
                    'y': crossing.y,
                    'timestamp': crossing.timestamp
                }
                self.current_act.crossings.append(crossing_data)
                
                # Обновляем счетчики
                if crossing.mode == 'enter':
                    if crossing.side == 'left':
                        self.current_act.left_count += 1
                    elif crossing.side == 'right':
                        self.current_act.right_count += 1
                
                # Добавляем в список увиденных
                self.current_act.seen_labels.add(crossing.track_id)
            
            # Обновляем пиковое значение
            if current_count > self.current_act.peak_count:
                self.current_act.peak_count = current_count
            
            # Проверяем условия завершения акта
            time_since_last_crossing = now - self._last_crossing_time
            if time_since_last_crossing > self.max_interval_sec:
                completed_act = self._complete_current_act(now)
                logger.info(
                    f"🏁 Завершен акт #{completed_act.act_id}: "
                    f"длительность={completed_act.duration:.1f}s, "
                    f"left={completed_act.left_count}, right={completed_act.right_count}, "
                    f"peak={completed_act.peak_count}, seen={len(completed_act.seen_labels)}"
                )
        
        return completed_act
    
    def _start_new_act(self, timestamp: float):
        """Начинает новый акт взвешивания"""
        self.current_act = WeighingAct(
            act_id=self._next_act_id,
            started_at=timestamp
        )
        self._next_act_id += 1
    
    def _complete_current_act(self, timestamp: float) -> WeighingAct:
        """Завершает текущий акт и возвращает его"""
        if self.current_act is None:
            raise ValueError("Нет активного акта для завершения")
        
        self.current_act.ended_at = timestamp
        self.completed_acts.append(self.current_act)
        
        completed = self.current_act
        self.current_act = None
        
        return completed
    
    def force_complete_current_act(self) -> Optional[WeighingAct]:
        """Принудительно завершает текущий акт (для конца видео)"""
        if self.current_act is None:
            return None
        
        return self._complete_current_act(time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику детектора"""
        stats = {
            'completed_acts_count': len(self.completed_acts),
            'current_act': self.current_act.to_dict() if self.current_act else None,
            'recent_crossings_in_window': len(self._recent_crossings_window),
            'time_since_last_crossing': time.time() - self._last_crossing_time if self._last_crossing_time > 0 else None
        }
        
        if self.completed_acts:
            stats['completed_acts'] = [act.to_dict() for act in self.completed_acts]
        
        return stats
    
    def reset(self):
        """Сбрасывает состояние детектора"""
        if self.current_act is not None:
            logger.warning(f"Сброс детектора с активным актом #{self.current_act.act_id}")
        
        self.current_act = None
        self.completed_acts.clear()
        self._next_act_id = 1
        self._recent_crossings_window.clear()
        self._last_crossing_time = 0.0
        logger.info("ActDetector сброшен")
