"""
Модуль подсчета пересечений линий.
Адаптирован из VideoStream._update_line_counters() в api/app.py
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CrossingEvent:
    """Событие пересечения линии"""
    track_id: int
    side: str  # 'left' или 'right'
    mode: str  # 'enter' или 'exit'
    x: float  # Позиция линии
    y: float  # Y-координата пересечения
    timestamp: float


class CrossingCounter:
    """
    Подсчет проходов через вертикальные линии с интерполяцией Y-координат.
    
    Параметры:
    - line_left_x: позиция левой линии (0.0-1.0)
    - line_right_x: позиция правой линии (0.0-1.0)
    - cooldown_sec: минимальный интервал между событиями для одного трека
    """
    
    def __init__(
        self,
        line_left_x: float = 0.25,
        line_right_x: float = 0.75,
        cooldown_sec: float = 1.0
    ):
        self.line_left_x = float(line_left_x)
        self.line_right_x = float(line_right_x)
        self.cooldown_sec = float(cooldown_sec)
        
        # Счетчики потоков
        self.left_in = 0  # Вход слева
        self.right_in = 0  # Вход справа
        self.total_crossings = 0
        
        # Направленные потоки для UI
        self.left_flow = 0  # +enter_left, -exit_left
        self.right_flow = 0  # +exit_right, -enter_right
        
        # Состояние треков
        self._track_prev_x: Dict[int, float] = {}
        self._track_prev_y: Dict[int, float] = {}
        self._track_is_inside: Dict[int, bool] = {}
        self._track_last_side_time: Dict[Tuple[int, str], float] = {}
        
        # Недавние пересечения для визуализации
        self.recent_crossings: List[CrossingEvent] = []
        
        logger.info(f"CrossingCounter инициализирован: L={line_left_x:.3f}, R={line_right_x:.3f}, cooldown={cooldown_sec}s")
    
    def update_line_positions(self, left_x: float, right_x: float):
        """Обновляет позиции линий"""
        self.line_left_x = float(left_x)
        self.line_right_x = float(right_x)
        logger.debug(f"Обновлены позиции линий: L={self.line_left_x:.3f}, R={self.line_right_x:.3f}")
    
    def process_tracks(
        self,
        track_ids: List[int],
        centers_x: List[float],
        centers_y: List[float]
    ) -> List[CrossingEvent]:
        """
        Обрабатывает треки и возвращает новые события пересечений.
        
        Args:
            track_ids: ID треков
            centers_x: X-координаты центров (нормализованные 0-1)
            centers_y: Y-координаты центров (нормализованные 0-1)
            
        Returns:
            Список новых событий пересечений
        """
        now = time.time()
        new_events: List[CrossingEvent] = []
        
        L = self.line_left_x
        R = self.line_right_x
        
        for tid, cx, cy in zip(track_ids, centers_x, centers_y):
            if tid is None:
                continue
            
            prev_x = self._track_prev_x.get(tid)
            prev_y = self._track_prev_y.get(tid)
            
            prev_inside = self._track_is_inside.get(tid, (prev_x is not None and L <= prev_x <= R))
            cur_inside = L <= cx <= R
            
            if prev_x is not None and prev_y is not None:
                # Проверяем общий cooldown для трека
                track_cooldown_key = f"track_{tid}"
                if now - self._track_last_side_time.get(track_cooldown_key, 0.0) < self.cooldown_sec:
                    # Обновляем позицию и пропускаем событие
                    self._track_prev_x[tid] = cx
                    self._track_prev_y[tid] = cy
                    self._track_is_inside[tid] = cur_inside
                    continue
                
                # Вход в зону между линиями
                if (not prev_inside) and cur_inside:
                    # Вход слева (свинья идет вправо)
                    if prev_x < L <= cx:
                        y_at = self._interpolate_y(prev_x, prev_y, cx, cy, L)
                        event = CrossingEvent(
                            track_id=tid,
                            side='left',
                            mode='enter',
                            x=L,
                            y=y_at,
                            timestamp=now
                        )
                        new_events.append(event)
                        self.recent_crossings.append(event)
                        
                        self.left_in += 1
                        self.total_crossings += 1
                        self.left_flow += 1
                        
                        key = (tid, 'enter_left')
                        self._track_last_side_time[key] = now
                        self._track_last_side_time[track_cooldown_key] = now
                        
                        logger.info(f"🔵 L={L:.3f} y={y_at:.3f} t{tid} ←IN ({self.left_in})")
                    
                    # Вход справа (свинья идет влево)
                    elif prev_x > R >= cx:
                        y_at = self._interpolate_y(prev_x, prev_y, cx, cy, R)
                        event = CrossingEvent(
                            track_id=tid,
                            side='right',
                            mode='enter',
                            x=R,
                            y=y_at,
                            timestamp=now
                        )
                        new_events.append(event)
                        self.recent_crossings.append(event)
                        
                        self.right_in += 1
                        self.total_crossings += 1
                        self.right_flow -= 1
                        
                        key = (tid, 'enter_right')
                        self._track_last_side_time[key] = now
                        self._track_last_side_time[track_cooldown_key] = now
                        
                        logger.info(f"🟢 R={R:.3f} y={y_at:.3f} t{tid} IN→ ({self.right_in})")
                
                # Выход из зоны между линиями
                elif prev_inside and (not cur_inside):
                    # Выход слева (свинья идет влево)
                    if cx < L <= prev_x:
                        y_at = self._interpolate_y(prev_x, prev_y, cx, cy, L)
                        event = CrossingEvent(
                            track_id=tid,
                            side='left',
                            mode='exit',
                            x=L,
                            y=y_at,
                            timestamp=now
                        )
                        new_events.append(event)
                        self.recent_crossings.append(event)
                        
                        self.left_flow -= 1
                        
                        key = (tid, 'exit_left')
                        self._track_last_side_time[key] = now
                        self._track_last_side_time[track_cooldown_key] = now
                        
                        logger.info(f"🔵 L={L:.3f} y={y_at:.3f} t{tid} OUT←")
                    
                    # Выход справа (свинья идет вправо)
                    elif cx > R >= prev_x:
                        y_at = self._interpolate_y(prev_x, prev_y, cx, cy, R)
                        event = CrossingEvent(
                            track_id=tid,
                            side='right',
                            mode='exit',
                            x=R,
                            y=y_at,
                            timestamp=now
                        )
                        new_events.append(event)
                        self.recent_crossings.append(event)
                        
                        self.right_flow += 1
                        
                        key = (tid, 'exit_right')
                        self._track_last_side_time[key] = now
                        self._track_last_side_time[track_cooldown_key] = now
                        
                        logger.info(f"🟢 R={R:.3f} y={y_at:.3f} t{tid} →OUT")
            
            # Обновляем состояние трека
            self._track_prev_x[tid] = cx
            self._track_prev_y[tid] = cy
            self._track_is_inside[tid] = cur_inside
        
        # Ограничиваем размер списка недавних пересечений
        if len(self.recent_crossings) > 100:
            self.recent_crossings = self.recent_crossings[-100:]
        
        return new_events
    
    def _interpolate_y(
        self,
        px: float,
        py: float,
        qx: float,
        qy: float,
        lx: float
    ) -> float:
        """
        Интерполяция Y-координаты в точке пересечения линии.
        
        Args:
            px, py: предыдущая позиция
            qx, qy: текущая позиция
            lx: X-координата линии
            
        Returns:
            Интерполированная Y-координата
        """
        try:
            # Проверяем, что линия не вертикальная
            if abs(float(qx) - float(px)) < 1e-6:
                return float(qy)
            
            # Линейная интерполяция
            t = (float(lx) - float(px)) / (float(qx) - float(px))
            interpolated_y = float(py) + t * (float(qy) - float(py))
            
            # Ограничиваем результат диапазоном [0, 1]
            return max(0.0, min(1.0, interpolated_y))
        except Exception as e:
            logger.warning(f"Ошибка интерполяции Y: {e}, используем текущую Y={qy}")
            return float(qy)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику подсчета"""
        return {
            'left_in': self.left_in,
            'right_in': self.right_in,
            'total_crossings': self.total_crossings,
            'left_flow': self.left_flow,
            'right_flow': self.right_flow,
            'active_tracks': len(self._track_prev_x),
            'recent_crossings_count': len(self.recent_crossings)
        }
    
    def reset(self):
        """Сбрасывает все счетчики и состояние"""
        self.left_in = 0
        self.right_in = 0
        self.total_crossings = 0
        self.left_flow = 0
        self.right_flow = 0
        self._track_prev_x.clear()
        self._track_prev_y.clear()
        self._track_is_inside.clear()
        self._track_last_side_time.clear()
        self.recent_crossings.clear()
        logger.info("CrossingCounter сброшен")
