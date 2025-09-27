"""
Сервис для журналирования ключевых событий в системе PigWeight.
Записывает события пересечения линий, пики количества и всплески активности.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EventData:
    """Данные события для журналирования"""
    event_id: str
    stream_id: str
    event_type: str  # 'line_crossing', 'peak_count', 'activity_spike'
    timestamp: float
    pig_count: int
    confidence: float
    frame_path: Optional[str] = None
    side: Optional[str] = None
    movement: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class EventLogger:
    """
    Сервис для журналирования ключевых событий в видеопотоке.
    
    Отслеживает:
    - Пересечения линий детекции
    - Пиковые значения количества свиней
    - Всплески активности (резкий рост количества)
    """
    
    def __init__(self, 
                 events_dir: str = "records/events",
                 frames_dir: str = "records/frames",
                 max_events_per_stream: int = 1000,
                 spike_threshold: int = 5,
                 spike_window_sec: float = 10.0):
        
        self.events_dir = Path(events_dir)
        self.frames_dir = Path(frames_dir)
        self.max_events_per_stream = max_events_per_stream
        self.spike_threshold = spike_threshold
        self.spike_window_sec = spike_window_sec
        
        # Создаем директории
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Хранилище событий по потокам
        self.stream_events: Dict[str, deque] = {}
        self.stream_peaks: Dict[str, int] = {}
        self.stream_history: Dict[str, deque] = {}  # История для отслеживания всплесков
        
        # Счетчики для генерации ID
        self.event_counter = 0
        
        logger.info(f"EventLogger initialized: events_dir={events_dir}, frames_dir={frames_dir}")
    
    def _generate_event_id(self) -> str:
        """Генерирует уникальный ID события"""
        self.event_counter += 1
        timestamp = int(time.time() * 1000)
        return f"evt_{timestamp}_{self.event_counter:06d}"
    
    def _save_frame(self, frame: np.ndarray, event_id: str) -> Optional[str]:
        """Сохраняет кадр на диск и возвращает путь к файлу (оптимизировано)"""
        try:
            # Для производительности сохраняем кадры только для важных событий
            # или если включен DEBUG режим
            import os
            if not os.getenv('DEBUG') and not os.getenv('SAVE_FRAMES'):
                return None
                
            frame_filename = f"{event_id}.jpg"
            frame_path = self.frames_dir / frame_filename
            
            # Оптимизированное сохранение кадра
            success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            
            if success:
                logger.debug(f"Frame saved: {frame_path}")
                return str(frame_path)
            else:
                logger.debug(f"Failed to save frame: {frame_path}")
                return None
                
        except Exception as e:
            logger.debug(f"Error saving frame for event {event_id}: {e}")
            return None
    
    def _save_event_to_file(self, stream_id: str, event: EventData):
        """Сохраняет событие в JSON файл"""
        try:
            events_file = self.events_dir / f"{stream_id}_events.jsonl"
            
            # Добавляем событие в JSONL формате (одна строка = одно событие)
            with open(events_file, 'a', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Error saving event to file: {e}")
    
    def _detect_activity_spike(self, stream_id: str, current_count: int) -> bool:
        """Определяет всплеск активности"""
        if stream_id not in self.stream_history:
            self.stream_history[stream_id] = deque(maxlen=100)
        
        history = self.stream_history[stream_id]
        current_time = time.time()
        
        # Добавляем текущее значение
        history.append((current_time, current_count))
        
        # Проверяем всплеск: рост более чем на spike_threshold за spike_window_sec
        if len(history) < 2:
            return False
        
        # Находим минимальное значение в окне
        window_start = current_time - self.spike_window_sec
        min_count_in_window = current_count
        
        for timestamp, count in reversed(history):
            if timestamp < window_start:
                break
            min_count_in_window = min(min_count_in_window, count)
        
        # Проверяем превышение порога
        growth = current_count - min_count_in_window
        return growth >= self.spike_threshold
    
    async def log_line_crossing(self, 
                               stream_id: str, 
                               pig_count: int, 
                               confidence: float,
                               frame: Optional[np.ndarray] = None,
                               metadata: Optional[Dict[str, Any]] = None):
        """Фиксация события пересечения контрольной линии."""

        event_id = self._generate_event_id()
        frame_path = None

        if frame is not None:
            frame_path = self._save_frame(frame, event_id)

        meta = dict(metadata or {})
        side = meta.get("side")
        direction = meta.get("direction")
        movement = None
        if side and direction:
            if side == "left" and direction == "enter":
                movement = "left_to_right"
            elif side == "left" and direction == "exit":
                movement = "right_to_left"
            elif side == "right" and direction == "enter":
                movement = "right_to_left"
            elif side == "right" and direction == "exit":
                movement = "left_to_right"
        if movement:
            meta.setdefault("movement", movement)
            meta.setdefault("direction_label", movement)

        event = EventData(
            event_id=event_id,
            stream_id=stream_id,
            event_type="line_crossing",
            timestamp=time.time(),
            pig_count=pig_count,
            confidence=confidence,
            frame_path=frame_path,
            side=side,
            movement=movement,
            metadata=meta
        )

        # Буферизуем в памяти
        if stream_id not in self.stream_events:
            self.stream_events[stream_id] = deque(maxlen=self.max_events_per_stream)

        self.stream_events[stream_id].append(event)

        # Сохраняем на диск в потоке ввода-вывода
        await asyncio.to_thread(self._save_event_to_file, stream_id, event)

        if movement:
            logger.info(f"Line crossing logged: stream={stream_id}, movement={movement}, count={pig_count}")
        else:
            logger.info(f"Line crossing logged: stream={stream_id}, count={pig_count}, confidence={confidence:.2f}")

    async def log_peak_count(self, 
                            stream_id: str, 
                            pig_count: int, 
                            confidence: float,
                            frame: Optional[np.ndarray] = None,
                            metadata: Optional[Dict[str, Any]] = None):
        """Логирует событие достижения пикового количества"""
        
        # Проверяем, является ли это новым пиком
        current_peak = self.stream_peaks.get(stream_id, 0)
        if pig_count <= current_peak:
            return  # Не новый пик
        
        # Обновляем пик
        self.stream_peaks[stream_id] = pig_count
        
        event_id = self._generate_event_id()
        frame_path = None
        
        if frame is not None:
            frame_path = self._save_frame(frame, event_id)
        
        event = EventData(
            event_id=event_id,
            stream_id=stream_id,
            event_type='peak_count',
            timestamp=time.time(),
            pig_count=pig_count,
            confidence=confidence,
            frame_path=frame_path,
            metadata=metadata or {'previous_peak': current_peak}
        )
        
        # Добавляем в память
        if stream_id not in self.stream_events:
            self.stream_events[stream_id] = deque(maxlen=self.max_events_per_stream)
        
        self.stream_events[stream_id].append(event)
        
        # Сохраняем на диск в фоне
        await asyncio.to_thread(self._save_event_to_file, stream_id, event)
        
        logger.info(f"Peak count logged: stream={stream_id}, new_peak={pig_count}, previous={current_peak}")
    
    async def log_activity_spike(self, 
                                stream_id: str, 
                                pig_count: int, 
                                confidence: float,
                                frame: Optional[np.ndarray] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """Логирует событие всплеска активности"""
        
        # Проверяем всплеск
        if not self._detect_activity_spike(stream_id, pig_count):
            return
        
        event_id = self._generate_event_id()
        frame_path = None
        
        if frame is not None:
            frame_path = self._save_frame(frame, event_id)
        
        event = EventData(
            event_id=event_id,
            stream_id=stream_id,
            event_type='activity_spike',
            timestamp=time.time(),
            pig_count=pig_count,
            confidence=confidence,
            frame_path=frame_path,
            metadata=metadata or {}
        )
        
        # Добавляем в память
        if stream_id not in self.stream_events:
            self.stream_events[stream_id] = deque(maxlen=self.max_events_per_stream)
        
        self.stream_events[stream_id].append(event)
        
        # Сохраняем на диск в фоне
        await asyncio.to_thread(self._save_event_to_file, stream_id, event)
        
        logger.info(f"Activity spike logged: stream={stream_id}, count={pig_count}")
    
    def get_events(self, 
                   stream_id: str, 
                   event_type: Optional[str] = None,
                   limit: Optional[int] = None,
                   since_timestamp: Optional[float] = None) -> List[EventData]:
        """Получает события для потока с фильтрацией"""
        
        if stream_id not in self.stream_events:
            return []
        
        events = list(self.stream_events[stream_id])
        
        # Фильтрация по типу события
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Фильтрация по времени
        if since_timestamp:
            events = [e for e in events if e.timestamp >= since_timestamp]
        
        # Сортировка по времени (новые первыми)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Ограничение количества
        if limit:
            events = events[:limit]
        
        return events
    
    def get_stream_stats(self, stream_id: str) -> Dict[str, Any]:
        """Получает статистику по потоку"""
        
        if stream_id not in self.stream_events:
            return {
                'total_events': 0,
                'peak_count': 0,
                'event_types': {}
            }
        
        events = list(self.stream_events[stream_id])
        event_types = {}
        
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            'total_events': len(events),
            'peak_count': self.stream_peaks.get(stream_id, 0),
            'event_types': event_types,
            'latest_event': events[-1].timestamp if events else None
        }
    
    async def cleanup_old_events(self, max_age_hours: int = 24):
        """Очищает старые события"""
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for stream_id in list(self.stream_events.keys()):
            events = self.stream_events[stream_id]
            
            # Удаляем старые события из памяти
            while events and events[0].timestamp < cutoff_time:
                old_event = events.popleft()
                
                # Удаляем связанный кадр
                if old_event.frame_path and Path(old_event.frame_path).exists():
                    try:
                        Path(old_event.frame_path).unlink()
                    except Exception as e:
                        logger.error(f"Error deleting old frame {old_event.frame_path}: {e}")
        
        logger.info(f"Cleaned up events older than {max_age_hours} hours")

# Глобальный экземпляр логгера событий
_event_logger: Optional[EventLogger] = None

def get_event_logger() -> EventLogger:
    """Получает глобальный экземпляр логгера событий"""
    global _event_logger
    if _event_logger is None:
        _event_logger = EventLogger()
    return _event_logger
