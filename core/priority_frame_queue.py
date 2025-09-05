"""
PriorityFrameQueue - Приоритетная очередь кадров с автоматическим сбросом старых кадров
Оптимизирована для устранения блокировок и контроля памяти
"""

import asyncio
import heapq
import time
import threading
import logging
from typing import Optional, Dict, Any, List, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import weakref
import gc

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

class FrameEntry(NamedTuple):
    """Запись кадра в очереди"""
    priority: float  # timestamp (меньше = выше приоритет)
    frame_id: str
    data: Any
    size_bytes: int
    created_at: float

@dataclass
class QueueConfig:
    """Конфигурация очереди"""
    max_size: int = 1000  # Максимальное количество кадров
    max_memory_mb: int = 200  # Лимит памяти в МБ
    drop_threshold: float = 0.8  # При какой заполненности начинать сброс
    max_age_seconds: float = 2.0  # Максимальный возраст кадра
    cleanup_interval: float = 1.0  # Интервал очистки старых кадров
    priority_boost: float = 0.1  # Буст приоритета для новых кадров

@dataclass
class QueueStats:
    """Статистика очереди"""
    total_frames: int = 0
    dropped_frames: int = 0
    current_size: int = 0
    memory_usage_mb: float = 0.0
    avg_age_ms: float = 0.0
    max_age_ms: float = 0.0
    queue_efficiency: float = 0.0  # % полезных кадров
    last_cleanup: float = field(default_factory=time.time)

class PriorityFrameQueue:
    """
    Приоритетная очередь кадров с оптимизациями:
    - Приоритет по timestamp (новые кадры важнее)
    - Автоматический сброс старых кадров
    - Контроль использования памяти (200MB лимит)
    - Устранение блокировок через lock-free операции где возможно
    """
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self._heap: List[FrameEntry] = []
        self._lock = threading.RLock()  # Используем RLock для вложенных вызовов
        
        # Статистика и мониторинг
        self.stats = QueueStats()
        self._frame_registry: Dict[str, FrameEntry] = {}
        self._memory_tracker: Dict[str, int] = defaultdict(int)
        
        # Асинхронные задачи
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Оптимизации производительности
        self._last_priority = 0.0
        self._priority_increment = 0.001  # Микросекундное разрешение
        self._batch_operations = []
        
        logger.info(f"PriorityFrameQueue инициализирована: max_size={config.max_size}, max_memory={config.max_memory_mb}MB")
        
    async def start(self):
        """Запуск очереди и фоновых задач"""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("PriorityFrameQueue запущена")
        
    async def stop(self):
        """Остановка очереди"""
        if not self._running:
            return
            
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Очистка всех кадров
        with self._lock:
            self._heap.clear()
            self._frame_registry.clear()
            self._memory_tracker.clear()
            
        logger.info("PriorityFrameQueue остановлена")
        
    def put(self, frame_id: str, data: Any, timestamp: Optional[float] = None, 
            estimated_size: Optional[int] = None) -> bool:
        """
        Добавление кадра в очередь
        
        Args:
            frame_id: Уникальный идентификатор кадра
            data: Данные кадра
            timestamp: Временная метка (по умолчанию текущее время)
            estimated_size: Оценочный размер в байтах
            
        Returns:
            True если кадр добавлен, False если отброшен
        """
        if not self._running:
            return False
            
        current_time = time.time()
        timestamp = timestamp or current_time
        
        # Оценка размера кадра
        if estimated_size is None:
            estimated_size = self._estimate_size(data)
            
        # Быстрая проверка памяти без блокировки
        if self._should_drop_frame(estimated_size):
            self.stats.dropped_frames += 1
            return False
            
        # Создание записи кадра
        priority = timestamp - self.config.priority_boost  # Буст для новых кадров
        entry = FrameEntry(
            priority=priority,
            frame_id=frame_id,
            data=data,
            size_bytes=estimated_size,
            created_at=current_time
        )
        
        # Атомарное добавление
        with self._lock:
            # Проверка дубликатов
            if frame_id in self._frame_registry:
                # Обновляем существующий кадр если он новее
                existing = self._frame_registry[frame_id]
                if timestamp > existing.priority:
                    self._remove_frame_unsafe(frame_id)
                else:
                    return False
                    
            # Проверка лимитов перед добавлением
            if not self._check_limits_unsafe(estimated_size):
                self._make_space_unsafe(estimated_size)
                
            # Добавление в очередь
            heapq.heappush(self._heap, entry)
            self._frame_registry[frame_id] = entry
            self._memory_tracker[frame_id] = estimated_size
            
            # Обновление статистики
            self.stats.total_frames += 1
            self.stats.current_size += 1
            self.stats.memory_usage_mb += estimated_size / (1024 * 1024)
            
        return True
        
    def get(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Any, float]]:
        """
        Получение кадра с наивысшим приоритетом
        
        Args:
            timeout: Таймаут ожидания (не используется в синхронной версии)
            
        Returns:
            Tuple (frame_id, data, timestamp) или None
        """
        with self._lock:
            if not self._heap:
                return None
                
            # Получение кадра с наивысшим приоритетом
            entry = heapq.heappop(self._heap)
            
            # Удаление из реестра
            if entry.frame_id in self._frame_registry:
                del self._frame_registry[entry.frame_id]
                size = self._memory_tracker.pop(entry.frame_id, 0)
                
                # Обновление статистики
                self.stats.current_size -= 1
                self.stats.memory_usage_mb -= size / (1024 * 1024)
                
            return entry.frame_id, entry.data, entry.priority
            
    async def get_async(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Any, float]]:
        """
        Асинхронное получение кадра
        
        Args:
            timeout: Таймаут ожидания
            
        Returns:
            Tuple (frame_id, data, timestamp) или None
        """
        start_time = time.time()
        
        while self._running:
            result = self.get()
            if result is not None:
                return result
                
            # Проверка таймаута
            if timeout and (time.time() - start_time) > timeout:
                return None
                
            # Короткая задержка для предотвращения busy-wait
            await asyncio.sleep(0.001)
            
        return None
        
    def peek(self) -> Optional[Tuple[str, float]]:
        """
        Просмотр следующего кадра без извлечения
        
        Returns:
            Tuple (frame_id, timestamp) или None
        """
        with self._lock:
            if not self._heap:
                return None
            entry = self._heap[0]
            return entry.frame_id, entry.priority
            
    def remove(self, frame_id: str) -> bool:
        """
        Удаление конкретного кадра
        
        Args:
            frame_id: Идентификатор кадра
            
        Returns:
            True если кадр удален
        """
        with self._lock:
            return self._remove_frame_unsafe(frame_id)
            
    def clear(self):
        """Очистка всей очереди"""
        with self._lock:
            self._heap.clear()
            self._frame_registry.clear()
            self._memory_tracker.clear()
            self.stats.current_size = 0
            self.stats.memory_usage_mb = 0.0
            
    def size(self) -> int:
        """Текущий размер очереди"""
        return len(self._heap)
        
    def is_empty(self) -> bool:
        """Проверка пустоты очереди"""
        return len(self._heap) == 0
        
    def get_stats(self) -> QueueStats:
        """Получение статистики"""
        with self._lock:
            # Обновление динамической статистики
            if self._heap:
                current_time = time.time()
                ages = [(current_time - entry.created_at) * 1000 for entry in self._heap]
                self.stats.avg_age_ms = sum(ages) / len(ages)
                self.stats.max_age_ms = max(ages)
            else:
                self.stats.avg_age_ms = 0.0
                self.stats.max_age_ms = 0.0
                
            # Эффективность очереди
            if self.stats.total_frames > 0:
                self.stats.queue_efficiency = (
                    (self.stats.total_frames - self.stats.dropped_frames) / 
                    self.stats.total_frames * 100
                )
                
        return self.stats
        
    def _should_drop_frame(self, estimated_size: int) -> bool:
        """Быстрая проверка необходимости сброса кадра"""
        # Проверка лимита памяти
        memory_limit_bytes = self.config.max_memory_mb * 1024 * 1024
        current_memory = sum(self._memory_tracker.values())
        
        if (current_memory + estimated_size) > memory_limit_bytes * self.config.drop_threshold:
            return True
            
        # Проверка лимита размера очереди
        if len(self._heap) >= self.config.max_size * self.config.drop_threshold:
            return True
            
        return False
        
    def _check_limits_unsafe(self, estimated_size: int) -> bool:
        """Проверка лимитов (без блокировки)"""
        memory_limit_bytes = self.config.max_memory_mb * 1024 * 1024
        current_memory = sum(self._memory_tracker.values())
        
        # Проверка памяти
        if (current_memory + estimated_size) > memory_limit_bytes:
            return False
            
        # Проверка размера
        if len(self._heap) >= self.config.max_size:
            return False
            
        return True
        
    def _make_space_unsafe(self, needed_size: int):
        """Освобождение места в очереди (без блокировки)"""
        removed_count = 0
        target_removal = max(1, int(len(self._heap) * 0.1))  # Удаляем 10% старых кадров
        
        # Сначала удаляем старые кадры
        current_time = time.time()
        to_remove = []
        
        for entry in self._heap:
            age = current_time - entry.created_at
            if age > self.config.max_age_seconds:
                to_remove.append(entry.frame_id)
                
        # Удаляем старые кадры
        for frame_id in to_remove:
            if self._remove_frame_unsafe(frame_id):
                removed_count += 1
                
        # Если недостаточно места, удаляем кадры с наименьшим приоритетом
        while (removed_count < target_removal and self._heap and 
               not self._check_limits_unsafe(needed_size)):
            # Находим кадр с наименьшим приоритетом (самый старый)
            max_priority = max(self._heap, key=lambda x: x.priority)
            if self._remove_frame_unsafe(max_priority.frame_id):
                removed_count += 1
                
        self.stats.dropped_frames += removed_count
        
    def _remove_frame_unsafe(self, frame_id: str) -> bool:
        """Удаление кадра без блокировки"""
        if frame_id not in self._frame_registry:
            return False
            
        entry = self._frame_registry[frame_id]
        
        # Удаляем из heap (помечаем как удаленный)
        # Физическое удаление произойдет при извлечении
        try:
            self._heap.remove(entry)
            heapq.heapify(self._heap)
        except ValueError:
            pass  # Кадр уже не в heap
            
        # Удаляем из реестров
        del self._frame_registry[frame_id]
        size = self._memory_tracker.pop(frame_id, 0)
        
        # Обновляем статистику
        self.stats.current_size -= 1
        self.stats.memory_usage_mb -= size / (1024 * 1024)
        
        return True
        
    def _estimate_size(self, data: Any) -> int:
        """Оценка размера данных"""
        try:
            # Для numpy массивов
            if hasattr(data, 'nbytes'):
                return data.nbytes
                
            # Для bytes
            if isinstance(data, (bytes, bytearray)):
                return len(data)
                
            # Для строк
            if isinstance(data, str):
                return len(data.encode('utf-8'))
                
            # Приблизительная оценка для других типов
            return len(str(data)) * 4  # 4 байта на символ (консервативная оценка)
            
        except Exception:
            return 1024  # Дефолтная оценка 1KB
            
    async def _cleanup_loop(self):
        """Фоновая очистка старых кадров"""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                current_time = time.time()
                if current_time - self.stats.last_cleanup < self.config.cleanup_interval:
                    continue
                    
                removed_count = 0
                
                with self._lock:
                    # Поиск устаревших кадров
                    to_remove = []
                    for frame_id, entry in self._frame_registry.items():
                        age = current_time - entry.created_at
                        if age > self.config.max_age_seconds:
                            to_remove.append(frame_id)
                            
                    # Удаление устаревших кадров
                    for frame_id in to_remove:
                        if self._remove_frame_unsafe(frame_id):
                            removed_count += 1
                            
                    self.stats.last_cleanup = current_time
                    
                if removed_count > 0:
                    logger.debug(f"Очищено {removed_count} устаревших кадров")
                    
                # Принудительная сборка мусора каждые 10 итераций
                if int(current_time) % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Ошибка в cleanup_loop: {e}")
                await asyncio.sleep(1.0)