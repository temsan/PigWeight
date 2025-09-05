"""
DynamicBatcher - Адаптивный батчер для оптимизации инференса
Автоматически адаптирует размер батча под латентность и производительность
"""

import asyncio
import logging
import time
import threading
from typing import List, Dict, Any, Optional, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from collections import deque
import statistics
import math

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Тип элементов батча

@dataclass
class BatcherConfig:
    """Конфигурация батчера"""
    min_batch_size: int = 1
    max_batch_size: int = 16
    initial_batch_size: int = 4
    target_latency_ms: float = 50.0
    max_wait_time_ms: float = 100.0
    adaptation_interval: float = 2.0  # Интервал адаптации в секундах
    latency_tolerance: float = 0.2  # 20% толерантность для латентности
    throughput_weight: float = 0.7  # Вес throughput vs latency (0.0 - только латентность, 1.0 - только throughput)
    warmup_batches: int = 10  # Количество батчей для прогрева
    
@dataclass
class BatchMetrics:
    """Метрики батча"""
    batch_size: int
    processing_time_ms: float
    wait_time_ms: float
    total_latency_ms: float
    throughput_fps: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class BatcherStats:
    """Статистика батчера"""
    current_batch_size: int = 4
    avg_latency_ms: float = 0.0
    avg_throughput_fps: float = 0.0
    total_batches: int = 0
    total_items: int = 0
    adaptations_count: int = 0
    efficiency_score: float = 0.0  # Комбинированная оценка производительности
    last_adaptation: float = field(default_factory=time.time)

class LatencyPredictor:
    """Предиктор латентности для разных размеров батча"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.measurements: Dict[int, List[float]] = {}
        
    def add_measurement(self, batch_size: int, latency_ms: float):
        """Добавление измерения латентности"""
        if batch_size not in self.measurements:
            self.measurements[batch_size] = []
            
        self.measurements[batch_size].append(latency_ms)
        
        # Ограничиваем историю
        if len(self.measurements[batch_size]) > self.max_history:
            self.measurements[batch_size].pop(0)
            
    def predict_latency(self, batch_size: int) -> Optional[float]:
        """Предсказание латентности для размера батча"""
        if batch_size in self.measurements and self.measurements[batch_size]:
            # Используем медиану последних измерений для стабильности
            recent_measurements = self.measurements[batch_size][-10:]  # Последние 10
            return statistics.median(recent_measurements)
            
        # Если нет прямых измерений, пытаемся интерполировать
        available_sizes = sorted([k for k, v in self.measurements.items() if v])
        
        if len(available_sizes) >= 2:
            return self._interpolate_latency(batch_size, available_sizes)
            
        return None
        
    def _interpolate_latency(self, target_size: int, available_sizes: List[int]) -> float:
        """Интерполяция латентности между известными размерами"""
        # Находим ближайшие размеры
        lower_size = max([s for s in available_sizes if s <= target_size], default=None)
        upper_size = min([s for s in available_sizes if s >= target_size], default=None)
        
        if lower_size and upper_size:
            lower_latency = statistics.median(self.measurements[lower_size][-5:])
            upper_latency = statistics.median(self.measurements[upper_size][-5:])
            
            # Линейная интерполяция
            if upper_size != lower_size:
                ratio = (target_size - lower_size) / (upper_size - lower_size)
                return lower_latency + ratio * (upper_latency - lower_latency)
            else:
                return lower_latency
                
        # Экстраполяция на основе ближайшего размера
        if lower_size:
            base_latency = statistics.median(self.measurements[lower_size][-5:])
            # Предполагаем сублинейный рост латентности
            scale_factor = math.pow(target_size / lower_size, 0.8)
            return base_latency * scale_factor
        elif upper_size:
            base_latency = statistics.median(self.measurements[upper_size][-5:])
            scale_factor = math.pow(target_size / upper_size, 0.8)
            return base_latency * scale_factor
            
        return 50.0  # Дефолтная оценка

class DynamicBatcher(Generic[T]):
    """
    Адаптивный батчер с автоматической оптимизацией размера батча:
    - Минимизация латентности под целевое значение (50ms)
    - Максимизация throughput при соблюдении латентности
    - Автоматическая адаптация каждые 2 секунды
    - Мониторинг производительности
    """
    
    def __init__(self, 
                 config: BatcherConfig, 
                 process_batch: Callable[[List[T]], Any]):
        self.config = config
        self.process_batch = process_batch
        
        # Состояние батчера
        self._current_batch: List[T] = []
        self._batch_lock = threading.RLock()
        self._running = False
        
        # Адаптивные параметры
        self._current_batch_size = config.initial_batch_size
        self._current_wait_time = config.max_wait_time_ms / 1000.0
        
        # Статистика и метрики
        self.stats = BatcherStats(current_batch_size=config.initial_batch_size)
        self._metrics_history: deque[BatchMetrics] = deque(maxlen=100)
        self._predictor = LatencyPredictor()
        
        # Задачи и события
        self._batch_ready_event = asyncio.Event()
        self._adaptation_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None
        
        # Тайминги
        self._last_adaptation = time.time()
        self._batch_start_time = 0.0
        self._warmup_completed = False
        
        logger.info(f"DynamicBatcher создан: batch_size={config.initial_batch_size}, target_latency={config.target_latency_ms}ms")
        
    async def start(self):
        """Запуск батчера"""
        if self._running:
            return
            
        self._running = True
        
        # Запуск фоновых задач
        self._adaptation_task = asyncio.create_task(self._adaptation_loop())
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        logger.info("DynamicBatcher запущен")
        
    async def stop(self):
        """Остановка батчера"""
        if not self._running:
            return
            
        self._running = False
        
        # Остановка задач
        if self._adaptation_task:
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass
                
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
                
        # Обработка оставшихся элементов
        await self._flush_batch()
        
        logger.info("DynamicBatcher остановлен")
        
    async def add_item(self, item: T) -> bool:
        """
        Добавление элемента в батч
        
        Args:
            item: Элемент для обработки
            
        Returns:
            True если элемент добавлен успешно
        """
        if not self._running:
            return False
            
        with self._batch_lock:
            self._current_batch.append(item)
            
            # Инициализация таймера батча
            if len(self._current_batch) == 1:
                self._batch_start_time = time.time()
                
            # Проверка готовности батча
            if len(self._current_batch) >= self._current_batch_size:
                self._batch_ready_event.set()
                
        return True
        
    async def _processing_loop(self):
        """Основной цикл обработки батчей"""
        while self._running:
            try:
                # Ожидание готовности батча или таймаута
                wait_task = asyncio.create_task(self._batch_ready_event.wait())
                timeout_task = asyncio.create_task(asyncio.sleep(self._current_wait_time))
                
                done, pending = await asyncio.wait(
                    [wait_task, timeout_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Отмена незавершенных задач
                for task in pending:
                    task.cancel()
                    
                # Обработка батча если есть элементы
                with self._batch_lock:
                    if self._current_batch:
                        await self._process_current_batch()
                        
                # Сброс события
                self._batch_ready_event.clear()
                
            except Exception as e:
                logger.error(f"Ошибка в processing_loop: {e}")
                await asyncio.sleep(0.1)
                
    async def _process_current_batch(self):
        """Обработка текущего батча"""
        if not self._current_batch:
            return
            
        batch = self._current_batch.copy()
        batch_size = len(batch)
        wait_time = time.time() - self._batch_start_time
        
        # Очистка текущего батча
        self._current_batch.clear()
        
        # Измерение времени обработки
        process_start = time.time()
        
        try:
            # Обработка батча в отдельном потоке
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.process_batch, batch)
            
        except Exception as e:
            logger.error(f"Ошибка обработки батча: {e}")
        finally:
            process_time = time.time() - process_start
            
            # Запись метрик
            await self._record_metrics(batch_size, process_time, wait_time)
            
    async def _record_metrics(self, batch_size: int, process_time: float, wait_time: float):
        """Запись метрик батча"""
        total_latency = (process_time + wait_time) * 1000  # в мс
        throughput = batch_size / process_time if process_time > 0 else 0
        
        metrics = BatchMetrics(
            batch_size=batch_size,
            processing_time_ms=process_time * 1000,
            wait_time_ms=wait_time * 1000,
            total_latency_ms=total_latency,
            throughput_fps=throughput
        )
        
        self._metrics_history.append(metrics)
        self._predictor.add_measurement(batch_size, process_time * 1000)
        
        # Обновление статистики
        self.stats.total_batches += 1
        self.stats.total_items += batch_size
        
        # Расчет средних значений
        if len(self._metrics_history) > 0:
            recent_metrics = list(self._metrics_history)[-10:]  # Последние 10 батчей
            self.stats.avg_latency_ms = statistics.mean(m.total_latency_ms for m in recent_metrics)
            self.stats.avg_throughput_fps = statistics.mean(m.throughput_fps for m in recent_metrics)
            
        # Проверка завершения прогрева
        if not self._warmup_completed and self.stats.total_batches >= self.config.warmup_batches:
            self._warmup_completed = True
            logger.info(f"Прогрев завершен после {self.stats.total_batches} батчей")
            
    async def _adaptation_loop(self):
        """Цикл адаптации размера батча"""
        while self._running:
            try:
                await asyncio.sleep(self.config.adaptation_interval)
                
                if self._warmup_completed and len(self._metrics_history) >= 5:
                    await self._adapt_batch_size()
                    
            except Exception as e:
                logger.error(f"Ошибка в adaptation_loop: {e}")
                await asyncio.sleep(1.0)
                
    async def _adapt_batch_size(self):
        """Адаптация размера батча"""
        current_time = time.time()
        if current_time - self._last_adaptation < self.config.adaptation_interval:
            return
            
        # Анализ текущей производительности
        performance_score = self._calculate_performance_score()
        
        # Поиск оптимального размера батча
        best_size = self._find_optimal_batch_size()
        
        if best_size != self._current_batch_size:
            old_size = self._current_batch_size
            self._current_batch_size = best_size
            self.stats.current_batch_size = best_size
            self.stats.adaptations_count += 1
            self._last_adaptation = current_time
            
            logger.info(
                f"Адаптация батча: {old_size} -> {best_size}, "
                f"производительность: {performance_score:.3f}, "
                f"латентность: {self.stats.avg_latency_ms:.1f}ms"
            )
            
        self.stats.efficiency_score = performance_score
        
    def _calculate_performance_score(self) -> float:
        """Расчет оценки производительности"""
        if not self._metrics_history:
            return 0.0
            
        recent_metrics = list(self._metrics_history)[-10:]
        
        # Латентность (нормализованная к целевой)
        avg_latency = statistics.mean(m.total_latency_ms for m in recent_metrics)
        latency_score = max(0, 1.0 - (avg_latency / self.config.target_latency_ms - 1.0))
        
        # Throughput (нормализованный к максимальному наблюдаемому)
        avg_throughput = statistics.mean(m.throughput_fps for m in recent_metrics)
        max_throughput = max((m.throughput_fps for m in self._metrics_history), default=1.0)
        throughput_score = avg_throughput / max_throughput if max_throughput > 0 else 0
        
        # Комбинированная оценка
        score = (
            (1.0 - self.config.throughput_weight) * latency_score +
            self.config.throughput_weight * throughput_score
        )
        
        return max(0.0, min(1.0, score))
        
    def _find_optimal_batch_size(self) -> int:
        """Поиск оптимального размера батча"""
        best_size = self._current_batch_size
        best_score = 0.0
        
        # Тестируем размеры в диапазоне
        for size in range(self.config.min_batch_size, self.config.max_batch_size + 1):
            score = self._evaluate_batch_size(size)
            
            if score > best_score:
                best_score = score
                best_size = size
                
        return best_size
        
    def _evaluate_batch_size(self, batch_size: int) -> float:
        """Оценка качества размера батча"""
        # Предсказание латентности
        predicted_latency = self._predictor.predict_latency(batch_size)
        
        if predicted_latency is None:
            # Если нет данных, используем эвристику
            predicted_latency = self._estimate_latency_heuristic(batch_size)
            
        # Оценка латентности
        latency_ratio = predicted_latency / self.config.target_latency_ms
        latency_penalty = max(0, latency_ratio - 1.0) ** 2  # Квадратичный штраф
        
        # Оценка throughput (предполагаем сублинейный рост)
        throughput_factor = math.pow(batch_size, 0.8)  # Сублинейный рост
        
        # Комбинированная оценка
        score = throughput_factor / (1.0 + latency_penalty)
        
        return score
        
    def _estimate_latency_heuristic(self, batch_size: int) -> float:
        """Эвристическая оценка латентности"""
        # Базовая латентность + рост с размером батча
        base_latency = 20.0  # мс
        scaling_factor = math.pow(batch_size, 0.7)  # Сублинейный рост
        return base_latency * scaling_factor
        
    async def _flush_batch(self):
        """Принудительная обработка текущего батча"""
        with self._batch_lock:
            if self._current_batch:
                await self._process_current_batch()
                
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики батчера"""
        return {
            'current_batch_size': self.stats.current_batch_size,
            'avg_latency_ms': self.stats.avg_latency_ms,
            'avg_throughput_fps': self.stats.avg_throughput_fps,
            'total_batches': self.stats.total_batches,
            'total_items': self.stats.total_items,
            'adaptations_count': self.stats.adaptations_count,
            'efficiency_score': self.stats.efficiency_score,
            'warmup_completed': self._warmup_completed,
            'current_queue_size': len(self._current_batch),
            'metrics_history_size': len(self._metrics_history),
            'predictor_sizes': list(self._predictor.measurements.keys())
        }
        
    @property
    def is_running(self) -> bool:
        """Проверка работы батчера"""
        return self._running