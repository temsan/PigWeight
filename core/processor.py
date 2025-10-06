"""
Единый, унифицированный видео-процессор для проекта PigWeight.
Интегрирован с DynamicBatcher для адаптивной обработки.
Поддерживает журналирование ключевых событий.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from core.config import CONFIG
from services.model_adapter import ModelAdapter
from core.preprocess import center_crop_resize
from core.preprocess import map_polys_from_center_crop
from core.dynamic_batcher import DynamicBatcher, BatcherConfig
from core.priority_frame_queue import PriorityFrameQueue

# Импорт системы событий
try:
    from services.event_logger import get_event_logger
    HAVE_EVENT_LOGGER = True
except ImportError:
    HAVE_EVENT_LOGGER = False

logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    """Опции для обработки кадра."""
    conf_threshold: float = 0.3
    img_size: int = 640  # Уменьшено для ускорения обработки

@dataclass
class FrameResult:
    """Структурированный результат обработки кадра."""
    detections: int = 0
    confidence: float = 0.0
    masks: List[np.ndarray] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    preprocessed_shape: Optional[Tuple[int, int]] = None
    original_shape: Optional[Tuple[int, int]] = None
    timestamp: float = 0.0

# Тип элемента в очереди батчера: (кадр, временная метка, future для результата)
BatchItem = Tuple[np.ndarray, float, asyncio.Future]

class UnifiedVideoProcessor:
    """
    Универсальный процессор с асинхронной обработкой и адаптивным батчингом.
    Поддерживает журналирование ключевых событий.
    """

    def __init__(self, stream_id: str, loop: asyncio.AbstractEventLoop, options: Optional[ProcessingOptions] = None):
        self.stream_id = stream_id
        self.loop = loop
        self.options = options or ProcessingOptions(
            conf_threshold=getattr(CONFIG, "CONF_THRESHOLD", 0.3),
            img_size=getattr(CONFIG, "IMG_SIZE", 960)
        )

        model_path = getattr(CONFIG, "MODEL_PATH", "")
        logger.info(f"🔧 Создание ModelAdapter с путем: {model_path}")
        self.model_adapter = ModelAdapter(model_path=model_path)
        logger.info(f"✅ ModelAdapter создан: {self.model_adapter}")

        # Инициализация системы событий
        self.event_logger = get_event_logger() if HAVE_EVENT_LOGGER else None
        
        # Состояние для отслеживания событий
        self.last_count = 0
        self.peak_count = 0
        self.last_event_time = 0.0
        self.event_cooldown = 2.0  # Минимальный интервал между событиями одного типа
        
        # Приоритетная очередь для кадров
        from core.priority_frame_queue import QueueConfig
        queue_config = QueueConfig(
            max_size=1000,  # Максимальное количество кадров
            max_memory_mb=100,  # Максимум 100MB в очереди
            max_age_seconds=5.0,  # Максимум 5 секунд в очереди
            cleanup_interval=1.0  # Очистка каждую секунду
        )
        self.priority_queue = PriorityFrameQueue(queue_config)
        
        # Настройки для детекции линий (можно вынести в конфиг)
        self.line_left_x = getattr(CONFIG, "LINE_LEFT_X", 0.25)
        self.line_right_x = getattr(CONFIG, "LINE_RIGHT_X", 0.75)

        if not self.model_adapter.backend:
            logger.error(f"[{self.stream_id}] Не удалось загрузить модель. Процессор неактивен.")
            self.is_active = False
            self.batcher = None
        else:
            self.is_active = True
            batcher_config = BatcherConfig(
                max_batch_size=getattr(CONFIG, "MAX_BATCH_SIZE", 16),
                target_latency_ms=getattr(CONFIG, "TARGET_LATENCY_MS", 50.0)
            )
            self.batcher = DynamicBatcher[BatchItem](batcher_config, self._execute_batch)
            logger.info(f"[{self.stream_id}] Унифицированный процессор активен. Backend: {self.model_adapter.backend}")
            
            if self.event_logger:
                logger.info(f"[{self.stream_id}] Система событий активна")
            else:
                logger.warning(f"[{self.stream_id}] Система событий недоступна")

    async def start(self):
        """Запускает фоновые задачи процессора (батчер)."""
        if self.is_active and self.batcher and not self.batcher.is_running:
            await self.batcher.start()

    async def stop(self):
        """Останавливает фоновые задачи процессора."""
        if self.is_active and self.batcher and self.batcher.is_running:
            await self.batcher.stop()

    async def process_frame_async(self, frame: np.ndarray, timestamp: float = 0.0) -> FrameResult:
        """
        Асинхронно отправляет кадр на обработку и возвращает результат.
        Включает журналирование ключевых событий.
        """
        if not self.is_active or self.batcher is None:
            return FrameResult(timestamp=timestamp) # Возвращаем пустой результат, если процессор неактивен

        future = self.loop.create_future()
        await self.batcher.add_item((frame, timestamp, future))
        result = await future
        
        # Обработка событий после получения результата
        if self.event_logger and result.detections > 0:
            await self._process_events(frame, result)
        
        return result
    
    async def _process_events(self, frame: np.ndarray, result: FrameResult):
        """Обрабатывает и логирует ключевые события"""
        current_time = time.time()
        current_count = result.detections
        
        # Проверяем cooldown для предотвращения спама событий
        if current_time - self.last_event_time < self.event_cooldown:
            return
        
        try:
            # 1. Проверка пересечения линий (упрощенная логика)
            if self._detect_line_crossing(result.centroids):
                await self.event_logger.log_line_crossing(
                    stream_id=self.stream_id,
                    pig_count=current_count,
                    confidence=result.confidence,
                    frame=frame.copy(),
                    metadata={
                        'centroids': result.centroids,
                        'bboxes': result.bboxes,
                        'line_positions': [self.line_left_x, self.line_right_x]
                    }
                )
                self.last_event_time = current_time
            
            # 2. Проверка нового пика количества
            if current_count > self.peak_count:
                await self.event_logger.log_peak_count(
                    stream_id=self.stream_id,
                    pig_count=current_count,
                    confidence=result.confidence,
                    frame=frame.copy(),
                    metadata={
                        'previous_peak': self.peak_count,
                        'centroids': result.centroids
                    }
                )
                self.peak_count = current_count
                self.last_event_time = current_time
            
            # 3. Проверка всплеска активности
            await self.event_logger.log_activity_spike(
                stream_id=self.stream_id,
                pig_count=current_count,
                confidence=result.confidence,
                frame=frame.copy(),
                metadata={
                    'previous_count': self.last_count,
                    'centroids': result.centroids
                }
            )
            
            self.last_count = current_count
            
        except Exception as e:
            logger.error(f"[{self.stream_id}] Ошибка при обработке событий: {e}")
    
    def update_line_positions(self, left_x: float, right_x: float):
        """Обновляет позиции линий для детекции пересечений"""
        self.line_left_x = float(left_x)
        self.line_right_x = float(right_x)
        logger.debug(f"[{self.stream_id}] Обновлены позиции линий: L={self.line_left_x:.3f}, R={self.line_right_x:.3f}")
    
    def _detect_line_crossing(self, centroids: List[Tuple[float, float]]) -> bool:
        """
        Упрощенная детекция пересечения линий.
        В реальной системе здесь должна быть более сложная логика отслеживания траекторий.
        """
        if not centroids:
            return False
        
        # Проверяем, есть ли объекты в зоне между линиями
        for x, y in centroids:
            # Нормализуем координаты (предполагаем, что они уже в диапазоне 0-1)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if self.line_left_x <= x <= self.line_right_x:
                    return True
        
        return False

    def _execute_batch(self, batch: List[BatchItem]):
        """
        Метод, выполняющий обработку батча. Вызывается из DynamicBatcher.
        ВНИМАНИЕ: Этот метод выполняется в отдельном потоке (через run_in_executor).
        """
        frames = [item[0] for item in batch]
        timestamps = [item[1] for item in batch]
        futures = [item[2] for item in batch]

        try:
            # 1. Предобработка всех кадров в батче
            preprocessed_batch = [center_crop_resize(f, self.options.img_size) for f in frames]
            processed_frames = [p_data['img'] for p_data in preprocessed_batch]

            # 2. Инференс всего батча
            inference_results = self.model_adapter.infer(processed_frames)
            # Проверяем только наличие масок, чтобы не засорять лог
            mask_presence = [m.get('masks') is not None and len(m.get('masks')) > 0 for m in inference_results]
            logger.debug(f"Batch of {len(inference_results)}. Masks presence: {mask_presence}")

            # 3. Постобработка и отправка результатов
            for i, future in enumerate(futures):
                timestamp = timestamps[i]
                if i < len(inference_results):
                    result_data = inference_results[i]
                    p_data = preprocessed_batch[i]

                    # Маппинг масок
                    mapped_masks = []
                    raw_masks = result_data.get('masks')
                    transform_meta = p_data.get('transform_meta')
                    logger.info(f"[{self.stream_id}] 🔍 Raw masks from model: {type(raw_masks)}, count: {len(raw_masks) if raw_masks else 0}")
                    
                    if raw_masks and transform_meta:
                        logger.info(f"[{self.stream_id}] 📊 Processing {len(raw_masks)} raw masks")
                        mapped_masks = map_polys_from_center_crop(
                            raw_masks,
                            transform_meta
                        )
                        logger.info(f"[{self.stream_id}] ✅ Mapped {len(mapped_masks)} masks")
                    else:
                        logger.info(f"[{self.stream_id}] ❌ No masks to map - raw_masks: {bool(raw_masks)}, transform_meta: {bool(transform_meta)}")
                    if transform_meta and 'original_size' in transform_meta:
                        ow, oh = transform_meta['original_size']
                        original_shape = (int(round(oh)), int(round(ow)))
                    else:
                        img_h, img_w = p_data['img'].shape[:2]
                        original_shape = (int(img_h), int(img_w))

                    frame_result = FrameResult(
                        detections=result_data.get("detections", 0),
                        confidence=result_data.get("confidence", 0.0),
                        masks=mapped_masks,
                        bboxes=result_data.get("bboxes", []),
                        centroids=result_data.get("centroids", []),
                        original_shape=original_shape,
                        preprocessed_shape=p_data['img'].shape[:2],
                        timestamp=timestamp
                    )
                    if not future.done():
                        self.loop.call_soon_threadsafe(future.set_result, frame_result)
                else:
                    # Если для кадра нет результата инференса
                    if not future.done():
                        self.loop.call_soon_threadsafe(future.set_result, FrameResult(timestamp=timestamp))

        except Exception as e:
            logger.error(f"[{self.stream_id}] Ошибка в цикле обработки батча: {e}", exc_info=True)
            # Уведомляем все фьючерсы в батче об ошибке
            for future in futures:
                if not future.done():
                    self.loop.call_soon_threadsafe(future.set_exception, e)


# Глобальный менеджер процессоров
_PROCESSORS: Dict[str, UnifiedVideoProcessor] = {}
_PROCESSOR_LOCK = asyncio.Lock()

async def get_processor(stream_id: str, options: Optional[ProcessingOptions] = None) -> UnifiedVideoProcessor:
    """
    Асинхронная фабричная функция для получения или создания инстанса процессора.
    """
    async with _PROCESSOR_LOCK:
        if stream_id not in _PROCESSORS:
            logger.info(f"Создание нового UnifiedVideoProcessor для stream_id: {stream_id}")
            loop = asyncio.get_running_loop()
            processor = UnifiedVideoProcessor(stream_id, loop, options)
            await processor.start()
            _PROCESSORS[stream_id] = processor
        return _PROCESSORS[stream_id]

async def remove_processor(stream_id: str):
    """
    Удаляет процессор из менеджера и останавливает его.
    """
    async with _PROCESSOR_LOCK:
        if stream_id in _PROCESSORS:
            logger.info(f"Удаление UnifiedVideoProcessor для stream_id: {stream_id}")
            processor = _PROCESSORS.pop(stream_id)
            await processor.stop()

async def reset_processors():
    """Сбрасывает все процессоры."""
    async with _PROCESSOR_LOCK:
        logger.info("Сброс всех процессоров UnifiedVideoProcessor")
        for stream_id in list(_PROCESSORS.keys()):
            processor = _PROCESSORS.pop(stream_id)
            await processor.stop()

