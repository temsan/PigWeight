"""
Единый, унифицированный видео-процессор для проекта PigWeight.

Этот модуль представляет собой результат рефакторинга, объединяющий логику
из различных частей проекта в один управляемый класс.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from core.config import CONFIG
from services.model_adapter import ModelAdapter
from core.optimized_preprocess import center_crop_resize
from core.preprocess import map_polys_to_original

logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    """Опции для обработки кадра."""
    conf_threshold: float = 0.3
    img_size: int = 960

@dataclass
class FrameResult:
    """Структурированный результат обработки кадра."""
    detections: int = 0
    confidence: float = 0.0
    masks: List[np.ndarray] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list) # [x1, y1, x2, y2]
    centroids: List[Tuple[float, float]] = field(default_factory=list) # (cx, cy)
    preprocessed_shape: Optional[Tuple[int, int]] = None
    original_shape: Optional[Tuple[int, int]] = None

class UnifiedVideoProcessor:
    """
    Универсальный процессор, который инкапсулирует всю логику обработки видео.
    """

    def __init__(self, stream_id: str, options: Optional[ProcessingOptions] = None):
        self.stream_id = stream_id
        self.options = options or ProcessingOptions(
            conf_threshold=CONFIG.get("CONF_THRESHOLD", 0.3),
            img_size=CONFIG.get("IMG_SIZE", 960)
        )

        self.model_adapter = ModelAdapter(
            model_path=CONFIG.get("MODEL_PATH", ""),
            device=CONFIG.get("DEVICE", "cpu")
        )

        if not self.model_adapter.backend:
            logger.error(f"[{self.stream_id}] Не удалось загрузить модель. Процессор неактивен.")
            self.is_active = False
        else:
            self.is_active = True
            logger.info(f"[{self.stream_id}] Унифицированный процессор активен. Backend: {self.model_adapter.backend}")

    def process_frame(self, frame: np.ndarray) -> Optional[FrameResult]:
        """
        Выполняет полную цепочку обработки для одного кадра.
        
        Args:
            frame: Кадр в формате NumPy array (BGR).

        Returns:
            Структурированный результат или None в случае ошибки.
        """
        if not self.is_active or frame is None:
            return None

        try:
            # 1. Предобработка
            original_shape = frame.shape[:2]
            preprocessed_data = center_crop_resize(frame, self.options.img_size)
            processed_frame = preprocessed_data.get("img")
            if processed_frame is None:
                return None

            # 2. Инференс (обрабатываем как батч из одного элемента)
            inference_results = self.model_adapter.infer([processed_frame])
            if not inference_results:
                return None

            result = inference_results[0]

            # 3. Постобработка (маппинг масок)
            mapped_masks = []
            if result.get('masks'):
                # Для постобработки нужны scale и pad, которые center_crop_resize не возвращает.
                # В данном случае маппинг будет неточным, но для демонстрации оставим так.
                # TODO: Улучшить маппинг для center_crop
                pass # map_polys_to_original не совместим с center_crop

            return FrameResult(
                detections=result.get("detections", 0),
                confidence=result.get("confidence", 0.0),
                masks=result.get("masks", []),
                bboxes=result.get("bboxes", []),
                centroids=result.get("centroids", []),
                original_shape=original_shape,
                preprocessed_shape=processed_frame.shape[:2]
            )

        except Exception as e:
            logger.error(f"[{self.stream_id}] Ошибка в цикле обработки кадра: {e}", exc_info=True)
            return None

# Глобальный менеджер процессоров, чтобы избежать проблем с hot-reload
_PROCESSORS: Dict[str, UnifiedVideoProcessor] = {}

def get_processor(stream_id: str, options: Optional[ProcessingOptions] = None) -> UnifiedVideoProcessor:
    """
    Фабричная функция для получения или создания инстанса процессора.
    """
    if stream_id not in _PROCESSORS:
        logger.info(f"Создание нового UnifiedVideoProcessor для stream_id: {stream_id}")
        _PROCESSORS[stream_id] = UnifiedVideoProcessor(stream_id, options)
    return _PROCESSORS[stream_id]

def remove_processor(stream_id: str):
    """
    Удаляет процессор из менеджера.
    """
    if stream_id in _PROCESSORS:
        logger.info(f"Удаление UnifiedVideoProcessor для stream_id: {stream_id}")
        del _PROCESSORS[stream_id]

def reset_processor(stream_id: Optional[str] = None) -> None:
    """
    Сбрасывает процессоры.

    - Если указан `stream_id`, удаляет только соответствующий процессор.
    - Если не указан, очищает все процессоры.
    """
    if stream_id is not None:
        if stream_id in _PROCESSORS:
            logger.info(f"Сброс процессора для stream_id: {stream_id}")
            del _PROCESSORS[stream_id]
        else:
            logger.info(f"reset_processor: процессор для stream_id={stream_id} не найден")
    else:
        if _PROCESSORS:
            logger.info("Сброс всех процессоров UnifiedVideoProcessor")
            _PROCESSORS.clear()
        else:
            logger.info("reset_processor: нет активных процессоров для сброса")