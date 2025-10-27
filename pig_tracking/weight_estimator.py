"""
Модуль оценки веса свиней на основе размера и формы.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class WeightEstimator:
    """
    Оценка веса свиней на основе визуальных характеристик.
    
    Использует эвристический подход на основе:
    - Площади bbox
    - Площади маски (если доступна)
    - Соотношения сторон
    - Калибровочных коэффициентов
    """
    
    def __init__(
        self,
        avg_weight_kg: float = 110.0,
        weight_std_kg: float = 15.0,
        use_masks: bool = True
    ):
        """
        Args:
            avg_weight_kg: Средний вес свиньи в кг
            weight_std_kg: Стандартное отклонение веса
            use_masks: Использовать маски для более точной оценки
        """
        self.avg_weight_kg = float(avg_weight_kg)
        self.weight_std_kg = float(weight_std_kg)
        self.use_masks = use_masks
        
        # Калибровочные параметры (настраиваются на реальных данных)
        self.bbox_area_to_weight_coef = 1.0
        self.mask_area_to_weight_coef = 1.2
        
        # История оценок для каждой свиньи (для сглаживания)
        self._pig_weight_history: Dict[int, list] = {}
        self._history_size = 5
        
        logger.info(
            f"WeightEstimator инициализирован: avg={avg_weight_kg}kg, "
            f"std={weight_std_kg}kg, use_masks={use_masks}"
        )
    
    def estimate_weight(
        self,
        pig_id: Optional[int] = None,
        bbox: Optional[list] = None,
        mask: Optional[np.ndarray] = None,
        frame_size: Optional[tuple] = None
    ) -> Optional[float]:
        """
        Оценивает вес свиньи.
        
        Args:
            pig_id: ID свиньи (для сглаживания оценок)
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Маска сегментации
            frame_size: Размер кадра (width, height)
            
        Returns:
            Оценка веса в кг или None
        """
        
        # Базовая оценка - средний вес с небольшим шумом
        base_estimate = self.avg_weight_kg
        
        # Если есть bbox, используем его для уточнения
        if bbox is not None and frame_size is not None:
            try:
                x1, y1, x2, y2 = bbox
                width, height = frame_size
                
                # Нормализованная площадь bbox
                bbox_area = ((x2 - x1) * (y2 - y1)) / (width * height)
                
                # Корректируем оценку на основе размера
                # Предполагаем, что средняя свинья занимает ~0.05 кадра
                size_factor = bbox_area / 0.05
                base_estimate *= (0.8 + 0.4 * size_factor)  # ±20% от среднего
                
            except Exception as e:
                logger.debug(f"Ошибка оценки по bbox: {e}")
        
        # Если есть маска, используем её для более точной оценки
        if self.use_masks and mask is not None:
            try:
                mask_area = np.sum(mask > 0)
                if frame_size:
                    total_pixels = frame_size[0] * frame_size[1]
                    mask_ratio = mask_area / total_pixels
                    
                    # Маска более точно отражает размер
                    size_factor = mask_ratio / 0.04  # Средняя маска ~4% кадра
                    base_estimate *= (0.85 + 0.3 * size_factor)
                    
            except Exception as e:
                logger.debug(f"Ошибка оценки по маске: {e}")
        
        # Добавляем небольшую случайную вариацию (±5%)
        variation = np.random.normal(0, self.weight_std_kg * 0.05)
        estimate = base_estimate + variation
        
        # Ограничиваем разумными пределами (60-180 кг)
        estimate = max(60.0, min(180.0, estimate))
        
        # Сглаживание по истории (если есть pig_id)
        if pig_id is not None:
            if pig_id not in self._pig_weight_history:
                self._pig_weight_history[pig_id] = []
            
            history = self._pig_weight_history[pig_id]
            history.append(estimate)
            
            # Ограничиваем размер истории
            if len(history) > self._history_size:
                history.pop(0)
            
            # Возвращаем среднее по истории
            estimate = float(np.mean(history))
        
        return round(estimate, 1)
    
    def calibrate(
        self,
        actual_weights: list,
        estimated_weights: list
    ):
        """
        Калибрует оценщик на основе реальных данных.
        
        Args:
            actual_weights: Список реальных весов
            estimated_weights: Список оценочных весов
        """
        if len(actual_weights) != len(estimated_weights):
            raise ValueError("Длины списков должны совпадать")
        
        if len(actual_weights) < 2:
            logger.warning("Недостаточно данных для калибровки")
            return
        
        try:
            # Простая линейная калибровка
            actual = np.array(actual_weights)
            estimated = np.array(estimated_weights)
            
            # Вычисляем коэффициент коррекции
            correction_factor = np.mean(actual) / np.mean(estimated)
            
            # Обновляем средний вес
            self.avg_weight_kg *= correction_factor
            
            # Обновляем стандартное отклонение
            self.weight_std_kg = float(np.std(actual))
            
            logger.info(
                f"Калибровка завершена: avg={self.avg_weight_kg:.1f}kg, "
                f"std={self.weight_std_kg:.1f}kg, correction={correction_factor:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка калибровки: {e}")
    
    def reset_history(self, pig_id: Optional[int] = None):
        """Сбрасывает историю оценок"""
        if pig_id is None:
            self._pig_weight_history.clear()
            logger.debug("История оценок сброшена для всех свиней")
        else:
            self._pig_weight_history.pop(pig_id, None)
            logger.debug(f"История оценок сброшена для свиньи {pig_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику оценщика"""
        return {
            'avg_weight_kg': self.avg_weight_kg,
            'weight_std_kg': self.weight_std_kg,
            'use_masks': self.use_masks,
            'tracked_pigs': len(self._pig_weight_history),
            'total_estimates': sum(len(h) for h in self._pig_weight_history.values())
        }


# Глобальный экземпляр оценщика
_weight_estimator: Optional[WeightEstimator] = None


def get_weight_estimator() -> WeightEstimator:
    """Получает глобальный экземпляр оценщика веса"""
    global _weight_estimator
    if _weight_estimator is None:
        _weight_estimator = WeightEstimator()
    return _weight_estimator


def calibrate_weight_estimator(actual_weights: list, estimated_weights: list):
    """Калибрует глобальный оценщик веса"""
    estimator = get_weight_estimator()
    estimator.calibrate(actual_weights, estimated_weights)
