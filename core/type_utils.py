"""
Утилиты для безопасного преобразования типов данных
"""

import logging
import numpy as np
from typing import Union, Optional, Any

logger = logging.getLogger(__name__)

def safe_tensor_conversion(tensor: Any, target_dtype: Any) -> Any:
    """
    Безопасное преобразование типов тензоров
    
    Args:
        tensor: Входной тензор (torch.Tensor, np.ndarray, или другой)
        target_dtype: Целевой тип данных
        
    Returns:
        Преобразованный тензор или исходный при ошибке
    """
    try:
        # Проверяем, нужно ли преобразование
        if hasattr(tensor, 'dtype') and tensor.dtype == target_dtype:
            return tensor
        
        # Для PyTorch тензоров
        if hasattr(tensor, 'to') and hasattr(tensor, 'dtype'):
            # Специальная обработка для c10::Half ↔ float
            if str(tensor.dtype) == 'torch.float16' and str(target_dtype) == 'torch.float32':
                return tensor.float()
            elif str(tensor.dtype) == 'torch.float32' and str(target_dtype) == 'torch.float16':
                return tensor.half()
            else:
                return tensor.to(target_dtype)
        
        # Для NumPy массивов
        elif hasattr(tensor, 'astype'):
            return tensor.astype(target_dtype)
        
        # Для обычных чисел
        else:
            return target_dtype(tensor)
            
    except Exception as e:
        logger.warning(f"Type conversion failed: {e}, using original tensor")
        return tensor

def ensure_float32(data: Any) -> Any:
    """Обеспечивает, что данные в формате float32"""
    try:
        if hasattr(data, 'dtype'):
            if 'float16' in str(data.dtype) or 'half' in str(data.dtype):
                return safe_tensor_conversion(data, 
                    getattr(data, 'float32', np.float32) if hasattr(data, 'float32') else np.float32)
        return data
    except Exception as e:
        logger.warning(f"Failed to ensure float32: {e}")
        return data

def ensure_half_precision(data: Any) -> Any:
    """Обеспечивает, что данные в формате half precision (float16)"""
    try:
        if hasattr(data, 'dtype'):
            if 'float32' in str(data.dtype):
                return safe_tensor_conversion(data,
                    getattr(data, 'float16', np.float16) if hasattr(data, 'float16') else np.float16)
        return data
    except Exception as e:
        logger.warning(f"Failed to ensure half precision: {e}")
        return data

def detect_optimal_dtype(device: str = "cpu") -> Any:
    """
    Определяет оптимальный тип данных для устройства
    
    Args:
        device: Устройство ("cpu", "cuda", etc.)
        
    Returns:
        Оптимальный dtype
    """
    try:
        # Для CPU лучше использовать float32
        if device.startswith("cpu"):
            try:
                import torch
                return torch.float32
            except ImportError:
                return np.float32
        
        # Для GPU можно использовать float16 для экономии памяти
        elif device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return torch.float16
                else:
                    return torch.float32
            except ImportError:
                return np.float32
        
        # По умолчанию float32
        else:
            try:
                import torch
                return torch.float32
            except ImportError:
                return np.float32
                
    except Exception as e:
        logger.warning(f"Failed to detect optimal dtype: {e}")
        try:
            import torch
            return torch.float32
        except ImportError:
            return np.float32

def validate_tensor_compatibility(tensor1: Any, tensor2: Any) -> bool:
    """
    Проверяет совместимость типов двух тензоров
    
    Args:
        tensor1: Первый тензор
        tensor2: Второй тензор
        
    Returns:
        True если типы совместимы
    """
    try:
        if not (hasattr(tensor1, 'dtype') and hasattr(tensor2, 'dtype')):
            return True  # Если нет dtype, считаем совместимыми
        
        dtype1 = str(tensor1.dtype)
        dtype2 = str(tensor2.dtype)
        
        # Проверяем точное совпадение
        if dtype1 == dtype2:
            return True
        
        # Проверяем совместимые типы
        float_types = ['float16', 'float32', 'float64', 'half', 'float', 'double']
        int_types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'byte', 'short', 'int', 'long']
        
        dtype1_category = 'float' if any(ft in dtype1 for ft in float_types) else 'int'
        dtype2_category = 'float' if any(ft in dtype2 for ft in float_types) else 'int'
        
        return dtype1_category == dtype2_category
        
    except Exception as e:
        logger.warning(f"Failed to validate tensor compatibility: {e}")
        return True  # В случае ошибки считаем совместимыми

class TypeSafetyManager:
    """Менеджер для обеспечения безопасности типов"""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.optimal_dtype = detect_optimal_dtype(device)
        self.conversion_cache = {}
        
    def prepare_tensor(self, tensor: Any, force_dtype: Optional[Any] = None) -> Any:
        """
        Подготавливает тензор к использованию с правильным типом
        
        Args:
            tensor: Входной тензор
            force_dtype: Принудительный тип данных
            
        Returns:
            Подготовленный тензор
        """
        try:
            target_dtype = force_dtype or self.optimal_dtype
            
            # Кэшируем результаты преобразования
            cache_key = (id(tensor), str(target_dtype))
            if cache_key in self.conversion_cache:
                return self.conversion_cache[cache_key]
            
            result = safe_tensor_conversion(tensor, target_dtype)
            
            # Кэшируем только если преобразование успешно
            if result is not tensor:
                self.conversion_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to prepare tensor: {e}")
            return tensor
    
    def clear_cache(self):
        """Очищает кэш преобразований"""
        self.conversion_cache.clear()
    
    def get_stats(self) -> dict:
        """Возвращает статистику использования"""
        return {
            "device": self.device,
            "optimal_dtype": str(self.optimal_dtype),
            "cache_size": len(self.conversion_cache),
            "conversions_cached": len(self.conversion_cache)
        }