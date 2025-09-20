"""
Оптимизированная предобработка для синхронизации с датасетом
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple


def center_crop_resize(frame: np.ndarray, target_size: int = 960) -> Dict[str, Any]:
    """
    Предобработка, соответствующая датасету:
    1. Center crop до квадрата
    2. Resize до target_size
    3. Добавление черных полос для соответствия обучению
    
    Это обеспечивает максимальное соответствие с датасетом.
    """
    h, w = frame.shape[:2]
    start_x, start_y = 0, 0
    
    # 1. Center crop до квадрата (как в датасете)
    if h > w:
        # Вертикальное изображение - обрезаем сверху/снизу
        crop_size = w
        start_y = (h - crop_size) // 2
        cropped = frame[start_y:start_y + crop_size, :, :]
    elif w > h:
        # Горизонтальное изображение - обрезаем слева/справа
        crop_size = h
        start_x = (w - crop_size) // 2
        cropped = frame[:, start_x:start_x + crop_size, :]
    else:
        # Уже квадрат
        cropped = frame
    
    # 2. Resize до целевого размера
    if cropped.shape[:2] != (target_size, target_size):
        resized = cv2.resize(cropped, (target_size, target_size))
    else:
        resized = cropped
    
    # 3. Добавляем черные полосы сверху/снизу для соответствия датасету
    # Датасет имеет ~15% черных полос сверху и снизу
    padding_height = int(target_size * 0.075)  # 7.5% сверху и снизу
    
    if padding_height > 0:
        # Создаем изображение с черными полосами
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        content_start = padding_height
        content_end = target_size - padding_height
        content_height = content_end - content_start
        
        # Масштабируем контент под доступную область
        content_resized = cv2.resize(resized, (target_size, content_height))
        padded[content_start:content_end, :, :] = content_resized
        
        final_img = padded
    else:
        final_img = resized
    
    # Для обратного преобразования координат
    transform_meta = {
        'original_size': (w, h),
        'crop_box': (start_x, start_y, crop_size, crop_size), # (x, y, width, height)
        'resize_target': target_size,
        'final_content_box': (0, padding_height, target_size, target_size - 2 * padding_height) # (x, y, width, height)
    }

    return {
        'img': final_img,
        'method': 'center_crop_with_padding',
        'transform_meta': transform_meta
    }


def letterbox_resize(frame: np.ndarray, target_size: int = 960) -> Dict[str, Any]:
    """
    Стандартная letterbox предобработка (как сейчас в системе)
    Сохраняет пропорции, добавляет padding
    """
    h, w = frame.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Padding до квадрата
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    img = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return {
        'img': img,
        'method': 'letterbox',
        'scale': scale,
        'pad': (top, bottom, left, right),
        'original_size': (w, h)
    }


def adaptive_preprocess(frame: np.ndarray, target_size: int = 960, 
                       force_method: str = None) -> Dict[str, Any]:
    """
    Адаптивная предобработка:
    - Для видео с черными полосами: удаляем их и применяем center_crop
    - Для обычного видео: используем letterbox
    """
    if force_method == 'center_crop':
        return center_crop_resize(frame, target_size)
    elif force_method == 'letterbox':
        return letterbox_resize(frame, target_size)
    
    # Автоматическое определение
    h, w = frame.shape[:2]
    
    # Проверяем наличие черных полос
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)
    
    # Ищем черные полосы сверху/снизу
    top_black_rows = 0
    bottom_black_rows = 0
    threshold = 15
    
    # Сверху
    for i, val in enumerate(row_mean):
        if val < threshold:
            top_black_rows += 1
        else:
            break
    
    # Снизу
    for i, val in enumerate(reversed(row_mean)):
        if val < threshold:
            bottom_black_rows += 1
        else:
            break
    
    # Если есть значительные черные полосы (>5% изображения)
    total_black = top_black_rows + bottom_black_rows
    if total_black > h * 0.05:
        # Удаляем черные полосы и применяем center crop
        y0 = top_black_rows
        y1 = h - bottom_black_rows
        cropped_frame = frame[y0:y1, :, :]
        return center_crop_resize(cropped_frame, target_size)
    else:
        # Используем letterbox для сохранения пропорций
        return letterbox_resize(frame, target_size)
