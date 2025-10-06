import cv2
import numpy as np
from typing import Tuple, Dict, Any


def hsv_filter(frame: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> np.ndarray:
    """Apply HSV threshold and return mask (uint8 0/255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_np, upper_np)
    return mask


def preprocess_for_model(frame: np.ndarray, target_size: int = 960, use_hsv: bool = False, hsv_bounds: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = ((0,0,0),(179,255,255))) -> Dict[str, Any]:
    """Return dict with 'img' ready for model and optional 'mask'.

    - Resizes keeping aspect ratio and pads to square if needed.
    - Optionally computes HSV mask and applies morphological cleaning.
    """
    h, w = frame.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    # pad to square
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

    out = {'img': img, 'scale': scale, 'pad': (top, bottom, left, right)}
    if use_hsv:
        mask = hsv_filter(frame, hsv_bounds[0], hsv_bounds[1])
        # morphological open/close to reduce noise
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        out['mask'] = mask
    return out


def map_polys_to_original(polys, scale: float, pad: Tuple[int,int,int,int], orig_size: Tuple[int,int]) -> list:
    """Map list of polygon points from processed image coords back to original image coords.

    polys: iterable of iterable of (x,y) in processed image pixel coordinates.
    scale: scale used to resize original -> resized.
    pad: (top, bottom, left, right) used when padding to target_size.
    orig_size: (width, height) of original frame.
    Returns list of polygons as list of (x,y) floats in original image coords.
    """
    top, bottom, left, right = pad
    ow, oh = orig_size
    mapped = []
    for poly in polys:
        pts = []
        try:
            for p in poly:
                x_proc = float(p[0])
                y_proc = float(p[1])
                x_resized = x_proc - float(left)
                y_resized = y_proc - float(top)
                x_orig = x_resized / max(1e-6, float(scale))
                y_orig = y_resized / max(1e-6, float(scale))
                # clamp
                x_orig = max(0.0, min(float(ow-1), x_orig))
                y_orig = max(0.0, min(float(oh-1), y_orig))
                pts.append((x_orig, y_orig))
        except Exception:
            continue
        if pts:
            mapped.append(pts)
    return mapped


def map_polys_from_center_crop(polys, transform_meta: Dict[str, Any]) -> list:
    """
    Maps polygons from the center_crop_resize processed image back to original coordinates.

    Args:
        polys: List of polygons from model output.
        transform_meta: Metadata dictionary from center_crop_resize.

    Returns:
        List of polygons mapped to original image coordinates.
    """
    orig_w, orig_h = transform_meta['original_size']
    crop_x, crop_y, crop_w, crop_h = transform_meta['crop_box']
    resize_target = transform_meta['resize_target']
    scale_factor = transform_meta['scale_factor']

    mapped_polys = []

    for poly in polys:
        new_poly = []
        for x_proc, y_proc in poly:
            # Упрощенная версия без padding:
            # 1. Reverse resize from (resize_target, resize_target) to (crop_w, crop_h)
            y_in_cropped = y_proc * (crop_h / resize_target)
            x_in_cropped = x_proc * (crop_w / resize_target)

            # 2. Reverse crop
            y_orig = y_in_cropped + crop_y
            x_orig = x_in_cropped + crop_x

            # Clamp to original image dimensions
            x_orig = max(0.0, min(float(orig_w - 1), x_orig))
            y_orig = max(0.0, min(float(orig_h - 1), y_orig))

            new_poly.append((x_orig, y_orig))
        
        if new_poly:
            mapped_polys.append(new_poly)

    return mapped_polys


def center_crop_resize(frame: np.ndarray, target_size: int = 640) -> Dict[str, Any]:
    """Центрирует кадр, приводит его к квадрату target_size и возвращает метаданные."""
    h, w = frame.shape[:2]
    crop_size = min(h, w)
    start_x = max(0, (w - crop_size) // 2)
    start_y = max(0, (h - crop_size) // 2)
    cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # Упрощенная версия без дополнительного padding для ускорения
    if cropped.shape[0] != target_size:
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        final_img = resized
    else:
        final_img = cropped

    transform_meta = {
        'original_size': (w, h),
        'crop_box': (start_x, start_y, crop_size, crop_size),
        'resize_target': target_size,
        'scale_factor': float(target_size) / crop_size,
        'pad_top': 0, 'pad_bottom': 0, 'pad_left': 0, 'pad_right': 0
    }

    return {
        'img': final_img,
        'method': 'center_crop_with_padding',
        'transform_meta': transform_meta,
    }


def letterbox_resize(frame: np.ndarray, target_size: int = 960) -> Dict[str, Any]:
    """Выполняет letterbox-ресайз с сохранением пропорций и полями."""
    h, w = frame.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    return {
        'img': img,
        'method': 'letterbox',
        'scale': scale,
        'pad': (top, bottom, left, right),
        'original_size': (w, h),
    }


def adaptive_preprocess(
    frame: np.ndarray,
    target_size: int = 960,
    force_method: str | None = None,
) -> Dict[str, Any]:
    """Подбирает стратегию предобработки: центр-кроп или letterbox."""
    if force_method == 'center_crop':
        return center_crop_resize(frame, target_size)
    if force_method == 'letterbox':
        return letterbox_resize(frame, target_size)

    h, _w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)

    top_black_rows = 0
    for value in row_mean:
        if value < 15:
            top_black_rows += 1
        else:
            break

    bottom_black_rows = 0
    for value in reversed(row_mean):
        if value < 15:
            bottom_black_rows += 1
        else:
            break

    total_black = top_black_rows + bottom_black_rows
    if total_black > h * 0.05:
        y0 = top_black_rows
        y1 = h - bottom_black_rows
        cropped_frame = frame[y0:y1, :, :]
        return center_crop_resize(cropped_frame, target_size)

    return letterbox_resize(frame, target_size)
