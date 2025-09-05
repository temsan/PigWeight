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


