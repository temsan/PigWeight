import os
import logging
import cv2
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional, Generator, Any, List, Deque, Tuple
import colorsys
from datetime import datetime, timedelta
import csv
from collections import deque
import abc
import subprocess

# Импортируем новый единый процессор
try:
    from core.processor import get_processor, ProcessingOptions, FrameResult
    HAVE_UNIFIED_PROCESSOR = True
    logging.info("✅ Unified processor available")
except ImportError as e:
    HAVE_UNIFIED_PROCESSOR = False
    logging.warning(f"⚠️ Unified processor not available: {e}, using legacy processors")

# Performance logging
perf_logger = logging.getLogger("perf.api")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response, JSONResponse, FileResponse
try:
    from fastapi.responses import ORJSONResponse
except Exception:
    ORJSONResponse = JSONResponse  # type: ignore
try:
    import orjson as _orjson  # noqa: F401
    _HAVE_ORJSON = True
except Exception:
    _HAVE_ORJSON = False
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

import numpy as np
from fastapi import Body, Request
import contextlib
from contextlib import asynccontextmanager
import math
from zoneinfo import ZoneInfo

# APScheduler for robust cron scheduling
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:
    AsyncIOScheduler = None
    CronTrigger = None

# Изолированные воркеры: PyAV (предпочтительно) и OpenCV (fallback)
import sys as _sys
_CUR_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CUR_DIR.parent
if str(_ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_DIR))
try:
    from api.av_worker import AVIsolate  # запуск как пакет: python -m api.app
except Exception:
    try:
            from .av_worker import AVIsolate  # относительный импорт внутри пакета
    except Exception:
        from av_worker import AVIsolate   # запуск как файл: python api/app.py

# --- Bootstrap ---
BASE_DIR = Path(__file__).parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

# Импорт упрощенной системы логирования
try:
    from core.config import setup_logging
    logger = setup_logging(debug=os.getenv("DEBUG", "false").lower() == "true")
except ImportError:
    # Fallback если core.config недоступен
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("api")

# --- Config from environment ---
CAM_DEFAULT = os.getenv("CAM_DEFAULT", "rtsp://admin:Qwerty.123@10.15.6.24/1/1")
CAM_URL = os.getenv("CAM_URL", CAM_DEFAULT)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
TARGET_FPS = float(os.getenv("FPS", "12"))
BOUNDARY = "frame"

# Lines positions (normalized) can be tuned via env; defaults near edges
def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

try:
    LINE_LEFT_X = _clamp01(os.getenv("LINE_LEFT_X", "0.25"))
except Exception:
    LINE_LEFT_X = 0.25
try:
    LINE_RIGHT_X = _clamp01(os.getenv("LINE_RIGHT_X", "0.75"))
except Exception:
    LINE_RIGHT_X = 0.75
if LINE_LEFT_X > LINE_RIGHT_X:
    LINE_LEFT_X, LINE_RIGHT_X = LINE_RIGHT_X, LINE_LEFT_X
if (LINE_RIGHT_X - LINE_LEFT_X) < 0.05:
    mid = 0.5 * (LINE_LEFT_X + LINE_RIGHT_X)
    LINE_LEFT_X = max(0.0, mid - 0.025)
    LINE_RIGHT_X = min(1.0, mid + 0.025)

# Model config (строго .pt из каталога ./models)
DETECTION_MODE = os.getenv("DETECTION_MODE", "pig-only").lower()
# Разрешаем указывать путь единой переменной MODEL_PATH, либо старым PIG_MODEL_PATH
_ENV_MODEL_PATH = os.getenv("MODEL_PATH")
PIG_MODEL_PATH = os.getenv("PIG_MODEL_PATH")
PIG_CLASS_ID = int(os.getenv("PIG_CLASS_ID", "0"))

# Выбор эффективной модели и классов
if DETECTION_MODE == "pig-only":
    # Для режима pig-only принимаем MODEL_PATH (приоритет) или PIG_MODEL_PATH
    _chosen_path = _ENV_MODEL_PATH or PIG_MODEL_PATH
    if not _chosen_path:
        raise RuntimeError("MODEL_PATH или PIG_MODEL_PATH не задан в .env")
    _p = Path(_chosen_path)
    if not _p.exists():
        raise RuntimeError(f"Файл модели не найден: {_chosen_path}")
    MODEL_PATH = _chosen_path
    TARGET_CLASS_IDS = {PIG_CLASS_ID}
else:
    TARGET_CLASS_IDS = set(map(int, os.getenv("TARGET_CLASS_IDS", "20,17,19").split(",")))
    # В остальных режимах ожидаем путь в MODEL_PATH
    if not _ENV_MODEL_PATH:
        raise RuntimeError("MODEL_PATH не задан в .env для текущего DETECTION_MODE")
    if not Path(_ENV_MODEL_PATH).exists():
        raise RuntimeError(f"Файл модели не найден: {_ENV_MODEL_PATH}")
    MODEL_PATH = _ENV_MODEL_PATH
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.30"))

AVG_WINDOW = int(os.getenv("AVG_WINDOW", "20"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))

# Оптимизированная предобработка для соответствия датасету
USE_OPTIMIZED_PREPROCESSING = os.getenv("USE_OPTIMIZED_PREPROCESSING", "true").lower() == "true"
PREPROCESSING_METHOD = os.getenv("PREPROCESSING_METHOD", "adaptive")  # adaptive, center_crop, letterbox



# Inference device (GPU/CPU)
try:
    import torch as _torch
    _cuda_ok = bool(getattr(_torch, 'cuda', None) and _torch.cuda.is_available())
except Exception:
    _cuda_ok = False
DEVICE = os.getenv("DEVICE") or ("cuda:0" if _cuda_ok else "cpu")
USE_HALF = (os.getenv("USE_HALF", "true").lower() == "true") and DEVICE.startswith("cuda")

# Auto-fallback: if torch exists but CUDA not available, force CPU and disable half
try:
    import torch as _torch_check
    if DEVICE and DEVICE.startswith('cuda') and not (_torch_check.cuda.is_available() if hasattr(_torch_check, 'cuda') else False):
        logger.warning("Requested DEVICE=%s but CUDA not available. Falling back to CPU (disabling half).", DEVICE)
        DEVICE = 'cpu'
        USE_HALF = False
except Exception:
    # if torch isn't importable here, leave env values as-is; ModelAdapter will warn later
    pass

# Inference device (GPU/CPU) and torch settings
try:
    import torch as _torch
    if hasattr(_torch.backends, 'cudnn'):
        try:
            _torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    try:
        _torch.set_grad_enabled(False)
    except Exception:
        pass
    _cuda_ok = bool(getattr(_torch, 'cuda', None) and _torch.cuda.is_available())
except Exception:
    _cuda_ok = False
DEVICE = os.getenv("DEVICE") or ("cuda:0" if _cuda_ok else "cpu")
USE_HALF = (os.getenv("USE_HALF", "true").lower() == "true") and DEVICE.startswith("cuda")

# --- Counting/estimation parameters ---
COUNT_WINDOW_SEC = float(os.getenv("COUNT_WINDOW_SEC", "10.0"))
COUNT_DECAY_HALFLIFE_SEC = float(os.getenv("COUNT_DECAY_HALFLIFE_SEC", "4.0"))
COUNT_SOFTMAX_BETA = float(os.getenv("COUNT_SOFTMAX_BETA", "0.8"))
CROSS_COOLDOWN_SEC = float(os.getenv("CROSS_COOLDOWN_SEC", "2.0"))

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RECORDS_DIR = BASE_DIR / "records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# --- Helper Functions ---

def cameras_from_env() -> Dict[str, str]:
    """Collect camera RTSP URLs from environment variables.

    Supported patterns (case-sensitive):
      - CAM_CH<digits>=rtsp://...
      - If no CAM_CH* are present, fallback to CAM_URL or CAM_DEFAULT as cam101
    Returns mapping like {"cam101": "rtsp://...", ...} ordered by channel number.
    """
    cams: Dict[str, str] = {}
    items = []
    for key, val in os.environ.items():
        if not val:
            continue
        if key.startswith("CAM_CH"):
            suf = key[6:]  # after CAM_CH
            if suf.isdigit():
                try:
                    items.append((int(suf), val))
                except Exception:
                    continue
    if items:
        for num, url in sorted(items, key=lambda t: t[0]):
            cams[f"cam{num}"] = url
        return cams
    # Fallback
    url = os.getenv("CAM_URL") or os.getenv("CAM_DEFAULT")
    if url:
        cams["cam101"] = url
    return cams

def encode_jpeg(frame, quality: int = None) -> bytes:
    q = quality or JPEG_QUALITY
    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY, q,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
        cv2.IMWRITE_JPEG_PROGRESSIVE, 1
    ]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        return b""
    return buf.tobytes()

def _open_file_cap_local(path: str):
    backends = [
        getattr(cv2, 'CAP_MSMF', 1400),
        getattr(cv2, 'CAP_DSHOW', 700),
        getattr(cv2, 'CAP_ANY', 0),
    ]
    last_err = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(str(path), backend)
            if not cap or not cap.isOpened():
                try: cap.release()
                except Exception: pass
                last_err = f"cannot open with backend={backend}"
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
            return cap, {"fps": fps, "frame_count": frame_count, "duration": duration, "type": "local", "backend": int(backend)}
        except Exception as e:
            last_err = str(e)
            try: cap.release()
            except Exception: pass
            continue
    return None, {"error": last_err or "all backends failed"}

AVW: Optional[AVIsolate] = None
def get_av() -> AVIsolate:
    global AVW
    if AVW is None:
        AVW = AVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=TARGET_FPS)
    return AVW

def _av_safe_call(method_name: str, *args, **kwargs):
    try:
        avw = get_av()
        method = getattr(avw, method_name, None)
        if not method:
            raise AttributeError(f"AVIsolate lacks {method_name}")
        return method(*args, **kwargs)
    except Exception:
        try:
            global AVW
            AVW = AVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=TARGET_FPS)
            method = getattr(AVW, method_name, None)
            if not method:
                raise AttributeError(f"AVIsolate lacks {method_name}")
            return method(*args, **kwargs)
        except Exception as e2:
            raise e2

def av_open_rtsp(stream_id: str, url: str) -> Dict[str, Any]:
    return _av_safe_call('open_rtsp', stream_id, url)

def av_open_file(stream_id: str, path: str) -> Dict[str, Any]:
    return _av_safe_call('open_file', stream_id, path)

def av_close(stream_id: str) -> None:
    try:
        _av_safe_call('close', stream_id)
    except Exception:
        pass

def ocv_close(stream_id: str) -> None:
    try:
        _av_safe_call('close', stream_id)
    except Exception:
        pass

def av_read_jpeg(stream_id: str, timeout: float = 1.0) -> Optional[bytes]:
    return _av_safe_call('read_jpeg', stream_id, timeout=timeout)

def av_seek_read_jpeg(stream_id: str, t: float, timeout: float = 2.0) -> Optional[bytes]:
    return _av_safe_call('seek_read_jpeg', stream_id, t, timeout=timeout)
def av_meta(stream_id: str) -> Dict[str, Any]:
    return _av_safe_call('meta', stream_id)

def ocv_probe_file(path: str) -> Dict[str, Any]:
    """Open a file temporarily in the worker to fetch meta, then close it."""
    tmp_id = f"probe_{int(time.time()*1000)}"
    try:
        meta = av_open_file(tmp_id, path)
        # ensure meta returns expected keys
        fps = float(meta.get("fps", 0.0) or 0.0)
        dur = float(meta.get("duration", 0.0) or 0.0)
        fc = int(meta.get("frame_count", 0) or 0)
        return {"fps": fps, "duration": dur, "frame_count": fc}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            ocv_close(tmp_id)
        except Exception:
            pass
def ocv_meta(stream_id: str) -> Dict[str, Any]:
    return _av_safe_call('meta', stream_id)

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, dist_weight=0.2):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.dist_weight = float(dist_weight)
        self.next_id = 1
        # id -> {bbox, age, cx, cy}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _center(b):
        x1, y1, x2, y2 = b
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _hungarian(self, cost: List[List[float]]):
        # Minimal Hungarian for small N (<= 50). Returns list of (row->col) or -1.
        n = max(len(cost), len(cost[0]) if cost else 0)
        # pad to square
        C = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                C[i][j] = cost[i][j] if i < len(cost) and j < len(cost[0]) else 1e6
        u = [0.0]*(n+1)
        v = [0.0]*(n+1)
        p = [0]*(n+1)
        way = [0]*(n+1)
        for i in range(1, n+1):
            p[0] = i
            j0 = 0
            minv = [float('inf')]*(n+1)
            used = [False]*(n+1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, n+1):
                    if used[j]:
                        continue
                    cur = C[i0-1][j-1]-u[i0]-v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(0, n+1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        ans = [-1]*n
        for j in range(1, n+1):
            if p[j] != 0 and p[j]-1 < len(cost):
                ans[p[j]-1] = j-1 if j-1 < len(cost[0]) else -1
        return ans

    def update(self, detections: List[Dict[str, Any]]):
        det_bboxes = [d['bbox'] for d in detections]
        det_centers = [self._center(b) for b in det_bboxes]
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        track_centers = [(self.tracks[tid].get('cx'), self.tracks[tid].get('cy')) for tid in track_ids]

        if det_bboxes and track_bboxes:
            # Build cost = (1 - IoU) + w * normalized center distance
            cost = []
            for i, bb in enumerate(det_bboxes):
                row = []
                cx, cy = det_centers[i]
                for k, tb in enumerate(track_bboxes):
                    iou = self._iou(bb, tb)
                    tcx, tcy = track_centers[k]
                    if tcx is None or tcy is None:
                        d = 0.0
                    else:
                        dx = cx - tcx
                        dy = cy - tcy
                        # normalize by size of union box diagonal to prefer nearby
                        ux1, uy1 = min(bb[0], tb[0]), min(bb[1], tb[1])
                        ux2, uy2 = max(bb[2], tb[2]), max(bb[3], tb[3])
                        diag = max(1.0, ((ux2-ux1)**2 + (uy2-uy1)**2) ** 0.5)
                        d = ((dx*dx + dy*dy) ** 0.5) / diag
                    # penalize low IoU heavily
                    base = (1.0 - iou)
                    if iou < self.iou_threshold:
                        base += 10.0
                    row.append(base + self.dist_weight * d)
                cost.append(row)
            assign = self._hungarian(cost)
            used_tracks = set()
            det_to_id: Dict[int, int] = {}
            for i, j in enumerate(assign):
                if j is None or j < 0 or i >= len(det_bboxes) or j >= len(track_ids):
                    continue
                # validate IoU threshold
                if self._iou(det_bboxes[i], track_bboxes[j]) < self.iou_threshold:
                    continue
                tid = track_ids[j]
                det_to_id[i] = tid
                used_tracks.add(tid)
                cx, cy = det_centers[i]
                self.tracks[tid].update({'bbox': det_bboxes[i], 'cx': cx, 'cy': cy, 'age': self.max_age})
            # New tracks
            for i in range(len(det_bboxes)):
                if i not in det_to_id:
                    tid = self.next_id
                    self.next_id += 1
                    cx, cy = det_centers[i]
                    self.tracks[tid] = {'bbox': det_bboxes[i], 'cx': cx, 'cy': cy, 'age': self.max_age}
                    det_to_id[i] = tid
            # Age unmatched tracks
            rm = []
            for tid in list(self.tracks.keys()):
                if tid not in used_tracks and self.tracks[tid]['age'] > 0:
                    # Агрессивнее уменьшаем TTL, чтобы быстрее освобождались id
                    self.tracks[tid]['age'] -= 2
                    if self.tracks[tid]['age'] <= 0:
                        rm.append(tid)
            for tid in rm:
                self.tracks.pop(tid, None)
            ids_out = [det_to_id[i] for i in range(len(det_bboxes))]
        else:
            # No existing tracks → create new for all dets
            ids_out = []
            for i, b in enumerate(det_bboxes):
                tid = self.next_id
                self.next_id += 1
                cx, cy = det_centers[i]
                self.tracks[tid] = {'bbox': b, 'cx': cx, 'cy': cy, 'age': self.max_age}
                ids_out.append(tid)
        return [{**detections[i], 'id': ids_out[i]} for i in range(len(detections))]

# --- Unified Video Stream Architecture ---

class VideoStream(abc.ABC):
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.running = False
        self.last_jpeg: Optional[bytes] = None
        self.lock = asyncio.Lock()
        self.model = None
        self.model_loaded = False
        self.last_count = 0
        # Оценка количества: максимум по окну и монотоничный отчёт (не прыгает)
        self.window_max = WindowMaxEstimator(COUNT_WINDOW_SEC)
        self.reported_count = 0
        # flow counters and per-track state
        self.left_in = 0
        self.right_in = 0
        # directional net flows for UI: left (+enter_left, -exit_left), right (+exit_right, -enter_right)
        self.left_flow = 0
        self.right_flow = 0
        self._track_prev_x: Dict[int, float] = {}
        self._track_last_side_time: Dict[Tuple[int, str], float] = {}
        self.last_masks = []
        # Сохранение размеров и позиций для маппинга масок
        self._last_mask_ref_size = None  # (width, height) оригинального кадра
        self._last_mask_crop = None  # (y0, y1) границы обрезки
        self.line_positions = {}  # Сохранение позиций линий для конкретных файлов
        self.tracker = SimpleTracker()
        self._infer_task: Optional[asyncio.Task] = None
        self._stream_task: Optional[asyncio.Task] = None
        # Recent crossings to render on UI (store as list of dicts with ts)
        self._recent_crossings: List[Dict[str, Any]] = []
        # Порядок событий пересечения слева и первое появление
        self._first_seen_order: Dict[int, int] = {}
        self._arrival_counter: int = 1
        self._left_cross_rank: Dict[int, int] = {}
        self._left_cross_counter: int = 1
        # Стабильная отображаемая метка для каждого track id (не прыгает)
        self._display_label_map: Dict[int, int] = {}
        self._next_display_label: int = 1
        # Session numbering and act-of-weighing metrics
        self._session_id_map: Dict[int, int] = {}
        self._next_session_label: int = 1
        self._act_seen_labels: set[int] = set()
        self._act_peak: int = 0
        self._act_start_ts: float = time.time()
        self._act_timeline: List[Dict[str, Any]] = []
        self._act_crossings: List[Dict[str, Any]] = []
        self._act_last_cross_ts: float = 0.0
        # таймлайн для дашборда: не писать чаще, чем раз в 0.5с
        self._last_timeline_ts: float = time.time()

    def _reset_act(self):
        self._session_id_map = {}
        self._next_session_label = 1
        self._act_seen_labels = set()
        self._act_peak = 0
        self._act_start_ts = time.time()
        self._act_timeline = []
        self._act_crossings = []
        self._act_last_cross_ts = 0.0
        self._last_timeline_ts = time.time()
        self._first_seen_order = {}
        self._arrival_counter = 1
        self._left_cross_rank = {}
        self._left_cross_counter = 1
        self._display_label_map = {}
        self._next_display_label = 1

    def _finalize_act_to_files(self):
        try:
            if not self._act_timeline:
                return
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            base = f"act_{self.stream_id}_{ts}"
            RECORDS_DIR.mkdir(parents=True, exist_ok=True)
            # JSON summary
            summary = {
                "stream_id": self.stream_id,
                "started_at": self._act_start_ts,
                "finished_at": time.time(),
                "duration_sec": float(max(0.0, time.time() - self._act_start_ts)),
                "seen_total": int(len(self._act_seen_labels)),
                "peak_concurrent": int(self._act_peak),
                "flow": {"left_in": int(getattr(self, 'left_in', 0)), "right_in": int(getattr(self, 'right_in', 0))},
                "timeline": self._act_timeline,
                "crossings": self._act_crossings,
            }
            with open(RECORDS_DIR / f"{base}.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            # SVG chart (lightweight)
            try:
                W, H = 1200, 360
                L, R, T, B = 60, 30, 20, 40
                max_t = max((p.get("t", 0.0) for p in self._act_timeline), default=1.0)
                max_c = max((p.get("count_est", 0) for p in self._act_timeline), default=1)
                def sx(t):
                    return L + int((t / max(1e-6, max_t)) * (W - L - R))
                def sy(c):
                    return T + int((1.0 - (c / max(1, max_c))) * (H - T - B))
                # path for count
                path = []
                first = True
                for p in self._act_timeline:
                    x = sx(float(p.get("t", 0.0)))
                    y = sy(float(p.get("count_est", 0)))
                    path.append((x, y, first))
                    first = False
                def path_d(items):
                    parts = []
                    for x, y, first_flag in items:
                        if first_flag:
                            parts.append(f"M{x},{y}")
                        else:
                            parts.append(f"L{x},{y}")
                    return " ".join(parts)
                # crossings circles
                circles = []
                for c in self._act_crossings[-150:]:  # cap for size
                    x = sx(float(c.get("t", 0.0)))
                    y = sy(float(c.get("count_est", 0.0)))
                    color = '#2c7be5' if c.get('side') == 'left' else '#51cf66'
                    circles.append(f"<circle cx='{x}' cy='{y}' r='3' fill='{color}' fill-opacity='0.9' />")
                svg = [
                    f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>",
                    "<defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>",
                    "<stop offset='0%' stop-color='rgba(44,123,229,0.35)'/>",
                    "<stop offset='100%' stop-color='rgba(44,123,229,0.00)'/></linearGradient></defs>",
                    f"<rect width='{W}' height='{H}' fill='white'/>",
                    f"<rect x='{L}' y='{T}' width='{W-L-R}' height='{H-T-B}' fill='rgba(240,245,252,0.8)' stroke='rgba(60,90,140,0.2)'/>",
                    f"<path d='{path_d(path)}' stroke='#2c7be5' fill='none' stroke-width='2'/>",
                    *circles,
                    f"<text x='{L}' y='{H-8}' font-size='12' fill='#33507a'>t, s</text>",
                    f"<text x='{W-52}' y='{T+14}' font-size='12' fill='#33507a'>count</text>",
                    "</svg>"
                ]
                with open(RECORDS_DIR / f"{base}.svg", "w", encoding="utf-8") as fsvg:
                    fsvg.write("\n".join(svg))
            except Exception:
                pass
            # Markdown отчёт
            try:
                md_lines = []
                md_lines.append(f"# Отчёт акта взвешивания — {self.stream_id}")
                md_lines.append("")
                md_lines.append(f"- Начало: {datetime.fromtimestamp(summary['started_at']).isoformat(sep=' ', timespec='seconds')}")
                md_lines.append(f"- Окончание: {datetime.fromtimestamp(summary['finished_at']).isoformat(sep=' ', timespec='seconds')}")
                md_lines.append(f"- Длительность: {summary['duration_sec']:.1f} с")
                md_lines.append("")
                md_lines.append("## Показатели")
                md_lines.append("")
                md_lines.append(f"- Вход слева: {summary['flow']['left_in']}")
                md_lines.append(f"- Выход справа: {summary['flow']['right_in']}")
                md_lines.append(f"- Пиковое количество одновременно: {summary['peak_concurrent']}")
                md_lines.append(f"- Всего уникально замечено: {summary['seen_total']}")
                md_lines.append("")
                md_lines.append("## График количества по времени")
                md_lines.append("")
                md_lines.append(f"![timeline]({base}.svg)")
                md_lines.append("")
                md_lines.append("## Данные")
                md_lines.append("")
                md_lines.append(f"`{base}.json`")
                with open(RECORDS_DIR / f"{base}.md", "w", encoding="utf-8") as fmd:
                    fmd.write("\n".join(md_lines))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Finalize act save error for {self.stream_id}: {e}")

    

    def _update_line_counters(self, ids: List[int], centers_x: List[float], centers_y: Optional[List[float]] = None):
        """Count entries from left/right by crossing vertical lines at 0.25 and 0.75 of width.
        A crossing is counted when center crosses from <0.25 to >=0.25 (left_in) or
        from >0.75 to <=0.75 (right_in). Directional and with cooldown to avoid bouncing.
        """
        now = time.time()
        # Линии сдвинуты к краям (настраиваются через env)
        L = float(LINE_LEFT_X)
        R = float(LINE_RIGHT_X)
        cy_iter: List[float] = centers_y if centers_y is not None else [0.5] * len(centers_x)
        for tid, cx, cy in zip(ids, centers_x, cy_iter):
            if tid is None:
                continue
            prev = self._track_prev_x.get(tid)
            prev_y = getattr(self, '_track_prev_y', {}).get(tid)
            if not hasattr(self, '_track_prev_y'):
                self._track_prev_y = {}
            if not hasattr(self, '_track_is_inside'):
                self._track_is_inside = {}
            prev_inside = bool(self._track_is_inside.get(tid, (prev is not None and L <= prev <= R)))
            cur_inside = bool(L <= cx <= R)

            def _interp_y(px, py, qx, qy, lx):
                try:
                    t = (float(lx) - float(px)) / (float(qx) - float(px))
                    return max(0.0, min(1.0, float(py) + t * (float(qy) - float(py))))
                except Exception:
                    return float(cy)

            if prev is not None and prev_y is not None:
                # enter events
                if (not prev_inside) and cur_inside:
                    if prev < L <= cx:
                        key = (tid, 'enter_left')
                        if now - self._track_last_side_time.get(key, 0.0) >= (CROSS_COOLDOWN_SEC * 0.6):
                            self.left_in += 1
                            self.left_flow += 1
                            self._track_last_side_time[key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, L)
                            self._recent_crossings.append({"id": int(tid), "side": "left", "mode": "enter", "x": float(L), "y": float(y_at), "ts": float(now)})
                            # долгосрочный лог для дашборда
                            try:
                                self._act_crossings.append({
                                    "id": int(tid), "side": "left", "mode": "enter",
                                    "t": float(max(0.0, now - self._act_start_ts)),
                                    "x": float(L), "y": float(y_at),
                                    "count_est": int(self.reported_count)
                                })
                                # порядковый номер пересечения слева
                                if int(tid) not in self._left_cross_rank:
                                    self._left_cross_rank[int(tid)] = self._left_cross_counter
                                    self._left_cross_counter += 1
                            except Exception:
                                pass
                    elif prev > R >= cx:
                        key = (tid, 'enter_right')
                        if now - self._track_last_side_time.get(key, 0.0) >= (CROSS_COOLDOWN_SEC * 0.6):
                            self.right_in += 1
                            self.right_flow -= 1
                            self._track_last_side_time[key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, R)
                            self._recent_crossings.append({"id": int(tid), "side": "right", "mode": "enter", "x": float(R), "y": float(y_at), "ts": float(now)})
                            try:
                                self._act_crossings.append({
                                    "id": int(tid), "side": "right", "mode": "enter",
                                    "t": float(max(0.0, now - self._act_start_ts)),
                                    "x": float(R), "y": float(y_at),
                                    "count_est": int(self.reported_count)
                                })
                            except Exception:
                                pass
                # exit events
                if prev_inside and (not cur_inside):
                    if cx < L <= prev:
                        key = (tid, 'exit_left')
                        if now - self._track_last_side_time.get(key, 0.0) >= (CROSS_COOLDOWN_SEC * 0.6):
                            # treat as -1 on left side
                            self.left_in = max(0, self.left_in - 1)
                            self.left_flow -= 1
                            self._track_last_side_time[key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, L)
                            self._recent_crossings.append({"id": int(tid), "side": "left", "mode": "exit", "x": float(L), "y": float(y_at), "ts": float(now)})
                            try:
                                self._act_crossings.append({
                                    "id": int(tid), "side": "left", "mode": "exit",
                                    "t": float(max(0.0, now - self._act_start_ts)),
                                    "x": float(L), "y": float(y_at),
                                    "count_est": int(self.reported_count)
                                })
                            except Exception:
                                pass
                    elif cx > R >= prev:
                        key = (tid, 'exit_right')
                        if now - self._track_last_side_time.get(key, 0.0) >= (CROSS_COOLDOWN_SEC * 0.6):
                            self.right_in = max(0, self.right_in - 1)
                            self.right_flow += 1
                            self._track_last_side_time[key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, R)
                            self._recent_crossings.append({"id": int(tid), "side": "right", "mode": "exit", "x": float(R), "y": float(y_at), "ts": float(now)})
                            try:
                                self._act_crossings.append({
                                    "id": int(tid), "side": "right", "mode": "exit",
                                    "t": float(max(0.0, now - self._act_start_ts)),
                                    "x": float(R), "y": float(y_at),
                                    "count_est": int(self.reported_count)
                                })
                            except Exception:
                                pass

            self._track_prev_x[tid] = cx
            self._track_prev_y[tid] = cy
            self._track_is_inside[tid] = cur_inside
        try:
            cutoff = now - 2.0
            if self._recent_crossings:
                self._recent_crossings = [c for c in self._recent_crossings if c.get("ts", 0) >= cutoff]
        except Exception:
            pass

    async def _infer_loop(self):
        # Делегируем в глобальную реализацию, чтобы избежать конфликтов hot-reload
        return await _global_infer_loop(self)

    async def get_jpeg(self) -> Optional[bytes]:
        async with self.lock:
            lj = self.last_jpeg
            # Нормализуем формат: всегда bytes
            if isinstance(lj, dict):
                try:
                    data = lj.get('jpeg')
                    if isinstance(data, (bytes, bytearray)):
                        return bytes(data)
                    else:
                        return None
                except Exception:
                    return None
            elif isinstance(lj, (bytes, bytearray)):
                return bytes(lj)
            else:
                return None


class WeightedMaxEstimator:
    def __init__(self, window_sec: float, half_life_sec: float, beta: float):
        from collections import deque
        self.window_sec = float(max(0.5, window_sec))
        self.half_life = float(max(0.1, half_life_sec))
        self.beta = float(max(0.0, beta))
        self.data: Deque[Tuple[float, int]] = deque()

    def update(self, t: float, count: int) -> float:
        self.data.append((float(t), int(max(0, count))))
        # drop old
        t0 = float(t)
        while self.data and (t0 - self.data[0][0]) > self.window_sec:
            self.data.popleft()
        if not self.data:
            return float(max(0, count))
        # recency decay
        import math
        lam = math.log(2.0) / self.half_life
        # softmax on counts (shifted by max for stability)
        maxc = max(c for _, c in self.data)
        num = 0.0
        den = 0.0
        for ti, ci in self.data:
            w_time = math.exp(-lam * max(0.0, t0 - ti))
            w_val = math.exp(self.beta * (ci - maxc))
            w = w_time * w_val
            num += ci * w
            den += w
        return (num / den) if den > 1e-9 else float(max(0, count))


class WindowMaxEstimator:
    """Простой максимум по скользящему окну времени."""
    def __init__(self, window_sec: float):
        from collections import deque
        self.window_sec = float(max(0.5, window_sec))
        self.data: Deque[Tuple[float, int]] = deque()

    def update(self, t: float, count: int) -> int:
        t = float(t)
        self.data.append((t, int(max(0, count))))
        # выкидываем устаревшие
        while self.data and (t - self.data[0][0]) > self.window_sec:
            self.data.popleft()
        if not self.data:
            return int(max(0, count))
        return int(max(c for _, c in self.data))

# NOTE: Глобальная реализация инференс-цикла; метод класса делегирует сюда
async def _global_infer_loop(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            try:
                if DEVICE:
                    self.model.to(DEVICE)
                if USE_HALF and hasattr(self.model, 'model'):
                    try:
                        self.model.model.half()
                    except Exception:
                        pass
            except Exception as _e:
                logger.warning(f"Model device/half setup warning: {_e}")
            self.model_loaded = True
            logger.info(f"YOLO model loaded: {MODEL_PATH}")
        except Exception as e:
            self.model_loaded = False
            logger.error(f"Failed to load YOLO model: {e}")
            try:
                await STREAM_MANAGER.broadcast(self.stream_id, {"type": "status", "inference": "disabled", "error": str(e)})
            except Exception:
                pass
            while self.running:
                await asyncio.sleep(1.0)
            return

        while self.running:
            try:
                jpeg = await self.get_jpeg()
                if jpeg:
                    # Исправляем проблему с типами данных
                    if isinstance(jpeg, dict):
                        # Если пришел dict вместо bytes, извлекаем jpeg данные
                        if 'jpeg' in jpeg:
                            jpeg_data = jpeg['jpeg']
                        else:
                            logger.error(f"Invalid jpeg data format: {type(jpeg)}")
                            continue
                    else:
                        jpeg_data = jpeg
                        
                    arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        H0, W0, _ = frame.shape
                        y0, y1 = 0, H0
                        proc = frame
                        # imgsz из переменной окружения (по умолчанию 960)
                        try:
                            _IMG_SIZE = int(os.getenv("IMG_SIZE", "960"))
                        except Exception:
                            _IMG_SIZE = 960
                        results = self.model.predict(proc, imgsz=_IMG_SIZE, conf=CONF_THRESHOLD, verbose=False, retina_masks=True)
                        r = results[0] if results else None
                        # Быстрая диагностика кадра
                        try:
                            frame_mean = float(np.mean(frame))
                            frame_size = (int(W0), int(H0))
                        except Exception:
                            frame_mean = None
                            frame_size = (int(W0), int(H0))

                        if r and hasattr(r, "masks") and r.masks is not None:
                            polys = r.masks.xy
                            self.last_count = len(polys)
                            # Build detections by bbox for simple tracking on cropped frame
                            dets = []
                            centroids_local: list[tuple[float, float]] = []
                            for m in polys:
                                if m is None or len(m) == 0:
                                    continue
                                xs = [float(p[0]) for p in m]
                                ys = [float(p[1]) for p in m]
                                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                                dets.append({'bbox': [x1, y1, x2, y2]})
                                # центроид по вершинам полигона (в координатах proc)
                                try:
                                    cx_m = sum(xs) / max(1, len(xs))
                                    cy_m = sum(ys) / max(1, len(ys))
                                except Exception:
                                    cx_m = 0.5 * (x1 + x2)
                                    cy_m = 0.5 * (y1 + y2)
                                centroids_local.append((cx_m, cy_m))
                            tracks = self.tracker.update(dets) if dets else []
                            ids = [t['id'] for t in tracks] if tracks else []
                            # Зафиксировать порядок первого появления (входа в кадр)
                            for tid in ids:
                                if tid not in self._first_seen_order:
                                    self._first_seen_order[tid] = self._arrival_counter
                                    self._arrival_counter += 1
                            # centers for flow counting (normalized X) — по центроидам масок
                            centers_x: List[float] = []
                            centers_y: List[float] = []
                            for (cxm, cym) in centroids_local:
                                centers_x.append(float(cxm) / float(W0))
                                centers_y.append((float(cym) + float(y0)) / float(H0))
                            # Присваиваем новые метки слева-направо, чтобы цифры были по порядку
                            if ids:
                                new_pairs: List[Tuple[int, float]] = []
                                for i, tid in enumerate(ids):
                                    if tid not in self._session_id_map:
                                        xnorm = centers_x[i] if i < len(centers_x) else 0.0
                                        new_pairs.append((tid, float(xnorm)))
                                new_pairs.sort(key=lambda t: t[1])
                                for tid, _ in new_pairs:
                                    if tid not in self._session_id_map:
                                        self._session_id_map[tid] = self._next_session_label
                                        self._next_session_label += 1
                            # Собираем последовательность меток по трекам текущего кадра
                            session_labels: List[int] = [self._session_id_map.get(tid, 0) for tid in ids]
                            # Подготовим стабильные отображаемые метки:
                            # 1) Для новых треков выдаём следующий номер (_next_display_label) и запоминаем в карте
                            # 2) Для уже виденных всегда используем прежнюю метку (не прыгает)
                            # 3) Для удобства человека дополнительно формируем ordered_labels,
                            #    но не подменяем стабильную карту (используем ordered_labels только если нужно)
                            try:
                                n = len(centroids_local)
                                ordered_labels = [0] * n
                                display_labels = [0] * n
                                # Назначаем стабильные метки
                                for i, tid in enumerate(ids):
                                    if tid not in self._display_label_map:
                                        self._display_label_map[tid] = self._next_display_label
                                        self._next_display_label += 1
                                    display_labels[i] = self._display_label_map.get(tid, 0)
                                # Дополнительно сформируем человеко-порядок по левому пересечению / первому появлению
                                order_idx = list(range(n))
                                def left_rank(i: int) -> int:
                                    try:
                                        tid = ids[i]
                                        if int(tid) in self._left_cross_rank:
                                            return int(self._left_cross_rank[int(tid)])
                                        return int(self._first_seen_order.get(tid, 1_000_000))
                                    except Exception:
                                        return 1_000_000
                                order_idx.sort(key=lambda i: (left_rank(i), -(centroids_local[i][0] if i < len(centroids_local) else 0.0)))
                                for rank, idx_i in enumerate(order_idx, start=1):
                                    ordered_labels[idx_i] = rank
                            except Exception:
                                ordered_labels = list(range(1, len(centroids_local) + 1))
                                display_labels = ordered_labels[:]
                            self._update_line_counters(ids, centers_x, centers_y)
                            # Normalize masks back to original frame
                            mapped: list[list[tuple[float, float]]] = []
                            for m in polys:
                                pts = []
                                for p in m:
                                    x = float(p[0])
                                    y = float(p[1]) + float(y0)
                                    pts.append((x / float(W0), y / float(H0)))
                                mapped.append(pts)
                            self.last_masks = mapped
                            try:
                                self._last_mask_ref_size = (int(W0), int(H0))
                                self._last_mask_crop = (int(y0), int(y1))
                            except Exception:
                                self._last_mask_ref_size = (int(W0), int(H0))
                                self._last_mask_crop = (0, int(H0))
                            # Update act-of-weighing metrics
                            cur_count = len(session_labels)
                            if cur_count > self._act_peak:
                                self._act_peak = cur_count
                            for lab in session_labels:
                                self._act_seen_labels.add(int(lab))
                        else:
                            self.last_count = 0
                            self.last_masks = []
                            try:
                                self._last_mask_ref_size = None
                                self._last_mask_crop = None
                            except Exception:
                                pass
                        # Статистический максимум: окно + монотоничность
                        wnd_max = self.window_max.update(time.time(), int(self.last_count))
                        est = max(self.reported_count, wnd_max)
                        self.reported_count = est
                        # Логируем таймлайн для дашборда не чаще 2 раз в секунду
                        try:
                            now_ts = time.time()
                            if (now_ts - getattr(self, '_last_timeline_ts', 0.0)) >= 0.5:
                                rel_t = float(max(0.0, now_ts - float(self._act_start_ts)))
                                self._act_timeline.append({"t": rel_t, "count_est": int(est)})
                                self._last_timeline_ts = now_ts
                        except Exception:
                            pass
                        payload = {
                            "type": "count_update",
                            "count": int(round(est)),
                            "debug": {
                                "masks": self.last_masks,
                                "count_raw": int(self.last_count),
                                "flow": {"left_in": self.left_in, "right_in": self.right_in, "left_flow": self.left_flow, "right_flow": self.right_flow},
                                "frame_mean": frame_mean,
                                "size": {"w": frame_size[0], "h": frame_size[1]}
                            }
                        }
                        # model meta for UI
                        try:
                            payload["debug"]["model"] = {
                                "path": str(MODEL_PATH),
                                "name": os.path.basename(str(MODEL_PATH)),
                                "device": str(DEVICE) if 'DEVICE' in globals() else 'cpu',
                                "half": bool(USE_HALF) if 'USE_HALF' in globals() else False,
                            }
                        except Exception:
                            pass
                        # include imgsz if known
                        try:
                            if '_IMG_SIZE' in locals():
                                payload["debug"]["imgsz"] = int(_IMG_SIZE)
                        except Exception:
                            pass
                        # include line positions for UI
                        try:
                            payload["debug"]["lines"] = {"left_x": float(LINE_LEFT_X), "right_x": float(LINE_RIGHT_X)}
                        except Exception:
                            pass
                        # include recent crossings for UI
                        try:
                            if getattr(self, "_recent_crossings", None):
                                payload["debug"]["crossings"] = list(self._recent_crossings)
                        except Exception:
                            pass
                        # include stable labels and act-of-weighing stats
                        try:
                            # По умолчанию отдаём стабильные метки (не прыгают)
                            if 'display_labels' in locals() and display_labels:
                                payload["debug"]["labels"] = display_labels
                            payload["debug"]["act"] = {
                                "seen_total": int(len(self._act_seen_labels)),
                                "peak_concurrent": int(self._act_peak),
                                "duration_sec": float(max(0.0, time.time() - self._act_start_ts))
                            }
                        except Exception:
                            pass
                        try:
                            payload["debug"]["ids"] = list(ids) if ids is not None else []
                        except Exception:
                            payload["debug"]["ids"] = []
                        # всегда отправляем счётчики входов, даже если детекций 0
                        payload["debug"]["flow"] = {"left_in": self.left_in, "right_in": self.right_in, "left_flow": self.left_flow, "right_flow": self.right_flow}
                        await STREAM_MANAGER.broadcast(self.stream_id, payload)
            except Exception as e:
                logger.error(f"Infer loop error on {self.stream_id}: {e}")
            # Адаптивный интервал цикла инференса для отзывчивых счётчиков
            try:
                now2 = time.time()
                recent = False
                try:
                    if getattr(self, "_recent_crossings", None):
                        recent = any((now2 - float(c.get("ts", 0))) < 0.8 for c in self._recent_crossings)
                except Exception:
                    recent = False
                if recent:
                    delay = 0.05
                elif getattr(self, "last_count", 0) > 0:
                    delay = 0.08
                else:
                    delay = 0.2
            except Exception:
                delay = 0.2
            await asyncio.sleep(delay)

    

class RtspStream(VideoStream):
    def __init__(self, stream_id: str, rtsp_url: str):
        super().__init__(stream_id)
        self.rtsp_url = rtsp_url

    async def start(self):
        # Явная реализация (на случай старых базовых классов в памяти)
        if not getattr(self, 'running', False):
            self.running = True
            self._stream_task = asyncio.create_task(self._stream_loop())
            # базовый метод доступен через делегат
            self._infer_task = asyncio.create_task(self._infer_loop())

    async def _stream_loop(self):
        try:
            use_ffmpeg = os.getenv("USE_FFMPEG", "false").lower() == "true"
            ffproc = None
            if use_ffmpeg:
                try:
                    q = os.getenv("FFMPEG_MJPEG_Q", "7")
                    cmd = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-rtsp_transport", "tcp",
                        "-fflags", "nobuffer",
                        "-flags", "low_delay",
                        "-i", self.rtsp_url,
                        "-an", "-c:v", "mjpeg", "-q:v", str(q),
                        "-f", "mjpeg", "-"
                    ]
                    ffproc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                    buf = bytearray()
                except Exception as e:
                    logger.warning(f"FFmpeg start failed, fallback to PyAV: {e}")
                    use_ffmpeg = False

            if not use_ffmpeg:
                av_open_rtsp(self.stream_id, self.rtsp_url)

            while self.running:
                if use_ffmpeg and ffproc and ffproc.stdout:
                    try:
                        chunk = ffproc.stdout.read(4096)
                        if not chunk:
                            await asyncio.sleep(0.01)
                            continue
                        buf.extend(chunk)
                        # extract JPEG frames by SOI(FFD8) / EOI(FFD9)
                        while True:
                            soi = buf.find(b"\xff\xd8")
                            if soi < 0:
                                # drop old data if buffer too large
                                if len(buf) > 2_000_000:
                                    del buf[:len(buf)-1024]
                                break
                            eoi = buf.find(b"\xff\xd9", soi + 2)
                            if eoi < 0:
                                # wait for more data
                                # trim left if useless bytes before SOI
                                if soi > 0:
                                    del buf[:soi]
                                break
                            frame = bytes(buf[soi:eoi+2])
                            del buf[:eoi+2]
                            async with self.lock:
                                self.last_jpeg = frame
                            # publish to in-process broker (fire-and-forget)
                            try:
                                if FRAME_BROKER is not None:
                                    asyncio.create_task(FRAME_BROKER.publish(self.stream_id, int(time.time()*1000), time.time(), frame))
                                    if start_global_worker_for is not None:
                                        start_global_worker_for(self.stream_id)
                            except Exception:
                                pass
                            break
                    except Exception:
                        await asyncio.sleep(0.01)
                else:
                    jpeg = av_read_jpeg(self.stream_id, timeout=1.0)
                    if jpeg:
                        async with self.lock:
                            self.last_jpeg = jpeg
                        try:
                            if FRAME_BROKER is not None:
                                asyncio.create_task(FRAME_BROKER.publish(self.stream_id, int(time.time()*1000), time.time(), jpeg))
                                if start_global_worker_for is not None:
                                    start_global_worker_for(self.stream_id)
                        except Exception:
                            pass
                    await asyncio.sleep(1.0 / TARGET_FPS)
        except Exception as e:
            logger.error(f"RTSP stream {self.stream_id} error: {e}")
        finally:
            try:
                av_close(self.stream_id)
            except Exception:
                pass
            try:
                if ffproc:
                    ffproc.kill()
            except Exception:
                pass
            self.running = False

    async def stop(self):
        # Без обращения к VideoStream.stop для устойчивости к hot-reload
        if getattr(self, 'running', False):
            self.running = False
        t1 = getattr(self, '_stream_task', None)
        t2 = getattr(self, '_infer_task', None)
        if t1:
            try:
                t1.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t1
            except Exception:
                pass
            self._stream_task = None
        if t2:
            try:
                t2.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t2
            except Exception:
                pass
            self._infer_task = None
        # Сохранить акт для дашборда
        try:
            self._finalize_act_to_files()
        except Exception:
            pass

class FileStream(VideoStream):
    def __init__(self, stream_id: str, file_path: str):
        super().__init__(stream_id)
        self.file_path = file_path
        self.duration = 0.0
        self.fps = 0.0
        self._seek_event = asyncio.Event()
        self._seek_time: Optional[float] = None
        self.current_time = 0.0

    async def start(self):
        if not getattr(self, 'running', False):
            self.running = True
            self._stream_task = asyncio.create_task(self._stream_loop())
            self._infer_task = asyncio.create_task(self._infer_loop())

    async def _stream_loop(self):
        try:
            meta = av_open_file(self.stream_id, self.file_path)
            self.duration = meta.get("duration", 0.0)
            self.fps = meta.get("fps", 25.0)
            while self.running:
                if self._seek_event.is_set():
                    await self._perform_seek()
                
                jpeg = av_read_jpeg(self.stream_id, timeout=1.0)
                if jpeg:
                    async with self.lock:
                        self.last_jpeg = jpeg
                    self.current_time += (1.0 / self.fps)
                    try:
                        if FRAME_BROKER is not None:
                            asyncio.create_task(FRAME_BROKER.publish(self.stream_id, int(time.time()*1000), time.time(), jpeg))
                            if start_global_worker_for is not None:
                                start_global_worker_for(self.stream_id)
                    except Exception:
                        pass
                await asyncio.sleep(1.0 / self.fps)
        except Exception as e:
            logger.error(f"File stream {self.stream_id} error: {e}")
        finally:
            ocv_close(self.stream_id)
            self.running = False

    async def seek(self, t: float):
        self._seek_time = t
        self._seek_event.set()

    async def _perform_seek(self):
        self._seek_event.clear()
        if self._seek_time is not None:
            jpeg = av_seek_read_jpeg(self.stream_id, self._seek_time)
            async with self.lock:
                self.last_jpeg = jpeg
            self.current_time = self._seek_time
            self._seek_time = None

    async def stop(self):
        if getattr(self, 'running', False):
            self.running = False
        t1 = getattr(self, '_stream_task', None)
        t2 = getattr(self, '_infer_task', None)
        if t1:
            try:
                t1.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t1
            except Exception:
                pass
            self._stream_task = None
        if t2:
            try:
                t2.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t2
            except Exception:
                pass
            self._infer_task = None
        # Сохранить акт для дашборда
        try:
            self._finalize_act_to_files()
        except Exception:
            pass

import asyncio
import logging
import threading
import time
import numpy as np
from typing import Optional, Dict, Any
from core.demo_generator import DemoVideoGenerator

class DemoStream(VideoStream):
    """Демо поток с генерируемым видео"""
    
    def __init__(self, stream_id: str, source_uri: str):
        super().__init__(stream_id)
        self.source_uri = source_uri
        self.demo_generator = None
        
    async def start(self):
        if self.running:
            return
        
        logger.info(f"🎬 Starting demo stream {self.stream_id}")
        self.running = True
        self._reset_act()
        
        # Создаем демо генератор
        self.demo_generator = DemoVideoGenerator()
        
        # Запускаем задачу генерации кадров
        self._stream_task = asyncio.create_task(self._demo_loop())
        
        # Запускаем инференс
        if start_global_worker_for and FRAME_BROKER:
            self._infer_task = asyncio.create_task(start_global_worker_for(self.stream_id))
    
    async def _demo_loop(self):
        """Основной цикл генерации демо кадров"""
        try:
            while self.running:
                if self.demo_generator:
                    # Генерируем кадр
                    frame = self.demo_generator.generate_frame()
                    
                    # Конвертируем в JPEG
                    jpeg_data = encode_jpeg(frame, JPEG_QUALITY)
                    if jpeg_data:
                        self.last_jpeg = jpeg_data
                        
                        # Отправляем в frame broker
                        if FRAME_BROKER:
                            try:
                                # Используем правильный метод API
                                await FRAME_BROKER.publish(
                                    self.stream_id, 
                                    self.demo_generator.frame_count, 
                                    time.time(), 
                                    jpeg_data
                                )
                            except Exception as e:
                                logger.debug(f"Demo stream {self.stream_id} frame broker error: {e}")
                
                # Ждем до следующего кадра
                await asyncio.sleep(1.0 / self.demo_generator.fps)
                
        except Exception as e:
            logger.error(f"Demo stream {self.stream_id} loop error: {e}")
        finally:
            logger.info(f"Demo stream {self.stream_id} loop ended")
    
    async def get_jpeg(self) -> Optional[bytes]:
        return self.last_jpeg
    
    async def stop(self):
        if not self.running:
            return
        
        logger.info(f"🛑 Stopping demo stream {self.stream_id}")
        self.running = False
        
        # Останавливаем задачи
        if self._stream_task:
            try:
                self._stream_task.cancel()
                await self._stream_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._stream_task = None
        
        if self._infer_task:
            try:
                self._infer_task.cancel()
                await self._infer_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._infer_task = None
        
        # Сохраняем акт для дашборда
        try:
            self._finalize_act_to_files()
        except Exception:
            pass

class StreamManager:
    def __init__(self):
        self.streams: Dict[str, VideoStream] = {}
        self.websockets: Dict[str, List[WebSocket]] = {}

    async def get_or_create_stream(self, stream_id: str, source_uri: str) -> VideoStream:
        # Защита от «застрявших» старых инстансов после hot-reload
        cur = self.streams.get(stream_id)
        if cur is not None:
            needs_replace = False
            try:
                # если у объекта нет необходимого метода/базы — пересоздаём
                if not hasattr(cur, 'start') or not isinstance(cur, VideoStream):
                    needs_replace = True
            except Exception:
                needs_replace = True
            if needs_replace:
                try:
                    await cur.stop()
                except Exception:
                    pass
                self.streams.pop(stream_id, None)

        if stream_id not in self.streams:
            # Отключаем демо-поток, вместо него используем последний активный файл
            # if source_uri.startswith("demo://"):
            #    from core.demo_generator import create_demo_stream
            #    self.streams[stream_id] = DemoStream(stream_id, source_uri)
            elif source_uri.startswith("rtsp://"):
                self.streams[stream_id] = RtspStream(stream_id, source_uri)
            else:
                self.streams[stream_id] = FileStream(stream_id, source_uri)
        return self.streams[stream_id]

    async def stop_stream(self, stream_id: str):
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            try:
                await stream.stop()
            finally:
                try:
                    # ensure act finalization is performed
                    if hasattr(stream, '_finalize_act_to_files'):
                        stream._finalize_act_to_files()
                except Exception:
                    pass
                try:
                    del self.streams[stream_id]
                except Exception:
                    pass

    def register_websocket(self, stream_id: str, ws: WebSocket):
        if stream_id not in self.websockets:
            self.websockets[stream_id] = []
        self.websockets[stream_id].append(ws)

    def unregister_websocket(self, stream_id: str, ws: WebSocket):
        if stream_id in self.websockets:
            self.websockets[stream_id].remove(ws)

    async def broadcast(self, stream_id: str, data: dict):
        if stream_id in self.websockets:
            for ws in self.websockets[stream_id]:
                await ws.send_json(data)

STREAM_MANAGER = StreamManager()
# attach frame broker and start inference workers on-demand
try:
    from core.frame_broker import FRAME_BROKER
    from services.inference_worker import start_global_worker_for
    from core.results_store import RESULTS_STORE
except Exception:
    FRAME_BROKER = None
    start_global_worker_for = None
    RESULTS_STORE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    for stream in list(STREAM_MANAGER.streams.values()):
        await stream.stop()

_DEFAULT_RESPONSE = ORJSONResponse if _HAVE_ORJSON else JSONResponse
app = FastAPI(title="PigWeight API v3.0 (Unified)", lifespan=lifespan, default_response_class=_DEFAULT_RESPONSE)

# Подключаем упрощенные endpoints
try:
    from api.simple_endpoints import setup_endpoints
    setup_endpoints(app)
    logging.info("[OK] Simplified endpoints loaded")
except Exception as e:
    logging.error(f"[ERROR] Failed to load simplified endpoints: {e}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Simple WebRTC signalling (aiortc) integrated into FastAPI ---
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
    import uuid as _uuid
    from av import VideoFrame as _VideoFrame
    _HAVE_AIORTC = True
except Exception:
    _HAVE_AIORTC = False

# peer_id -> RTCPeerConnection
_PEER_CONNECTIONS: Dict[str, RTCPeerConnection] = {}


class BrokerVideoTrack(VideoStreamTrack):
    def __init__(self, stream_id: str, fps: float = 15.0):
        super().__init__()
        self.stream_id = stream_id
        self.fps = float(fps)
        self.frame_duration = 1.0 / max(1.0, self.fps)
        self._last_good_jpeg: bytes | None = None

    async def recv(self):
        # Pull latest frame from FRAME_BROKER or fallback to StreamManager.last_jpeg
        jpeg = None
        source = "none"

        if 'FRAME_BROKER' in globals() and FRAME_BROKER is not None:
            latest = FRAME_BROKER.get_latest(self.stream_id)
            if latest and latest.get('jpeg'):
                jpeg = latest.get('jpeg')
                source = "FRAME_BROKER"
                # Проверяем что jpeg это bytes, а не dict
                if not isinstance(jpeg, bytes):
                    logger.warning(f"BrokerVideoTrack {self.stream_id}: FRAME_BROKER returned non-bytes jpeg: {type(jpeg)}")
                    jpeg = None
            else:
                logger.debug(f"BrokerVideoTrack {self.stream_id}: FRAME_BROKER returned empty or no jpeg")

        if jpeg is None:
            # fallback to StreamManager
            stream = STREAM_MANAGER.streams.get(self.stream_id)
            if stream:
                try:
                    jpeg = await stream.get_jpeg()
                    source = "StreamManager"
                except Exception as e:
                    logger.warning(f"BrokerVideoTrack {self.stream_id}: StreamManager.get_jpeg failed: {e}")

        # Use last good jpeg if current is empty
        if jpeg:
            self._last_good_jpeg = jpeg
        elif self._last_good_jpeg:
            jpeg = self._last_good_jpeg
            source = f"cached:{source}"
        else:
            # return black frame if nothing ever received
            import numpy as _np
            h, w = 480, 640
            black = _np.zeros((h, w, 3), dtype=_np.uint8)
            vf = _VideoFrame.from_ndarray(black[..., ::-1], format='bgr24')
            # timestamp via next_timestamp for proper pacing
            pts, time_base = await self.next_timestamp()
            vf.pts = pts
            vf.time_base = time_base
            return vf

        # Проверяем тип данных - исправляем проблему с оптимизированной предобработкой
        if isinstance(jpeg, dict):
            # Если пришел dict вместо bytes, извлекаем jpeg данные
            if 'jpeg' in jpeg:
                jpeg_data = jpeg['jpeg']
            else:
                logger.error(f"BrokerVideoTrack {self.stream_id}: Invalid data format: {type(jpeg)}")
                return None
        else:
            jpeg_data = jpeg
            
        try:
            logger.debug(f"BrokerVideoTrack {self.stream_id}: Using frame from {source}, size: {len(jpeg_data)} bytes")
        except Exception:
            pass

        # decode JPEG to ndarray
        import numpy as _np
        arr = _np.frombuffer(jpeg_data, dtype=_np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            h, w = 480, 640
            img = _np.zeros((h, w, 3), dtype=_np.uint8)

        # Optional: draw segmentation masks on server side (disabled by default, set WEBRTC_OVERLAY_SERVER=true to enable)
        try:
            if os.getenv('WEBRTC_OVERLAY_SERVER', 'false').lower() != 'true':
                raise RuntimeError('server overlay disabled')
            stream = STREAM_MANAGER.streams.get(self.stream_id)
            if stream and getattr(stream, 'last_masks', None):
                masks = stream.last_masks  # уже нормализованы к полному кадру (0..1)
                overlay = img.copy()
                alpha = 0.35
                fill = (0, 200, 255)
                edge = (0, 120, 200)
                Hc, Wc = img.shape[0], img.shape[1]
                for poly in masks:
                    try:
                        pts = []
                        for (nx, ny) in poly:
                            nx2 = 0.0 if nx is None else max(0.0, min(1.0, float(nx)))
                            ny2 = 0.0 if ny is None else max(0.0, min(1.0, float(ny)))
                            sx = int(round(nx2 * Wc))
                            sy = int(round(ny2 * Hc))
                            pts.append((sx, sy))
                        pts = _np.array(pts, dtype=_np.int32)
                        if pts.ndim == 2 and pts.shape[0] >= 3:
                            cv2.fillPoly(overlay, [pts], fill)
                            cv2.polylines(overlay, [pts], isClosed=True, color=edge, thickness=2)
                    except Exception:
                        continue
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        except Exception:
            pass

        # convert BGR -> RGB
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            img_rgb = img

        vf = _VideoFrame.from_ndarray(img_rgb, format='rgb24')
        # Proper timestamping for aiortc pacing
        pts, time_base = await self.next_timestamp()
        vf.pts = pts
        vf.time_base = time_base
        return vf


@app.post('/api/webrtc/offer')
async def api_webrtc_offer(payload: Dict[str, Any]):
    start_time = time.time()
    if not _HAVE_AIORTC:
        perf_logger.warning(".3f")
        return JSONResponse({'error': 'aiortc not available'}, status_code=500)
    try:
        sdp = payload.get('sdp')
        typ = payload.get('type')
        stream_id = payload.get('stream_id')
        fps = float(payload.get('fps') or TARGET_FPS)
        # Validate inputs with clear messages to avoid vague exceptions
        if not isinstance(sdp, str) or len(sdp) < 10:
            return JSONResponse({'error': 'invalid or missing sdp'}, status_code=400)
        if typ not in ('offer', 'answer'):
            return JSONResponse({'error': "invalid type, expected 'offer' or 'answer'"}, status_code=400)
        if not stream_id:
            return JSONResponse({'error': 'missing stream_id'}, status_code=400)

        pc = RTCPeerConnection()
        peer_id = _uuid.uuid4().hex
        _PEER_CONNECTIONS[peer_id] = pc
        logger.info(f"WebRTC: Created peer connection {peer_id} for stream {stream_id}")

        @pc.on('connectionstatechange')
        async def on_state():
            if pc.connectionState == 'failed' or pc.connectionState == 'closed':
                try:
                    await pc.close()
                except Exception:
                    pass
                _PEER_CONNECTIONS.pop(peer_id, None)

        try:
            offer = RTCSessionDescription(sdp=sdp, type=typ)
            await pc.setRemoteDescription(offer)
        except Exception as e:
            logger.exception(f"Failed to set remote description for peer={peer_id} stream={stream_id}")
            return JSONResponse({'error': f'Failed to set remote description: {str(e)}'}, status_code=400)
        # add broker track AFTER remote description to avoid transceiver direction mismatch
        try:
            track = BrokerVideoTrack(stream_id=stream_id, fps=fps)
            pc.addTrack(track)
            logger.info(f"WebRTC: Added BrokerVideoTrack for peer={peer_id}, stream={stream_id}, fps={fps}")
        except Exception:
            logger.exception(f"Failed to add track for peer={peer_id}")
        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except ValueError as e:
            # handle SDP direction mismatch (aiortc error) gracefully
            logger.exception(f"SDP direction error for peer={peer_id} stream={stream_id}: {e}")
            try:
                await pc.close()
            except Exception:
                pass
            _PEER_CONNECTIONS.pop(peer_id, None)
            return JSONResponse({'error': 'SDP direction mismatch or missing media in offer'}, status_code=400)
        except Exception as e:
            logger.exception(f"Failed to create/set local description for peer={peer_id} stream={stream_id}")
            try:
                await pc.close()
            except Exception:
                pass
            _PEER_CONNECTIONS.pop(peer_id, None)
            return JSONResponse({'error': str(e)}, status_code=500)

        end_time = time.time()
        perf_logger.info(".3f")
        return {
            'peer_id': peer_id,
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }
    except Exception as e:
        end_time = time.time()
        perf_logger.error(".3f")
        logger.exception(f"Error handling WebRTC offer for stream_id={payload.get('stream_id')}")
        return JSONResponse({'error': str(e)}, status_code=500)


@app.post('/api/webrtc/candidate')
async def api_webrtc_candidate(body: Dict[str, Any]):
    start_time = time.time()
    try:
        peer_id = body.get('peer_id')
        candidate = body.get('candidate')
        if not peer_id or not candidate:
            perf_logger.warning(".3f")
            return JSONResponse({'error': 'peer_id and candidate required'}, status_code=400)
        pc = _PEER_CONNECTIONS.get(peer_id)
        if not pc:
            perf_logger.warning(".3f")
            return JSONResponse({'error': 'peer not found'}, status_code=404)
        # create RTCIceCandidate
        c = RTCIceCandidate(**candidate)
        await pc.addIceCandidate(c)
        end_time = time.time()
        perf_logger.debug(".3f")
        return {'status': 'ok'}
    except Exception as e:
        end_time = time.time()
        perf_logger.error(".3f")
        logger.exception(f"Error handling WebRTC candidate for peer_id={body.get('peer_id')}")
        return JSONResponse({'error': str(e)}, status_code=500)


@app.post('/api/webrtc/stop')
async def api_webrtc_stop(body: Dict[str, Any]):
    try:
        peer_id = body.get('peer_id')
        pc = _PEER_CONNECTIONS.pop(peer_id, None)
        if pc:
            try:
                await pc.close()
            except Exception:
                pass
        return {'status': 'stopped'}
    except Exception as e:
        logger.exception(f"Error stopping WebRTC peer peer_id={body.get('peer_id')}")
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/health")
async def api_health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index.html"
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return JSONResponse({"error": f"Cannot read index.html: {e}"}, status_code=500)

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Always save under the original safe filename (overwrite if exists)
        safe_name = "".join(c for c in (file.filename or "") if c.isalnum() or c in "._-") or "upload.bin"
        dst = UPLOAD_DIR / safe_name
        content = await file.read()
        try:
            # Skip rewrite if file exists with same size to avoid SSD churn
            if not (dst.exists() and dst.stat().st_size == len(content)):
                with open(dst, "wb") as buffer:
                    buffer.write(content)
        except Exception:
            # Fallback to simple write
            with open(dst, "wb") as buffer:
                buffer.write(content)
        meta = ocv_probe_file(str(dst))
        resp = {"file_path": str(dst)}
        if meta and not meta.get("error"):
            resp.update({
                "duration": float(meta.get("duration", 0.0) or 0.0),
                "fps": float(meta.get("fps", 0.0) or 0.0),
                "frame_count": int(meta.get("frame_count", 0) or 0)
            })
        return resp
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/stream/start")
async def api_stream_start(stream_id: str, source_uri: str):
    start_time = time.time()
    try:
        perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting stream {stream_id} with source {source_uri}")

        # Basic validation for file paths to give clearer errors
        if not source_uri.startswith("rtsp://") and not source_uri.startswith("demo://"):
            p = Path(source_uri)
            if not p.exists():
                perf_logger.warning(".3f")
                return JSONResponse({"error": f"file not found: {source_uri}"}, status_code=404)
            if not p.is_file():
                perf_logger.warning(".3f")
                return JSONResponse({"error": f"not a file: {source_uri}"}, status_code=400)

        stream = await STREAM_MANAGER.get_or_create_stream(stream_id, source_uri)
        await stream.start()
        resp = {"status": "started", "stream_id": stream_id}
        
        if isinstance(stream, FileStream):
            # provide best-known meta immediately
            resp.update({
                "type": "file",
                "duration": float(stream.duration or 0.0),
            })
        
        perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Stream started in {(time.time() - start_time):.3f}s")
        return resp
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/stream/{stream_id}/stop")
async def api_stream_stop(stream_id: str):
    start_time = time.time()
    try:
        perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Stopping stream {stream_id}")
        await STREAM_MANAGER.stop_stream(stream_id)
        end_time = time.time()
        perf_logger.info(".3f")
        return {"status": "stopped", "stream_id": stream_id}
    except Exception as e:
        end_time = time.time()
        perf_logger.warning(".3f")
        # Be resilient: never break UI on stop
        logger.error(f"Stop stream error for {stream_id}: {e}")
        return JSONResponse({"status": "stopped", "stream_id": stream_id, "warning": str(e)}, status_code=200)

@app.get("/api/stream/{stream_id}/snapshot")
async def api_stream_snapshot(stream_id: str):
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream or not stream.running:
        return JSONResponse({"error": "stream not found or not running"}, status_code=404)
    
    jpeg = await stream.get_jpeg()
    if not jpeg:
        return JSONResponse({"error": "no frame"}, status_code=404)
    
    return Response(content=jpeg, media_type="image/jpeg")

async def mjpeg_generator(stream: VideoStream):
    while stream.running:
        jpeg = await stream.get_jpeg()
        if jpeg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        await asyncio.sleep(1.0 / TARGET_FPS)

@app.get("/api/stream/{stream_id}/feed")
async def api_stream_feed(stream_id: str):
    start_time = time.time()
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream or not stream.running:
        perf_logger.warning(".3f")
        return JSONResponse({"error": "stream not found or not running"}, status_code=404)

    perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting MJPEG feed for {stream_id}")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(mjpeg_generator(stream), media_type="multipart/x-mixed-replace; boundary=frame", headers=headers)

@app.get("/api/cameras")
async def api_cameras():
    """Return available cameras as defined in .env (env vars)."""
    return cameras_from_env()

@app.get("/api/stream/{stream_id}/info")
async def api_stream_info(stream_id: str):
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream:
        return JSONResponse({"error": "stream not found"}, status_code=404)
    if isinstance(stream, FileStream):
        # Refresh meta lazily if duration is unknown
        if not stream.duration or stream.duration <= 0.0:
            try:
                meta = av_meta(stream.stream_id)
                stream.duration = float(meta.get("duration", stream.duration or 0.0) or 0.0)
                stream.fps = float(meta.get("fps", stream.fps or 0.0) or 0.0)
            except Exception:
                pass
        return {
            "type": "file",
            "duration": float(stream.duration or 0.0),
            "current_time": float(stream.current_time or 0.0),
            "fps": float(stream.fps or 0.0),
        }
    else:
        return {"type": "rtsp", "duration": None}

# --- Records API for dashboard ---
@app.get("/api/records")
async def api_records_list():
    try:
        items = []
        for p in sorted(RECORDS_DIR.glob("act_*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    js = json.load(f)
                items.append({
                    "name": p.name,
                    "stream_id": js.get("stream_id"),
                    "duration_sec": js.get("duration_sec"),
                    "seen_total": js.get("seen_total"),
                    "peak_concurrent": js.get("peak_concurrent"),
                    "started_at": js.get("started_at"),
                    "finished_at": js.get("finished_at")
                })
            except Exception:
                continue
        items.sort(key=lambda x: x.get("finished_at") or 0, reverse=True)
        return {"items": items}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- Verification API ---
try:
    from core.verification import verification_system
    _HAVE_VERIFICATION = True
except Exception:
    _HAVE_VERIFICATION = False
    verification_system = None

# Глобальный экземпляр процессора
_unified_processor = None

def get_unified_processor():
    """Получение глобального экземпляра процессора"""
    global _unified_processor
    if _unified_processor is None and HAVE_UNIFIED_PROCESSOR:
        # Получаем путь к модели из конфигурации
        model_path = os.getenv("MODEL_PATH", "models/pig_yolo11-seg.v4.pt")
        options = ProcessingOptions(
            confidence_threshold=float(os.getenv("CONF_THRESHOLD", "0.3")),
            img_size=int(os.getenv("IMG_SIZE", "640")),
            device=os.getenv("DEVICE", "auto"),
            batch_size=int(os.getenv("BATCH_SIZE", "1"))
        )
        _unified_processor = get_processor(model_path, options)
        logging.info("🚀 Unified processor initialized")
    return _unified_processor

@app.get("/api/verification/stats")
async def api_verification_stats():
    """Получить статистику верификации всех актов"""
    if not _HAVE_VERIFICATION or verification_system is None:
        return JSONResponse({"error": "Система верификации недоступна"}, status_code=500)

    try:
        stats = verification_system.get_verification_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting verification stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/verification/grouped")
async def api_verification_grouped():
    """Получить группированные по датам данные верификации"""
    if not _HAVE_VERIFICATION or verification_system is None:
        return JSONResponse({"error": "Система верификации недоступна"}, status_code=500)

    try:
        stats = verification_system.get_verification_stats()
        return {
            "grouped_by_date": stats.get("grouped_by_date", {}),
            "summary": {
                "total_dates": len(stats.get("grouped_by_date", {})),
                "total_acts": stats.get("total_acts", 0),
                "verified_acts": stats.get("verified_count", 0),
                "discrepancy_acts": stats.get("discrepancy_count", 0)
            }
        }
    except Exception as e:
        logger.error(f"Error getting grouped verification data: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/verification/verify/{act_name}")
async def api_verify_act(act_name: str):
    """Проверить конкретный акт взвешивания"""
    if not _HAVE_VERIFICATION or verification_system is None:
        return JSONResponse({"error": "Система верификации недоступна"}, status_code=500)

    try:
        result = verification_system.verify_weighing_act(act_name)
        return result
    except Exception as e:
        logger.error(f"Error verifying act {act_name}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/verification/analyze_excel")
async def api_analyze_excel(file: UploadFile = File(...)):
    """Анализировать загруженный Excel файл с замерами"""
    if not _HAVE_VERIFICATION or verification_system is None:
        return JSONResponse({"error": "Система верификации недоступна"}, status_code=500)

    try:
        # Сохраняем файл во временную директорию
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_path = temp_dir / file.filename
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        # Анализируем файл
        result = verification_system.analyze_excel_measurements(str(file_path))

        # Удаляем временный файл
        try:
            file_path.unlink()
        except Exception:
            pass

        return result

    except Exception as e:
        logger.error(f"Error analyzing Excel file: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/verification/report")
async def api_verification_report():
    """Получить отчет о несоответствиях в счетчиках"""
    if not _HAVE_VERIFICATION or verification_system is None:
        return JSONResponse({"error": "Система верификации недоступна"}, status_code=500)

    try:
        stats = verification_system.get_verification_stats()

        # Генерируем отчет
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_acts": stats["total_acts"],
                "verified_acts": stats["verified_count"],
                "discrepancy_acts": stats["discrepancy_count"],
                "error_acts": stats["error_count"],
                "total_pigs_counted": stats["total_pigs"],
                "verification_rate": (stats["verified_count"] / max(stats["total_acts"], 1)) * 100,
                "avg_duration_sec": stats["avg_duration"]
            },
            "issues": [],
            "recommendations": []
        }

        # Анализируем проблемы
        for act in stats["results"]:
            if act["status"] == "success":
                verification = act.get("verification", {})
                if not verification.get("verified"):
                    issue = {
                        "act_file": act["act_file"],
                        "stream_id": act.get("stream_id"),
                        "timestamp": act.get("finished_at"),
                        "left_count": act.get("flow", {}).get("left_in", 0),
                        "right_count": act.get("flow", {}).get("right_in", 0),
                        "difference": verification.get("difference", 0),
                        "relative_difference": verification.get("relative_difference", 0),
                        "status": verification.get("status"),
                        "message": verification.get("message", "")
                    }
                    report["issues"].append(issue)

        # Генерируем рекомендации
        if report["summary"]["discrepancy_acts"] > 0:
            discrepancy_rate = (report["summary"]["discrepancy_acts"] / report["summary"]["total_acts"]) * 100
            if discrepancy_rate > 20:
                report["recommendations"].append({
                    "priority": "high",
                    "message": f"{discrepancy_rate:.1f}% актов имеют расхождения",
                    "action": "Проверьте настройки камер и линий отсечки"
                })
            elif discrepancy_rate > 10:
                report["recommendations"].append({
                    "priority": "medium",
                    "message": f"{discrepancy_rate:.1f}% актов имеют расхождения",
                    "action": "Рекомендуется калибровка системы распознавания"
                })

        # Анализ паттернов ошибок
        if len(report["issues"]) > 0:
            # Группировка по времени
            time_pattern = {}
            for issue in report["issues"]:
                hour = datetime.fromtimestamp(issue["timestamp"]).hour
                time_pattern[hour] = time_pattern.get(hour, 0) + 1

            peak_hour = max(time_pattern.items(), key=lambda x: x[1])[0] if time_pattern else None
            if peak_hour is not None:
                report["recommendations"].append({
                    "priority": "info",
                    "message": f"Большинство расхождений происходит в {peak_hour}:00",
                    "action": "Проверьте условия освещения в это время"
                })

        return report

    except Exception as e:
        logger.error(f"Error generating verification report: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/records/{name}")
async def api_records_get(name: str):
    try:
        safe = "".join(c for c in name if c.isalnum() or c in "._-" )
        if not safe:
            return JSONResponse({"error": "invalid name"}, status_code=400)
        path = RECORDS_DIR / safe
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        # прикладываем ссылку на svg, если есть
        svg_name = safe.replace(".json", ".svg")
        if (RECORDS_DIR / svg_name).exists():
            js["svg"] = f"/records/{svg_name}"
        return js
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- Runtime adjustable cut lines ---
@app.get("/api/lines")
async def api_get_lines():
    try:
        return {"left_x": float(LINE_LEFT_X), "right_x": float(LINE_RIGHT_X)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/lines")
async def api_set_lines(left_x: float = Query(None), right_x: float = Query(None), body: dict = Body(None)):
    try:
        # Поддерживаем как query, так и JSON body: {left_x, right_x}
        lx = left_x
        rx = right_x
        if body and isinstance(body, dict):
            if lx is None and body.get("left_x") is not None:
                lx = float(body.get("left_x"))
            if rx is None and body.get("right_x") is not None:
                rx = float(body.get("right_x"))
        if lx is None or rx is None:
            return JSONResponse({"error": "left_x and right_x are required"}, status_code=400)
        # Нормализуем и гарантируем разнос и порядок
        try:
            lx = _clamp01(float(lx))
            rx = _clamp01(float(rx))
        except Exception:
            return JSONResponse({"error": "invalid values"}, status_code=400)
        if lx > rx:
            lx, rx = rx, lx
        # Минимальный зазор 0.05
        min_gap = 0.05
        if (rx - lx) < min_gap:
            mid = 0.5 * (lx + rx)
            lx = max(0.0, mid - min_gap / 2)
            rx = min(1.0, mid + min_gap / 2)
        # Применяем глобально (для всех потоков)
        global LINE_LEFT_X, LINE_RIGHT_X
        LINE_LEFT_X, LINE_RIGHT_X = float(lx), float(rx)
        return {"left_x": LINE_LEFT_X, "right_x": LINE_RIGHT_X}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/stream/{stream_id}/seek")
async def api_stream_seek(stream_id: str, t: float = Query(...)):
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream:
        return JSONResponse({"error": "stream not found"}, status_code=404)
    if not isinstance(stream, FileStream):
        return JSONResponse({"error": "seek supported only for file streams"}, status_code=400)
    await stream.seek(max(0.0, float(t)))
    return {"status": "ok", "current_time": float(stream.current_time)}

@app.post("/api/stream/{stream_id}/line_positions")
async def api_set_line_positions(stream_id: str, positions: Dict[str, Any] = Body(...)):
    """Сохранить позиции линий для конкретного видеофайла."""
    try:
        stream = await STREAM_MANAGER.get_stream(stream_id)
        if not stream:
            return JSONResponse({"error": f"Stream {stream_id} not found"}, status_code=404)
        
        # Сохраняем позиции линий
        file_path = stream_id
        if "://" in stream_id:
            # Извлекаем имя файла из URL
            file_path = stream_id.split("/")[-1]
        
        stream.line_positions[file_path] = positions
        
        return {"status": "success", "message": f"Line positions saved for {file_path}"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/stream/{stream_id}/optimize")
async def api_stream_optimize(stream_id: str, transport: str = Query("mjpeg")):
    """Optimize stream settings based on transport type"""
    try:
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse({"error": "stream not found"}, status_code=404)

        # For WebRTC, we can optimize by reducing some polling
        if transport == "webrtc":
            # This is a hint to potentially adjust internal settings
            # Could be extended to modify batch sizes, polling rates, etc.
            logger.info(f"Optimizing stream {stream_id} for WebRTC transport")
            return {"status": "optimized", "transport": "webrtc"}

        return {"status": "no_change", "transport": transport}
    except Exception as e:
        logger.error(f"Error optimizing stream {stream_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



# === API для работы с актами взвешивания ===

# Путь к файлу базы данных актов взвешивания
WEIGHING_DB_FILE = "weighing_acts.json"

def load_weighing_acts():
    """Загрузка актов взвешивания из JSON файла"""
    try:
        if os.path.exists(WEIGHING_DB_FILE):
            with open(WEIGHING_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading weighing acts: {e}")
        return []

def save_weighing_acts(acts):
    """Сохранение актов взвешивания в JSON файл"""
    try:
        with open(WEIGHING_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(acts, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving weighing acts: {e}")
        return False

@app.post("/api/weighing/save")
async def save_weighing_act(data: Dict[str, Any]):
    """Сохранение акта взвешивания"""
    try:
        # Валидация данных
        required_fields = ['date', 'group', 'total', 'weight']
        for field in required_fields:
            if field not in data or not data[field]:
                return JSONResponse({"error": f"Missing required field: {field}"}, status_code=400)
        
        # Создаем запись акта
        act = {
            'id': f"{data['date']}_{data['group']}_{int(time.time())}",
            'date': data['date'],
            'time': data.get('time', datetime.now().strftime('%H:%M')),
            'group': data['group'],
            'total': int(data['total']),
            'weight': float(data['weight']),
            'avg_weight': float(data.get('avg_weight', data['weight'] / data['total'])),
            'left_count': int(data.get('left_count', 0)),
            'right_count': int(data.get('right_count', 0)),
            'stream_id': data.get('stream_id', ''),
            'created_at': datetime.now().isoformat()
        }
        
        # Загружаем существующие акты
        acts = load_weighing_acts()
        acts.append(act)
        
        # Сохраняем
        if save_weighing_acts(acts):
            logger.info(f"Saved weighing act: {act['group']} on {act['date']}")
            return {"status": "success", "id": act['id']}
        else:
            return JSONResponse({"error": "Failed to save act"}, status_code=500)
            
    except Exception as e:
        logger.error(f"Error saving weighing act: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/weighing/logs")
async def get_weighing_logs(date_from: str = Query(...), date_to: str = Query(...)):
    """Получение журнала актов взвешивания за период"""
    try:
        acts = load_weighing_acts()
        
        # Фильтруем по датам
        filtered_acts = []
        for act in acts:
            if date_from <= act['date'] <= date_to:
                filtered_acts.append(act)
        
        # Сортируем по дате и времени (новые сначала)
        filtered_acts.sort(key=lambda x: (x['date'], x['time']), reverse=True)
        
        return filtered_acts
        
    except Exception as e:
        logger.error(f"Error getting weighing logs: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/weighing/export")
async def export_weighing_logs(date_from: str = Query(...), date_to: str = Query(...)):
    """Экспорт журнала актов взвешивания в Excel"""
    try:
        acts = load_weighing_acts()
        
        # Фильтруем по датам
        filtered_acts = []
        for act in acts:
            if date_from <= act['date'] <= date_to:
                filtered_acts.append(act)
        
        if not filtered_acts:
            return JSONResponse({"error": "No data to export"}, status_code=404)
        
        # Создаем CSV данные (проще, чем Excel)
        csv_data = []
        csv_data.append(['Дата', 'Время', 'Группа/Секция', 'Количество свиней', 'Общий вес (кг)', 'Средний вес (кг)', 'Счетчик слева', 'Счетчик справа', 'Разница счетчиков', 'Поток'])
        
        for act in sorted(filtered_acts, key=lambda x: (x['date'], x['time'])):
            diff = abs(act['left_count'] - act['right_count'])
            csv_data.append([
                act['date'],
                act['time'],
                act['group'],
                act['total'],
                act['weight'],
                act['avg_weight'],
                act['left_count'],
                act['right_count'],
                diff,
                act['stream_id']
            ])
        
        # Создаем CSV файл в памяти
        import io
        output = io.StringIO()
        writer = csv.writer(output, delimiter=';')  # Используем точку с запятой для лучшей совместимости с Excel
        for row in csv_data:
            writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()
        
        # Возвращаем как файл
        from fastapi.responses import Response
        return Response(
            content=csv_content.encode('utf-8-sig'),  # BOM для правильного отображения в Excel
            media_type='text/csv; charset=utf-8',
            headers={
                "Content-Disposition": f"attachment; filename=weighing_logs_{date_from}_{date_to}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting weighing logs: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === API для журнала взвешиваний ===

@app.get("/api/journal/acts")
async def get_weighing_acts(
    date_from: str = None,
    date_to: str = None,
    camera: str = None,
    limit: int = 100
):
    """Получить акты взвешивания с фильтрацией"""
    try:
        acts = load_weighing_acts()
        
        # Применяем фильтры
        filtered_acts = []
        for act in acts:
            # Фильтр по дате
            if date_from and act['date'] < date_from:
                continue
            if date_to and act['date'] > date_to:
                continue
            # Фильтр по камере
            if camera and act.get('camera') != camera:
                continue
            
            filtered_acts.append(act)
        
        # Сортируем по дате (новые сверху)
        filtered_acts.sort(key=lambda x: (x['date'], x['time']), reverse=True)
        
        # Ограничиваем количество
        filtered_acts = filtered_acts[:limit]
        
        # Добавляем статистику
        total_count = sum(act['count'] for act in filtered_acts)
        total_weight = sum(act['weight'] for act in filtered_acts)
        avg_weight = total_weight / total_count if total_count > 0 else 0
        
        return {
            "acts": filtered_acts,
            "summary": {
                "total_acts": len(filtered_acts),
                "total_count": total_count,
                "total_weight": round(total_weight, 1),
                "avg_weight": round(avg_weight, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting weighing acts: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/journal/save")
async def save_weighing_act(request: Request):
    """Сохранить новый акт взвешивания"""
    try:
        data = await request.json()
        
        # Валидация данных
        count = data.get('count', 0)
        weight = data.get('weight', 0)
        camera = data.get('camera', 'cam1')
        
        if count <= 0:
            return JSONResponse({"error": "Количество должно быть больше 0"}, status_code=400)
        if weight <= 0:
            return JSONResponse({"error": "Вес должен быть больше 0"}, status_code=400)
        
        # Создаем новый акт
        from datetime import datetime
        now = datetime.now()
        
        new_act = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "count": int(count),
            "weight": float(weight),
            "camera": camera
        }
        
        # Загружаем существующие акты
        acts = load_weighing_acts()
        
        # Добавляем новый акт в начало
        acts.insert(0, new_act)
        
        # Сохраняем обратно в файл
        save_weighing_acts(acts)
        
        logger.info(f"Сохранен новый акт взвешивания: {count} шт., {weight} кг")
        
        return {"success": True, "act": new_act}
        
    except Exception as e:
        logger.error(f"Error saving weighing act: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === API для сверки с файлами замеров ===

@app.post("/api/journal/compare")
async def compare_with_excel(file: UploadFile = File(...)):
    """Простая сверка актов взвешивания с Excel файлом"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            return JSONResponse({"error": "Поддерживаются только Excel файлы (.xlsx, .xls)"}, status_code=400)
        
        # Сохраняем временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            import pandas as pd
            # Читаем Excel файл
            excel_df = pd.read_excel(tmp_path)
            
            # Загружаем наши акты
            acts = load_weighing_acts()
            
            # Ищем столбцы с количеством и весом в Excel
            count_cols = [col for col in excel_df.columns if any(word in col.lower() for word in ['количество', 'count', 'шт', 'голов'])]
            weight_cols = [col for col in excel_df.columns if any(word in col.lower() for word in ['вес', 'weight', 'кг'])]
            
            if not count_cols or not weight_cols:
                return JSONResponse({"error": "Не найдены столбцы с количеством или весом в Excel файле"}, status_code=400)
            
            # Суммируем данные из Excel
            excel_total_count = int(excel_df[count_cols[0]].sum()) if count_cols else 0
            excel_total_weight = float(excel_df[weight_cols[0]].sum()) if weight_cols else 0
            
            # Суммируем данные из наших актов
            acts_total_count = sum(act['count'] for act in acts)
            acts_total_weight = sum(act['weight'] for act in acts)
            
            # Вычисляем расхождения
            count_diff = abs(acts_total_count - excel_total_count)
            weight_diff = abs(acts_total_weight - excel_total_weight)
            
            count_diff_percent = (count_diff / max(acts_total_count, excel_total_count)) * 100 if max(acts_total_count, excel_total_count) > 0 else 0
            weight_diff_percent = (weight_diff / max(acts_total_weight, excel_total_weight)) * 100 if max(acts_total_weight, excel_total_weight) > 0 else 0
            
            # Определяем статус сверки
            matches = 0
            differences = 0
            
            if count_diff_percent <= 5 and weight_diff_percent <= 5:  # Допуск 5%
                matches = 1
                status = "success"
                message = "Данные соответствуют"
            else:
                differences = 1
                status = "warning"
                message = f"Расхождения: количество {count_diff_percent:.1f}%, вес {weight_diff_percent:.1f}%"
            
            return {
                "status": status,
                "message": message,
                "matches": matches,
                "differences": differences,
                "comparison": {
                    "excel": {
                        "total_count": excel_total_count,
                        "total_weight": round(excel_total_weight, 1),
                        "rows": len(excel_df)
                    },
                    "acts": {
                        "total_count": acts_total_count,
                        "total_weight": round(acts_total_weight, 1),
                        "rows": len(acts)
                    },
                    "differences": {
                        "count_diff": count_diff,
                        "weight_diff": round(weight_diff, 1),
                        "count_diff_percent": round(count_diff_percent, 1),
                        "weight_diff_percent": round(weight_diff_percent, 1)
                    }
                }
            }
            
        finally:
            # Удаляем временный файл
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Error comparing with Excel: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/verification/compare")
async def compare_with_measurements(file: UploadFile = File(...)):
    """Сверка актов взвешивания с загруженным файлом замеров"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            return JSONResponse({"error": "Unsupported file format. Use .xlsx, .xls or .csv"}, status_code=400)
        
        # Сохраняем временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Читаем файл замеров
            if file.filename.endswith('.csv'):
                import pandas as pd
                measurements_df = pd.read_csv(tmp_path)
            else:
                import pandas as pd
                measurements_df = pd.read_excel(tmp_path, engine='openpyxl')
            
            # Загружаем наши акты
            acts = load_weighing_acts()
            
            # Анализируем структуру файла замеров (используя наш анализ)
            # Предполагаем, что в файле есть столбцы с датами, группами, весами и количеством
            
            matches = 0
            differences = 0
            missing = 0
            details = []
            
            # Простая логика сверки (может быть улучшена)
            for act in acts:
                # Ищем соответствующую запись в файле замеров
                found_match = False
                for _, row in measurements_df.iterrows():
                    # Пытаемся найти совпадение по дате и группе
                    # Это упрощенная логика, может потребовать доработки
                    if str(act['group']).lower() in str(row.values).lower():
                        found_match = True
                        # Сравниваем веса (с допуском 5%)
                        file_weight = 0
                        file_count = 0
                        
                        # Пытаемся извлечь числовые значения из строки
                        for val in row.values:
                            if isinstance(val, (int, float)) and 10 <= val <= 2000:  # Диапазон весов
                                if abs(val - act['weight']) < abs(file_weight - act['weight']):
                                    file_weight = val
                            elif isinstance(val, (int, float)) and 1 <= val <= 500:  # Диапазон количества
                                if abs(val - act['total']) < abs(file_count - act['total']):
                                    file_count = val
                        
                        weight_diff = abs(act['weight'] - file_weight) / act['weight'] if act['weight'] > 0 else 1
                        count_diff = abs(act['total'] - file_count) / act['total'] if act['total'] > 0 else 1
                        
                        if weight_diff <= 0.05 and count_diff <= 0.1:  # Допуск 5% по весу, 10% по количеству
                            matches += 1
                            status = 'match'
                        else:
                            differences += 1
                            status = 'diff'
                        
                        details.append({
                            'group': act['group'],
                            'date': act['date'],
                            'status': status,
                            'system_weight': act['weight'],
                            'system_count': act['total'],
                            'file_weight': file_weight,
                            'file_count': file_count
                        })
                        break
                
                if not found_match:
                    missing += 1
                    details.append({
                        'group': act['group'],
                        'date': act['date'],
                        'status': 'missing',
                        'system_weight': act['weight'],
                        'system_count': act['total'],
                        'file_weight': 0,
                        'file_count': 0
                    })
            
            return {
                'matches': matches,
                'differences': differences,
                'missing': missing,
                'total_acts': len(acts),
                'details': details[:20]  # Ограничиваем количество деталей
            }
            
        finally:
            # Удаляем временный файл
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error comparing with measurements: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws/count")
async def ws_count(ws: WebSocket, id: str):
    start_time = time.time()
    perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket connection established for stream {id}")

    await ws.accept()
    STREAM_MANAGER.register_websocket(id, ws)
    messages_sent = 0

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        end_time = time.time()
        perf_logger.info(".3f")
        STREAM_MANAGER.unregister_websocket(id, ws)
    except Exception as e:
        end_time = time.time()
        perf_logger.error(".3f")
        logger.error(f"WebSocket error for stream {id}: {e}")
        STREAM_MANAGER.unregister_websocket(id, ws)
