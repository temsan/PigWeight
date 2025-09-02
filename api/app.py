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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

import numpy as np
from fastapi import Body
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

# Изолированный воркер для OpenCV (устойчивый импорт как пакетом, так и отдельным скриптом)
import sys as _sys
_CUR_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CUR_DIR.parent
if str(_ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_DIR))
try:
    from api.opencv_worker import OpenCVIsolate  # запуск как пакет: python -m api.app
except Exception:
    try:
        from .opencv_worker import OpenCVIsolate  # относительный импорт внутри пакета
    except Exception:
        from opencv_worker import OpenCVIsolate   # запуск как файл: python api/app.py

from logging.handlers import RotatingFileHandler

# --- Bootstrap ---
BASE_DIR = Path(__file__).parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

# --- Logging Configuration ---
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "api.log"

# Configure root logger for file logging and rotation
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Rotating file handler
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5) # 10MB per file, 5 backups
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Get root logger and add handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

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
# Новая версия весов по умолчанию (v3). Если её нет, далее будет fallback на best.pt
PIG_MODEL_PATH = os.getenv("PIG_MODEL_PATH", "models/pig_yolo11-seg.v3.pt")
PIG_CLASS_ID = int(os.getenv("PIG_CLASS_ID", "0"))

# Выбор эффективной модели и классов
if DETECTION_MODE == "pig-only":
    # Безопасный фоллбек: если указанный файл отсутствует, используем models/best.pt
    _p = Path(PIG_MODEL_PATH)
    if not _p.exists():
        MODEL_PATH = str((BASE_DIR / "models" / "best.pt").as_posix()).replace("\\", "/")
        logger.warning(f"PIG_MODEL_PATH not found: {PIG_MODEL_PATH}. Falling back to: {MODEL_PATH}")
    else:
        MODEL_PATH = PIG_MODEL_PATH
    TARGET_CLASS_IDS = {PIG_CLASS_ID}
else:
    TARGET_CLASS_IDS = set(map(int, os.getenv("TARGET_CLASS_IDS", "20,17,19").split(",")))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.30"))
AVG_WINDOW = int(os.getenv("AVG_WINDOW", "20"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))

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

OCV: Optional[OpenCVIsolate] = None
def get_ocv() -> OpenCVIsolate:
    global OCV
    if OCV is None:
        OCV = OpenCVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=TARGET_FPS)
    return OCV

def _ocv_safe_call(method_name: str, *args, **kwargs):
    try:
        ocv = get_ocv()
        method = getattr(ocv, method_name, None)
        if not method:
            raise AttributeError(f"OpenCVIsolate lacks {method_name}")
        return method(*args, **kwargs)
    except Exception:
        try:
            global OCV
            OCV = OpenCVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=TARGET_FPS)
            method = getattr(OCV, method_name, None)
            if not method:
                raise AttributeError(f"OpenCVIsolate lacks {method_name}")
            return method(*args, **kwargs)
        except Exception as e2:
            raise e2

def ocv_open_rtsp(stream_id: str, url: str) -> Dict[str, Any]:
    return _ocv_safe_call('open_rtsp', stream_id, url, timeout=8.0)

def ocv_open_file(stream_id: str, path: str) -> Dict[str, Any]:
    return _ocv_safe_call('open_file', stream_id, path, timeout=3.0)

def ocv_close(stream_id: str) -> None:
    try:
        _ocv_safe_call('close', stream_id)
    except Exception:
        pass

def ocv_read_jpeg(stream_id: str, timeout: float = 1.0) -> Optional[bytes]:
    return _ocv_safe_call('read_jpeg', stream_id, timeout=timeout)

def ocv_seek_read_jpeg(stream_id: str, t: float, timeout: float = 2.0) -> Optional[bytes]:
    return _ocv_safe_call('seek_read_jpeg', stream_id, t, timeout=timeout)
def ocv_meta(stream_id: str) -> Dict[str, Any]:
    return _ocv_safe_call('meta', stream_id)

def ocv_probe_file(path: str) -> Dict[str, Any]:
    """Open a file temporarily in the worker to fetch meta, then close it."""
    tmp_id = f"probe_{int(time.time()*1000)}"
    try:
        meta = ocv_open_file(tmp_id, path)
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
    return _ocv_safe_call('meta', stream_id)

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
        except Exception as e:
            logger.error(f"Finalize act save error for {self.stream_id}: {e}")

    @staticmethod
    def _detect_black_bars_top_bottom(frame: np.ndarray) -> tuple[int, int]:
        """Detect top/bottom black bars and return crop indices [y0:y1].
        Only crops vertical bars (letterbox). Returns (0, H) if nothing significant.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        H, W = gray.shape[:2]
        if H < 10 or W < 10:
            return 0, H

        # Row brightness and threshold
        row_mean = gray.mean(axis=1)
        # Dynamic threshold: low brightness + margin
        thr = max(8.0, row_mean.mean() * 0.15)
        min_run = 4  # consecutive non-black rows to consider start of content

        # Find y0 from top
        y0 = 0
        run = 0
        for i, val in enumerate(row_mean):
            if val > thr:
                run += 1
                if run >= min_run:
                    y0 = max(0, i - (min_run - 1))
                    break
            else:
                run = 0

        # Find y1 from bottom
        y1 = H
        run = 0
        for off, val in enumerate(reversed(row_mean)):
            if val > thr:
                run += 1
                if run >= min_run:
                    y1 = H - max(0, off - (min_run - 1))
                    break
            else:
                run = 0

        # Sanity: ensure meaningful crop (at least 1% trimmed each side combined)
        min_trim = int(0.01 * H)
        if y0 <= min_trim and (H - y1) <= min_trim:
            return 0, H
        if y1 - y0 < int(0.5 * H):
            # Avoid over-cropping when detection fails
            return 0, H
        return max(0, y0), min(H, y1)

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
            return self.last_jpeg


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
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        H0, W0, _ = frame.shape
                        y0, y1 = self._detect_black_bars_top_bottom(frame)
                        proc = frame[y0:y1, :, :] if (0 <= y0 < y1 <= H0) else frame
                        results = self.model.predict(proc, imgsz=640, conf=CONF_THRESHOLD, verbose=False, retina_masks=True)
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
                            # Дополнительно подготовим «человеческий» порядок: по порядку пересечения ЛЕВОЙ линии,
                            # если объект её ещё не пересёк — по времени входа (первого появления),
                            # при равенстве — правее раньше (справа-налево), чтобы слои масок были читаемы
                            try:
                                n = len(centroids_local)
                                ordered_labels = [0] * n
                                order_idx = list(range(n))
                                # Привязать индекс i к track id
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
                            # Update act-of-weighing metrics
                            cur_count = len(session_labels)
                            if cur_count > self._act_peak:
                                self._act_peak = cur_count
                            for lab in session_labels:
                                self._act_seen_labels.add(int(lab))
                        else:
                            self.last_count = 0
                            self.last_masks = []
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
                            if 'ordered_labels' in locals() and ordered_labels:
                                payload["debug"]["labels"] = ordered_labels
                            payload["debug"]["act"] = {
                                "seen_total": int(len(self._act_seen_labels)),
                                "peak_concurrent": int(self._act_peak),
                                "duration_sec": float(max(0.0, time.time() - self._act_start_ts))
                            }
                        except Exception:
                            pass
                        if ids is not None:
                            payload["debug"]["ids"] = ids
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
            ocv_open_rtsp(self.stream_id, self.rtsp_url)
            while self.running:
                jpeg = ocv_read_jpeg(self.stream_id, timeout=1.0)
                async with self.lock:
                    self.last_jpeg = jpeg
                await asyncio.sleep(1.0 / TARGET_FPS)
        except Exception as e:
            logger.error(f"RTSP stream {self.stream_id} error: {e}")
        finally:
            ocv_close(self.stream_id)
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
            meta = ocv_open_file(self.stream_id, self.file_path)
            self.duration = meta.get("duration", 0.0)
            self.fps = meta.get("fps", 25.0)
            while self.running:
                if self._seek_event.is_set():
                    await self._perform_seek()
                
                jpeg = ocv_read_jpeg(self.stream_id, timeout=1.0)
                if jpeg:
                    async with self.lock:
                        self.last_jpeg = jpeg
                    self.current_time += (1.0 / self.fps)
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
            jpeg = ocv_seek_read_jpeg(self.stream_id, self._seek_time)
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
            if source_uri.startswith("rtsp://"):
                self.streams[stream_id] = RtspStream(stream_id, source_uri)
            else:
                self.streams[stream_id] = FileStream(stream_id, source_uri)
        return self.streams[stream_id]

    async def stop_stream(self, stream_id: str):
        if stream_id in self.streams:
            await self.streams[stream_id].stop()
            del self.streams[stream_id]

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    for stream in list(STREAM_MANAGER.streams.values()):
        await stream.stop()

app = FastAPI(title="PigWeight API (FastAPI)", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/api/health")
async def api_health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")

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
    try:
        # Basic validation for file paths to give clearer errors
        if not source_uri.startswith("rtsp://"):
            p = Path(source_uri)
            if not p.exists():
                return JSONResponse({"error": f"file not found: {source_uri}"}, status_code=404)
            if not p.is_file():
                return JSONResponse({"error": f"not a file: {source_uri}"}, status_code=400)

        stream = await STREAM_MANAGER.get_or_create_stream(stream_id, source_uri)
        await stream.start()
        resp = {"status": "started", "stream_id": stream_id}
        if isinstance(stream, FileStream):
            # provide best-known meta immediately
            resp.update({
                "type": "file",
                "duration": float(stream.duration or 0.0),
                "fps": float(stream.fps or 0.0)
            })
        else:
            resp.update({"type": "rtsp"})
        return resp
    except Exception as e:
        logger.error(f"Start stream error for {stream_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/stream/{stream_id}/stop")
async def api_stream_stop(stream_id: str):
    try:
        await STREAM_MANAGER.stop_stream(stream_id)
        return {"status": "stopped", "stream_id": stream_id}
    except Exception as e:
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
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream or not stream.running:
        return JSONResponse({"error": "stream not found or not running"}, status_code=404)
    return StreamingResponse(mjpeg_generator(stream), media_type="multipart/x-mixed-replace; boundary=frame")

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
                meta = ocv_meta(stream.stream_id)
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



@app.websocket("/ws/count")
async def ws_count(ws: WebSocket, id: str):
    await ws.accept()
    STREAM_MANAGER.register_websocket(id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        STREAM_MANAGER.unregister_websocket(id, ws)
