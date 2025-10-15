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

# Импорт системы событий
try:
    from services.event_logger import get_event_logger
    HAVE_EVENT_LOGGER = True
except ImportError:
    HAVE_EVENT_LOGGER = False

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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form, Body, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from fastapi.responses import ORJSONResponse
except ImportError:
    ORJSONResponse = JSONResponse

try:
    import orjson as _orjson  # noqa: F401
    _HAVE_ORJSON = True
except ImportError:
    _HAVE_ORJSON = False

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

# Импорт упрощенной системы логирования и конфигурации
try:
    from core.config import setup_logging, get_config
    config = get_config()
    logger = setup_logging(debug=config.DEBUG)

except ImportError:
    # Fallback если core.config недоступен
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("api")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    MAX_WAIT_MS = int(os.getenv("MAX_WAIT_MS", "50"))

# --- Config from environment ---
# Model config
DETECTION_MODE = config.DETECTION_MODE
PIG_CLASS_ID = config.PIG_CLASS_ID

if DETECTION_MODE == "pig-only":
    _chosen_path = config.MODEL_PATH or config.PIG_MODEL_PATH
    if not _chosen_path:
        raise RuntimeError("MODEL_PATH или PIG_MODEL_PATH не задан в .env")
    _p = Path(_chosen_path)
    if not _p.exists():
        raise RuntimeError(f"Файл модели не найден: {_chosen_path}")
    MODEL_PATH = _chosen_path
    TARGET_CLASS_IDS = {PIG_CLASS_ID}
else:
    TARGET_CLASS_IDS = set(map(int, config.TARGET_CLASS_IDS.split(",")))
    _ENV_MODEL_PATH = config.MODEL_PATH
    if not _ENV_MODEL_PATH:
        raise RuntimeError("MODEL_PATH не задан в .env для текущего DETECTION_MODE")
    if not Path(_ENV_MODEL_PATH).exists():
        raise RuntimeError(f"Файл модели не найден: {_ENV_MODEL_PATH}")
    MODEL_PATH = _ENV_MODEL_PATH  # Уменьшен кулдаун для более точного подсчета

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
    # ВРЕМЕННО: Отключаем все камеры для тестирования масок
    logger.info("🚫 ВРЕМЕННО: Все камеры отключены для тестирования масок")
    return {}

def encode_jpeg(frame, quality: int = None) -> bytes:
    q = quality or config.JPEG_QUALITY
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
        AVW = AVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=config.TARGET_FPS)
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
            AVW = AVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=config.TARGET_FPS)
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
        self.last_frame_data: Optional[Dict[str, Any]] = None
        self.lock = asyncio.Lock()
        self.model = None
        self.model_loaded = False
        self.last_count = 0
        # Оценка количества: максимум по окну и монотоничный отчёт (не прыгает)
        self.window_max = WindowMaxEstimator(config.AVG_WINDOW)
        self.reported_count = 0
        # flow counters and per-track state
        self.left_in = 0
        self.right_in = 0
        self.total_crossings = 0  # Общий счетчик всех пересечений
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
        
        # Инициализация системы событий
        self.event_logger = get_event_logger() if HAVE_EVENT_LOGGER else None
        if self.event_logger:
            logger.info(f"[{self.stream_id}] Система журналирования событий активна")
        else:
            logger.warning(f"[{self.stream_id}] Система журналирования событий недоступна")
        
        # Session numbering and act-of-weighing metrics
        self._session_id_map: Dict[int, int] = {}
        self._next_session_label: int = 1
        self._act_seen_labels: set[int] = set()
        self._act_peak: int = 0
        self._act_start_ts: float = time.time()
        
        # Очередь для неблокирующего журналирования
        self._logging_queue = asyncio.Queue(maxsize=100)
        self._logging_task: Optional[asyncio.Task] = None
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

    def _get_current_line_positions(self) -> Dict[str, float]:
        """Возвращает текущие позиции линий (локальные или глобальные)"""
        if hasattr(self, 'line_positions') and self.line_positions:
            return {
                "left_x": float(self.line_positions.get('left_x', config.LINE_LEFT_X)),
                "right_x": float(self.line_positions.get('right_x', config.LINE_RIGHT_X))
            }
        else:
            return {
                "left_x": float(config.LINE_LEFT_X),
                "right_x": float(config.LINE_RIGHT_X)
            }
    
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

    

    async def _update_line_counters(self, ids: List[int], centers_x: List[float], centers_y: Optional[List[float]] = None):
        """Count entries from left/right by crossing vertical lines.
        Directional and with cooldown to avoid bouncing.
        """
        async with self.lock:
            now = time.time()
            
            # Используем локальные позиции линий, если они есть, иначе глобальные
            if hasattr(self, 'line_positions') and self.line_positions:
                L = float(self.line_positions.get('left_x', config.LINE_LEFT_X))
                R = float(self.line_positions.get('right_x', config.LINE_RIGHT_X))
                logger.debug(f"[{self.stream_id}] Используем локальные позиции линий: L={L:.3f}, R={R:.3f}")
            else:
                L = config.LINE_LEFT_X
                R = config.LINE_RIGHT_X
                logger.debug(f"[{self.stream_id}] Используем глобальные позиции линий: L={L:.3f}, R={R:.3f}")
            cy_iter: List[float] = centers_y if centers_y is not None else [0.5] * len(centers_x)
            
            for tid, cx, cy in zip(ids, centers_x, cy_iter):
                if tid is None:
                    continue
                
                prev = self._track_prev_x.get(tid)
                prev_y = getattr(self, '_track_prev_y', {}).get(tid)
                if not hasattr(self, '_track_prev_y'): self._track_prev_y = {}
                if not hasattr(self, '_track_is_inside'): self._track_is_inside = {}
                
                prev_inside = bool(self._track_is_inside.get(tid, (prev is not None and L <= prev <= R)))
                cur_inside = bool(L <= cx <= R)

                # Отладочная информация для понимания логики
                if prev is not None and prev_y is not None:
                    logger.debug(f"Track {tid}: prev=({prev:.3f}, {prev_y:.3f}) -> cur=({cx:.3f}, {cy:.3f}), "
                               f"prev_inside={prev_inside}, cur_inside={cur_inside}, lines=({L:.3f}, {R:.3f})")

                def _interp_y(px, py, qx, qy, lx):
                    """Интерполяция Y-координаты в точке пересечения линии"""
                    try:
                        # Проверяем, что линия не вертикальная
                        if abs(float(qx) - float(px)) < 1e-6:
                            return float(cy)
                        
                        # Линейная интерполяция
                        t = (float(lx) - float(px)) / (float(qx) - float(px))
                        interpolated_y = float(py) + t * (float(qy) - float(py))
                        
                        # Ограничиваем результат диапазоном [0, 1]
                        return max(0.0, min(1.0, interpolated_y))
                    except Exception as e:
                        logger.warning(f"Ошибка интерполяции Y: {e}, используем текущую Y={cy}")
                        return float(cy)

                if prev is not None and prev_y is not None:
                    # Проверяем общий cooldown для трека
                    track_cooldown_key = f"track_{tid}"
                    if now - self._track_last_side_time.get(track_cooldown_key, 0.0) < config.CROSS_COOLDOWN_SEC:
                        # Пропускаем событие из-за cooldown
                        self._track_prev_x[tid] = cx
                        self._track_prev_y[tid] = cy
                        self._track_is_inside[tid] = cur_inside
                        continue
                    
                    # enter events
                    if (not prev_inside) and cur_inside:
                        if prev < L <= cx:  # Вход слева (свинья идет вправо)
                            key = (tid, 'enter_left')
                            self.left_in += 1
                            self.total_crossings += 1
                            self.left_flow += 1
                            self._track_last_side_time[key] = now
                            self._track_last_side_time[track_cooldown_key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, L)
                            logger.info(f"🔵 L={L:.3f} y={y_at:.3f} t{tid} ←IN ({self.left_in}) | centroid: prev=({prev:.3f},{prev_y:.3f}) cur=({cx:.3f},{cy:.3f})")
                            self._recent_crossings.append({"id": int(tid), "side": "left", "mode": "enter", "x": float(L), "y": float(y_at), "ts": float(now)})
                            
                            # Журналирование события пересечения линии (неблокирующее)
                            if self.event_logger:
                                try:
                                    asyncio.create_task(self._log_crossing_async(
                                        'left', 'enter', int(tid), self.left_in, L, y_at
                                    ))
                                except Exception:
                                    pass
                            try:
                                self._act_crossings.append({
                                    "id": int(tid), "side": "left", "mode": "enter",
                                    "t": float(max(0.0, now - self._act_start_ts)),
                                    "x": float(L), "y": float(y_at),
                                    "count_est": int(self.reported_count)
                                })
                                if int(tid) not in self._left_cross_rank:
                                    self._left_cross_rank[int(tid)] = self._left_cross_counter
                                    self._left_cross_counter += 1
                            except Exception:
                                pass
                        elif cx < R <= prev:  # Вход справа (свинья идет влево)
                            key = (tid, 'enter_right')
                            self.right_in += 1
                            self.total_crossings += 1
                            self.right_flow += 1
                            self._track_last_side_time[key] = now
                            self._track_last_side_time[track_cooldown_key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, R)
                            logger.info(f"🔴 R={R:.3f} y={y_at:.3f} t{tid} ←IN ({self.right_in}) | centroid: prev=({prev:.3f},{prev_y:.3f}) cur=({cx:.3f},{cy:.3f})")
                            self._recent_crossings.append({"id": int(tid), "side": "right", "mode": "enter", "x": float(R), "y": float(y_at), "ts": float(now)})
                            
                            # Журналирование события пересечения линии (неблокирующее)
                            if self.event_logger:
                                try:
                                    asyncio.create_task(self._log_crossing_async(
                                        'right', 'enter', int(tid), self.right_in, R, y_at
                                    ))
                                except Exception:
                                    pass
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
                        if cx < L <= prev:  # Выход слева (свинья идет влево)
                            key = (tid, 'exit_left')
                            self.left_in = max(0, self.left_in - 1)
                            self.left_flow -= 1
                            self._track_last_side_time[key] = now
                            self._track_last_side_time[track_cooldown_key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, L)
                            logger.info(f"🔵 L={L:.3f} y={y_at:.3f} t{tid} OUT→ ({self.left_in})")
                            self._recent_crossings.append({"id": int(tid), "side": "left", "mode": "exit", "x": float(L), "y": float(y_at), "ts": float(now)})
                            
                            # Журналирование события выхода (неблокирующее)
                            if self.event_logger:
                                try:
                                    asyncio.create_task(self._log_crossing_async(
                                        'left', 'exit', int(tid), self.left_in, L, y_at
                                    ))
                                except Exception:
                                    pass
                            try:
                                self._act_crossings.append({
                                    "id": int(tid), "side": "left", "mode": "exit",
                                    "t": float(max(0.0, now - self._act_start_ts)),
                                    "x": float(L), "y": float(y_at),
                                    "count_est": int(self.reported_count)
                                })
                            except Exception:
                                pass
                        elif prev <= R < cx:  # Выход справа (свинья идет вправо, выходит через правую линию)
                            key = (tid, 'exit_right')
                            self.right_in = max(0, self.right_in - 1)
                            self.right_flow -= 1
                            self._track_last_side_time[key] = now
                            self._track_last_side_time[track_cooldown_key] = now
                            y_at = _interp_y(prev, prev_y, cx, cy, R)
                            logger.info(f"🔴 R={R:.3f} y={y_at:.3f} t{tid} OUT→ ({self.right_in}) | centroid: prev=({prev:.3f},{prev_y:.3f}) cur=({cx:.3f},{cy:.3f})")
                            self._recent_crossings.append({"id": int(tid), "side": "right", "mode": "exit", "x": float(R), "y": float(y_at), "ts": float(now)})
                            
                            # Журналирование события выхода (неблокирующее)
                            if self.event_logger:
                                try:
                                    asyncio.create_task(self._log_crossing_async(
                                        'right', 'exit', int(tid), self.right_in, R, y_at
                                    ))
                                except Exception:
                                    pass
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



    async def _log_crossing_async(self, side: str, direction: str, track_id: int, pig_count: int, x: float, y: float):
        """Асинхронное журналирование пересечения линии"""
        try:
            await self.event_logger.log_line_crossing(
                stream_id=self.stream_id,
                pig_count=pig_count,
                confidence=0.9,
                metadata={
                    'track_id': track_id,
                    'side': side,
                    'direction': direction,
                    'position': {'x': float(x), 'y': float(y)},
                    'total_crossings': self.total_crossings
                }
            )
        except Exception:
            pass  # Игнорируем ошибки журналирования

    async def _log_peak_async(self, pig_count: int, confidence: float, metadata: Dict[str, Any]):
        """Асинхронное журналирование пикового значения"""
        try:
            await self.event_logger.log_peak_count(
                stream_id=self.stream_id,
                pig_count=pig_count,
                confidence=confidence,
                metadata=metadata
            )
        except Exception:
            pass  # Игнорируем ошибки журналирования

    async def get_jpeg(self) -> Optional[bytes]:
        async with self.lock:
            frame_data = self.last_frame_data
            if isinstance(frame_data, dict):
                jpeg_bytes = frame_data.get('jpeg')
                if isinstance(jpeg_bytes, (bytes, bytearray)):
                    return bytes(jpeg_bytes)
            # Fallback for older structure just in case
            elif isinstance(frame_data, (bytes, bytearray)):
                return bytes(frame_data)
            return None

    async def _infer_loop(self):
        """Inference loop for processing frames"""
        return await _global_infer_loop(self)
    
    async def get_frame_data(self) -> Optional[Dict[str, Any]]:
        async with self.lock:
            return self.last_frame_data


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
        logger.info(f"🔍 _global_infer_loop вызван для потока {self.stream_id}")
        
        # Use the unified processor
        try:
            logger.info(f"🔍 Получаем процессор для потока {self.stream_id}")
            processor = await get_processor(self.stream_id)
            logger.info(f"🔍 Процессор получен: {processor}, активен: {processor.is_active if processor else 'None'}")
        except Exception as e:
            logger.error(f"❌ Failed to get processor for stream {self.stream_id}: {e}", exc_info=True)
            return

        if not processor or not processor.is_active:
            logger.error(f"❌ Unified processor for stream {self.stream_id} is not active. Inference loop will not run.")
            return

        logger.info(f"✅ Starting unified inference loop for stream {self.stream_id}")

        while self.running:
            try:
                frame_data = await self.get_frame_data()
                if not frame_data or not frame_data.get('jpeg'):
                    # Минимальная задержка для стабильности
                    await asyncio.sleep(0.001)  # 1ms
                    continue

                jpeg = frame_data['jpeg']
                timestamp = frame_data.get('ts', time.time()) # Get real timestamp
                
                # Проверяем синхронизацию для файлов
                if hasattr(self, 'duration') and self.duration > 0:
                    # Это файл - проверяем синхронизацию
                    # Проверка синхронизации без блокировки основного потока
                    # Используем timestamp вместо frame_time для лучшей стабильности
                    if not hasattr(self, '_last_frame_ts'):
                        self._last_frame_ts = timestamp
                    
                    frame_delta = timestamp - self._last_frame_ts
                    
                    # Если кадры приходят слишком медленно (> 1 сек между кадрами), логируем
                    if frame_delta > 1.0 and self._last_frame_ts > 0:
                        logger.warning(f"⏱️ Медленные кадры: delta={frame_delta:.2f}s")
                    
                    self._last_frame_ts = timestamp

                arr = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # 1. Process frame using the new async processor method
                    result: Optional[FrameResult] = await processor.process_frame_async(frame, timestamp=timestamp)

                    if result is None:
                        continue

                    # 2. Use results for tracking and counting (existing logic)
                    H0, W0 = result.original_shape
                    
                    # Build detections for SimpleTracker from FrameResult
                    dets = [{'bbox': bbox} for bbox in result.bboxes]
                    tracks = self.tracker.update(dets) if dets else []
                    ids = [t['id'] for t in tracks] if tracks else []

                    # Get centroids from FrameResult
                    centroids_x_norm = [c[0] / W0 for c in result.centroids]
                    centroids_y_norm = [c[1] / H0 for c in result.centroids]

                    await self._update_line_counters(ids, centroids_x_norm, centroids_y_norm)
                    
                    self.last_count = result.detections
                    try:
                        height = max(1.0, float(H0))
                        width = max(1.0, float(W0))
                    except Exception:
                        height = width = 1.0
                    normalized_masks = []
                    for poly in (result.masks or []):
                        try:
                            norm_poly = []
                            for point in poly:
                                x = float(point[0]) / width
                                y = float(point[1]) / height
                                x = max(0.0, min(1.0, x))
                                y = max(0.0, min(1.0, y))
                                norm_poly.append([x, y])
                            if norm_poly:
                                normalized_masks.append(norm_poly)
                        except Exception:
                            continue
                    self.last_masks = normalized_masks

                    # Экспорт полученных масок для диагностики
                    if self.last_masks:
                        logger.info(f"[{self.stream_id}] ⚠ Получено масок: {len(self.last_masks)} шт.")
                        logger.info(f"[{self.stream_id}] ❓ Первая маска: {type(self.last_masks[0])}, точек: {len(self.last_masks[0]) if hasattr(self.last_masks[0], '__len__') else 'N/A'}")
                    else:
                        logger.info(f"[{self.stream_id}] ⚠ Маски не поступили в этом результате")
                    # 3. Update statistics and broadcast (existing logic)
                    wnd_max = self.window_max.update(result.timestamp, int(self.last_count))
                    est = max(self.reported_count, wnd_max)
                    self.reported_count = est
                    
                    # Журналирование пиковых значений (неблокирующее)
                    if self.event_logger and est > self._act_peak:
                        try:
                            asyncio.create_task(self._log_peak_async(
                                pig_count=int(est),
                                confidence=result.confidence,
                                metadata={
                                    'previous_peak': self._act_peak,
                                    'detection_count': int(self.last_count),
                                    'window_max': wnd_max
                                }
                            ))
                            self._act_peak = int(est)
                        except Exception:
                            pass  # Игнорируем ошибки журналирования
                    
                    # Update act timeline
                    now_ts = result.timestamp
                    if (now_ts - self._last_timeline_ts) >= 0.5:
                        rel_t = float(max(0.0, now_ts - float(self._act_start_ts)))
                        self._act_timeline.append({"t": rel_t, "count_est": int(est)})
                        self._last_timeline_ts = now_ts

                    # Update act peak
                    if self.last_count > self._act_peak:
                        self._act_peak = self.last_count
                    
                    # Create payload
                    payload = {
                        "type": "count_update",
                        "count": int(round(est)),
                        "debug": {
                            "masks": self.last_masks,
                            "masks_debug": {
                                "count": len(self.last_masks) if self.last_masks else 0,
                                "has_masks": bool(self.last_masks),
                                "mask_types": [type(m).__name__ for m in (self.last_masks or [])]
                            },
                            "count_raw": int(self.last_count),
                            "flow": {"left_in": self.left_in, "right_in": self.right_in, "total_crossings": self.total_crossings},
                            "ids": ids,
                            "act": {
                                "seen_total": int(len(self._act_seen_labels)),
                                "peak_concurrent": int(self._act_peak),
                                "duration_sec": float(max(0.0, time.time() - self._act_start_ts))
                            },
                            # Отправляем актуальные позиции линий (локальные или глобальные)
                            "lines": self._get_current_line_positions(),
                            "crossings": list(self._recent_crossings)
                        }
                    }
                    await STREAM_MANAGER.broadcast(self.stream_id, payload)

            except Exception as e:
                logger.error(f"Infer loop error on {self.stream_id}: {e}", exc_info=True)
            
            # Адаптивная задержка для стабильности
            await asyncio.sleep(0.005)  # 5ms для предотвращения перегрузки CPU

    

class RtspStream(VideoStream):
    def __init__(self, stream_id: str, rtsp_url: str):
        super().__init__(stream_id)
        self.rtsp_url = rtsp_url

    async def start(self):
        if not getattr(self, 'running', False):
            self.running = True
            self._stream_task = asyncio.create_task(self._stream_loop())
            self._infer_task = asyncio.create_task(self._infer_loop())
        self.fps = 25.0  # Default RTSP framerate
        self.current_time = 0.0

    async def _stream_loop(self):
        try:
            meta = av_open_rtsp(self.stream_id, self.rtsp_url)
            self.fps = meta.get("fps", 25.0)
            
            frame_counter = 0
            start_time = time.time()
            
            while self.running:
                frame_data = av_read_jpeg(self.stream_id, timeout=2.0)
                if frame_data and isinstance(frame_data, dict) and frame_data.get('jpeg'):
                    async with self.lock:
                        self.last_frame_data = frame_data
                    
                    # Calculate current time for RTSP stream
                    frame_counter += 1
                    self.current_time = (time.time() - start_time)
                    
                    jpeg = frame_data['jpeg']
                    frame_id = frame_counter
                    
                    try:
                        if FRAME_BROKER is not None:
                            asyncio.create_task(FRAME_BROKER.publish(self.stream_id, frame_id, self.current_time, jpeg))
                    except Exception:
                        pass
                    
                    # Убрана задержка для максимальной производительности
                    # await asyncio.sleep(1.0 / self.fps)
                else:
                    # Минимальная задержка для стабильности
                    await asyncio.sleep(0.001)  # 1ms
        except Exception as e:
            logger.error(f"RTSP stream {self.stream_id} error: {e}")
        finally:
            ocv_close(self.stream_id)
            self.running = False

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
                
                frame_data = av_read_jpeg(self.stream_id, timeout=1.0)
                if frame_data and isinstance(frame_data, dict) and frame_data.get('jpeg'):
                    async with self.lock:
                        self.last_frame_data = frame_data
                    
                    jpeg = frame_data['jpeg']
                    pts = frame_data.get('pts')
                    time_base = frame_data.get('time_base')
                    
                    ts = self.current_time
                    if pts is not None and time_base is not None and time_base > 0:
                        ts = pts * time_base
                    
                    self.current_time = ts

                    frame_id = pts if pts is not None else int(ts * 1000)

                    try:
                        if FRAME_BROKER is not None:
                            asyncio.create_task(FRAME_BROKER.publish(self.stream_id, frame_id, ts, jpeg))
                    except Exception:
                        pass
                else:
                    break # Assume EOF
                # Минимальная задержка для стабильности
                await asyncio.sleep(0.001)  # 1ms
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
            frame_data = av_seek_read_jpeg(self.stream_id, self._seek_time)
            async with self.lock:
                self.last_frame_data = frame_data
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
        


class StreamManager:
    def __init__(self):
        self.streams: Dict[str, VideoStream] = {}
        self.websockets: Dict[str, List[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def get_or_create_stream(self, stream_id: str, source_uri: str) -> VideoStream:
        async with self.lock:
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
                if source_uri.startswith("rtsp://"):
                    self.streams[stream_id] = RtspStream(stream_id, source_uri)
                else:
                    self.streams[stream_id] = FileStream(stream_id, source_uri)

                # Загружаем сохраненные позиции линий
                stream = self.streams[stream_id]
                all_positions = load_line_positions()
                
                file_key = stream.file_path if isinstance(stream, FileStream) else stream_id
                if "/" in file_key:
                    file_key = file_key.split("/")[-1]

                if 'files' in all_positions and file_key in all_positions['files']:
                    stream.line_positions = all_positions['files'][file_key]
                    logger.info(f"Loaded line positions for {file_key}")
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

    async def register_websocket(self, stream_id: str, ws: WebSocket):
        async with self.lock:
            if stream_id not in self.websockets:
                self.websockets[stream_id] = []
            self.websockets[stream_id].append(ws)

    async def unregister_websocket(self, stream_id: str, ws: WebSocket):
        async with self.lock:
            if stream_id in self.websockets:
                self.websockets[stream_id].remove(ws)

    async def broadcast(self, stream_id: str, data: dict):
        if stream_id in self.websockets:
            # Отправляем данные асинхронно без блокировки
            tasks = []
            for ws in self.websockets[stream_id]:
                try:
                    task = asyncio.create_task(ws.send_json(data))
                    tasks.append(task)
                except Exception as e:
                    logger.warning(f"Ошибка отправки WebSocket данных: {e}")
            
            # Ждем завершения всех отправок с таймаутом
            if tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning(f"Таймаут отправки WebSocket данных для {stream_id}")

STREAM_MANAGER = StreamManager()
# attach frame broker and start inference workers on-demand
try:
    from core.frame_broker import FRAME_BROKER

    from core.results_store import RESULTS_STORE
except Exception:
    FRAME_BROKER = None
    start_global_worker_for = None
    RESULTS_STORE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 PigWeight API starting up...")
    
    # Запускаем фоновую задачу очистки событий
    cleanup_task = None
    try:
        from services.event_logger import get_event_logger
        event_logger = get_event_logger()
        
        async def cleanup_events_periodically():
            """Периодическая очистка старых событий"""
            while True:
                try:
                    await asyncio.sleep(3600)  # Каждый час
                    await event_logger.cleanup_old_events(max_age_hours=24)
                    logger.info("✅ Автоматическая очистка событий выполнена")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Ошибка при очистке событий: {e}")
        
        cleanup_task = asyncio.create_task(cleanup_events_periodically())
        logger.info("✅ Фоновая задача очистки событий запущена")
        
    except ImportError:
        logger.warning("⚠️ Система событий недоступна")
    
    yield
    
    # Shutdown
    logger.info("🛑 PigWeight API shutting down...")
    
    # Останавливаем фоновую задачу
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    
    # Останавливаем потоки
    for stream in list(STREAM_MANAGER.streams.values()):
        await stream.stop()
    
    logger.info("✅ PigWeight API shutdown complete")

_DEFAULT_RESPONSE = ORJSONResponse if _HAVE_ORJSON else JSONResponse
app = FastAPI(title="PigWeight API v3.0 (Unified)", lifespan=lifespan, default_response_class=_DEFAULT_RESPONSE)

# Initialize shared dependencies
from api.dependencies import init_dependencies
init_dependencies(STREAM_MANAGER, config.TARGET_FPS, FileStream, perf_logger, av_meta, RECORDS_DIR)

# Setup middleware
from api.middleware import setup_cors, setup_error_handling, setup_request_logging, setup_security_headers
setup_cors(app)
setup_error_handling(app)
setup_request_logging(app)
setup_security_headers(app)

# Подключаем эндпоинты из модулей
from api.endpoints import video, stream, health, files, diagnostics, events, records, verification, system, validation
app.include_router(health.router, tags=["health"])
app.include_router(video.router, tags=["video"])
app.include_router(stream.router, tags=["stream"])
app.include_router(files.router, tags=["files"])
app.include_router(diagnostics.router, tags=["diagnostics"])
app.include_router(events.router, tags=["events"])
app.include_router(records.router, tags=["records"])
app.include_router(system.router, tags=["system"])
app.include_router(verification.router, tags=["verification"])
app.include_router(validation.router, tags=["validation"], prefix="/api")

# Include WebRTC routes
from api import webrtc
webrtc.init_webrtc(app, STREAM_MANAGER, FRAME_BROKER, config)

# Подключаем упрощенные endpoints

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



# Moved to api/endpoints/health.py
# @app.get("/api/health")

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

@app.get("/monitoring", response_class=HTMLResponse)
async def read_monitoring():
    """Страница мониторинга системы"""
    return FileResponse(STATIC_DIR / "monitoring.html")

# Moved to api/endpoints/stream.py
# @app.post("/api/stream/start")

# Moved to api/endpoints/stream.py
# @app.get("/api/stream/{stream_id}/stop")

# Moved to api/endpoints/stream.py
# @app.get("/api/stream/{stream_id}/snapshot")

# Moved to api/endpoints/stream.py
# mjpeg_generator and @app.get("/api/stream/{stream_id}/feed")

@app.get("/api/cameras")
async def api_cameras():
    """Return available cameras as defined in .env (env vars)."""
    logger.info("🔍 API /api/cameras вызван")
    result = cameras_from_env()
    logger.info(f"📹 API возвращает камеры: {result}")
    return result

# Moved to api/endpoints/stream.py
# @app.get("/api/stream/{stream_id}/info")
# @app.get("/api/stream/{stream_id}/seek") - moved to api/endpoints/stream.py

@app.post("/api/lines")
async def api_set_lines(data: Dict[str, float]):
    global LINE_LEFT_X, LINE_RIGHT_X
    left_x = data.get('left_x')
    right_x = data.get('right_x')
    if left_x is not None:
        LINE_LEFT_X = _clamp01(left_x)
    if right_x is not None:
        LINE_RIGHT_X = _clamp01(right_x)
    return {"status": "ok", "left_x": LINE_LEFT_X, "right_x": LINE_RIGHT_X}

@app.post("/api/stream/{stream_id}/line_positions")
async def api_set_line_positions(stream_id: str, positions: Dict[str, Any] = Body(...)):
    """Сохранить позиции линий для конкретного видеофайла."""
    try:
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse({"error": f"Stream {stream_id} not found"}, status_code=404)
        
        # Определяем ключ для файла (без пути)
        file_key = stream.file_path if isinstance(stream, FileStream) else stream_id
        if "/" in file_key:
            file_key = file_key.split("/")[-1]

        # Загружаем, обновляем и сохраняем
        all_positions = load_line_positions()
        if 'files' not in all_positions:
            all_positions['files'] = {}
        all_positions['files'][file_key] = positions
        save_line_positions(all_positions)

        # Также обновляем в текущем стриме
        stream.line_positions = positions
        
        return {"status": "success", "message": f"Line positions saved for {file_key}"}
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
            logger.info(f"Настройка потока {stream_id} под транспорт WebRTC")
            return {"status": "tuned", "transport": "webrtc"}

        return {"status": "no_change", "transport": transport}
    except Exception as e:
        logger.error(f"Error Tuning stream {stream_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



# === API для работы с линиями ===

LINE_POSITIONS_FILE = "line_positions.json"

def load_line_positions():
    """Загрузка позиций линий из JSON файла"""
    try:
        if os.path.exists(LINE_POSITIONS_FILE):
            with open(LINE_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading line positions: {e}")
        return {}

def save_line_positions(positions):
    """Сохранение позиций линий в JSON файл"""
    try:
        with open(LINE_POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(positions, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving line positions: {e}")
        return False

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

@app.post("/api/stream/{stream_id}/line_positions")
async def api_set_line_positions(stream_id: str, positions: Dict[str, Any] = Body(...)):
    """Сохранить позиции линий для конкретного видеофайла."""
    try:
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse({"error": f"Stream {stream_id} not found"}, status_code=404)
        
        # Определяем ключ для файла (без пути)
        file_key = stream.file_path if isinstance(stream, FileStream) else stream_id
        if "/" in file_key:
            file_key = file_key.split("/")[-1]

        # Загружаем, обновляем и сохраняем
        all_positions = load_line_positions()
        if 'files' not in all_positions:
            all_positions['files'] = {}
        all_positions['files'][file_key] = positions
        save_line_positions(all_positions)

        # Также обновляем в текущем стриме
        stream.line_positions = positions
        
        # Обновляем позиции линий в процессоре для корректной детекции пересечений
        try:
            from core.processor import get_processor
            processor = await get_processor(stream_id)
            if processor and processor.is_active:
                left_x = float(positions.get('left_x', config.LINE_LEFT_X))
                right_x = float(positions.get('right_x', config.LINE_RIGHT_X))
                processor.update_line_positions(left_x, right_x)
                logger.info(f"[успешно] Позиции линий обновлены в процессоре: L={left_x:.3f}, R={right_x:.3f}")
        except Exception as e:
            logger.warning(f"[предупреждение] Не удалось обновить позиции линий в процессоре: {e}")
        
        return {"status": "success", "message": f"Line positions saved for {file_key}"}
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
            logger.info(f"Настройка потока {stream_id} под транспорт WebRTC")
            return {"status": "tuned", "transport": "webrtc"}

        return {"status": "no_change", "transport": transport}
    except Exception as e:
        logger.error(f"Error Tuning stream {stream_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



# === API для работы с линиями ===

LINE_POSITIONS_FILE = "line_positions.json"

def load_line_positions():
    """Загрузка позиций линий из JSON файла"""
    try:
        if os.path.exists(LINE_POSITIONS_FILE):
            with open(LINE_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading line positions: {e}")
        return {}

def save_line_positions(positions):
    """Сохранение позиций линий в JSON файл"""
    try:
        with open(LINE_POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(positions, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving line positions: {e}")
        return False

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

# === API для загрузки видеофайлов ===
# ОТКЛЮЧЕН: используется модульный endpoint из api/endpoints/video.py

# @app.post("/api/upload")
async def upload_video_file(file: UploadFile = File(...)):
    """Загрузка видеофайла для обработки"""
    try:
        # Валидация типа файла
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            return JSONResponse(
                {"error": f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"},
                status_code=400
            )
        
        # Проверка размера файла (максимум 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        file_content = await file.read()
        
        if len(file_content) > max_size:
            return JSONResponse(
                {"error": "Файл слишком большой. Максимальный размер: 500MB"},
                status_code=400
            )
        
        # Создание уникального имени файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Сохранение файла
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Video file uploaded: {safe_filename}, size: {len(file_content)} bytes")
        
        return {
            "status": "success",
            "filename": safe_filename,
            "path": str(file_path),
            "size": len(file_content),
            "message": "Файл успешно загружен"
        }
        
    except Exception as e:
        logger.error(f"Error uploading video file: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === API для управления актами взвешивания ===

@app.post("/api/weighing/manual/save")
async def save_manual_weighing(data: Dict[str, Any]):
    """Сохранение ручного акта взвешивания из Live панели"""
    try:
        # Валидация данных
        required_fields = ['count', 'total_weight']
        for field in required_fields:
            if field not in data:
                return JSONResponse(
                    {"error": f"Отсутствует обязательное поле: {field}"},
                    status_code=400
                )
        
        count = int(data['count'])
        total_weight = float(data['total_weight'])
        
        if count <= 0 or total_weight <= 0:
            return JSONResponse(
                {"error": "Количество и вес должны быть больше нуля"},
                status_code=400
            )
        
        # Создание записи акта
        act_data = {
            'id': f"manual_{int(time.time())}",
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'group': data.get('group', 'Ручной ввод'),
            'total': count,
            'weight': total_weight,
            'avg_weight': round(total_weight / count, 2),
            'source': 'manual',
            'stream_id': data.get('stream_id', 'manual'),
            'created_at': datetime.now().isoformat()
        }
        
        # Сохранение в файл
        acts_file = RECORDS_DIR / "weighing_acts.json"
        acts = []
        
        if acts_file.exists():
            try:
                with open(acts_file, 'r', encoding='utf-8') as f:
                    acts = json.load(f)
            except Exception:
                acts = []
        
        acts.append(act_data)
        
        with open(acts_file, 'w', encoding='utf-8') as f:
            json.dump(acts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Manual weighing act saved: {act_data['id']}")
        
        return {
            "status": "success",
            "act_id": act_data['id'],
            "message": "Акт взвешивания сохранен"
        }
        
    except Exception as e:
        logger.error(f"Error saving manual weighing: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/weighing/stats")
async def get_weighing_stats(
    date_from: str = Query(None),
    date_to: str = Query(None),
    stream_id: str = Query(None)
):
    """Получение статистики актов взвешивания"""
    try:
        acts_file = RECORDS_DIR / "weighing_acts.json"
        if not acts_file.exists():
            return {
                "total_acts": 0,
                "total_count": 0,
                "total_weight": 0,
                "avg_weight": 0
            }
        
        with open(acts_file, 'r', encoding='utf-8') as f:
            acts = json.load(f)
        
        # Фильтрация по датам
        if date_from or date_to:
            filtered_acts = []
            for act in acts:
                act_date = act.get('date', '')
                if date_from and act_date < date_from:
                    continue
                if date_to and act_date > date_to:
                    continue
                filtered_acts.append(act)
            acts = filtered_acts
        
        # Фильтрация по stream_id
        if stream_id:
            acts = [act for act in acts if act.get('stream_id') == stream_id]
        
        # Вычисление статистики
        total_acts = len(acts)
        total_count = sum(act.get('total', 0) for act in acts)
        total_weight = sum(act.get('weight', 0) for act in acts)
        avg_weight = round(total_weight / total_count, 2) if total_count > 0 else 0
        
        return {
            "total_acts": total_acts,
            "total_count": total_count,
            "total_weight": round(total_weight, 2),
            "avg_weight": avg_weight
        }
        
    except Exception as e:
        logger.error(f"Error getting weighing stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === API для экспорта данных ===

@app.get("/api/export/weighing")
async def export_weighing_data(
    format: str = Query("excel", pattern="^(excel|csv)$"),
    date_from: str = Query(None),
    date_to: str = Query(None)
):
    """Экспорт данных актов взвешивания в Excel или CSV"""
    try:
        acts_file = RECORDS_DIR / "weighing_acts.json"
        if not acts_file.exists():
            return JSONResponse({"error": "Нет данных для экспорта"}, status_code=404)
        
        with open(acts_file, 'r', encoding='utf-8') as f:
            acts = json.load(f)
        
        # Фильтрация по датам
        if date_from or date_to:
            filtered_acts = []
            for act in acts:
                act_date = act.get('date', '')
                if date_from and act_date < date_from:
                    continue
                if date_to and act_date > date_to:
                    continue
                filtered_acts.append(act)
            acts = filtered_acts
        
        if not acts:
            return JSONResponse({"error": "Нет данных в указанном диапазоне дат"}, status_code=404)
        
        # Подготовка данных для экспорта
        export_data = []
        for act in acts:
            export_data.append({
                'Дата': act.get('date', ''),
                'Время': act.get('time', ''),
                'Группа': act.get('group', ''),
                'Количество голов': act.get('total', 0),
                'Общий вес (кг)': act.get('weight', 0),
                'Средний вес (кг)': act.get('avg_weight', 0),
                'Источник': act.get('source', ''),
                'ID потока': act.get('stream_id', '')
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "excel":
            try:
                import pandas as pd
                df = pd.DataFrame(export_data)
                filename = f"weighing_export_{timestamp}.xlsx"
                filepath = RECORDS_DIR / filename
                df.to_excel(filepath, index=False, engine='openpyxl')
                
                return FileResponse(
                    filepath,
                    filename=filename,
                    media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            except ImportError:
                # Fallback to CSV if pandas/openpyxl not available
                format = "csv"
        
        if format == "csv":
            filename = f"weighing_export_{timestamp}.csv"
            filepath = RECORDS_DIR / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                if export_data:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(export_data)
            
            return FileResponse(
                filepath,
                filename=filename,
                media_type='text/csv'
            )
        
    except Exception as e:
        logger.error(f"Error exporting weighing data: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === API для загрузки видеофайлов ===
# ОТКЛЮЧЕН: используется модульный endpoint из api/endpoints/video.py

# @app.post("/api/upload")
async def upload_video_file(file: UploadFile = File(...)):
    """Загрузка видеофайла для обработки"""
    try:
        # Валидация типа файла
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            return JSONResponse(
                {"error": f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"},
                status_code=400
            )
        
        # Проверка размера файла (максимум 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        file_content = await file.read()
        
        if len(file_content) > max_size:
            return JSONResponse(
                {"error": "Файл слишком большой. Максимальный размер: 500MB"},
                status_code=400
            )
        
        # Создание безопасного имени файла без timestamp префикса
        # Сохраняем оригинальное имя файла для предотвращения искажения
        import re
        
        # Очищаем имя файла от потенциально опасных символов
        safe_filename = re.sub(r'[^\w\-_\.]', '_', file.filename)
        
        # Если файл с таким именем уже существует, добавляем уникальный суффикс
        file_path = UPLOAD_DIR / safe_filename
        if file_path.exists():
            name_part = file_path.stem
            extension = file_path.suffix
            counter = 1
            while file_path.exists():
                safe_filename = f"{name_part}_{counter}{extension}"
                file_path = UPLOAD_DIR / safe_filename
                counter += 1
        
        # Сохранение файла
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Video file uploaded: {safe_filename}, size: {len(file_content)} bytes")
        
        return {
            "status": "success",
            "filename": safe_filename,
            "path": str(file_path),
            "size": len(file_content),
            "message": "Файл успешно загружен"
        }
        
    except Exception as e:
        logger.error(f"Error uploading video file: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === API для управления актами взвешивания ===

@app.post("/api/weighing/manual/save")
async def save_manual_weighing(data: Dict[str, Any]):
    """Сохранение ручного акта взвешивания из Live панели"""
    try:
        # Валидация данных
        required_fields = ['count', 'total_weight']
        for field in required_fields:
            if field not in data:
                return JSONResponse(
                    {"error": f"Отсутствует обязательное поле: {field}"},
                    status_code=400
                )
        
        count = int(data['count'])
        total_weight = float(data['total_weight'])
        
        if count <= 0 or total_weight <= 0:
            return JSONResponse(
                {"error": "Количество и вес должны быть больше нуля"},
                status_code=400
            )
        
        # Создание записи акта
        act_data = {
            'id': f"manual_{int(time.time())}",
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'group': data.get('group', 'Ручной ввод'),
            'total': count,
            'weight': total_weight,
            'avg_weight': round(total_weight / count, 2),
            'source': 'manual',
            'stream_id': data.get('stream_id', 'manual'),
            'created_at': datetime.now().isoformat()
        }
        
        # Сохранение в файл
        acts_file = RECORDS_DIR / "weighing_acts.json"
        acts = []
        
        if acts_file.exists():
            try:
                with open(acts_file, 'r', encoding='utf-8') as f:
                    acts = json.load(f)
            except Exception:
                acts = []
        
        acts.append(act_data)
        
        with open(acts_file, 'w', encoding='utf-8') as f:
            json.dump(acts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Manual weighing act saved: {act_data['id']}")
        
        return {
            "status": "success",
            "act_id": act_data['id'],
            "message": "Акт взвешивания сохранен"
        }
        
    except Exception as e:
        logger.error(f"Error saving manual weighing: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/weighing/stats")
async def get_weighing_stats(
    date_from: str = Query(None),
    date_to: str = Query(None),
    stream_id: str = Query(None)
):
    """Получение статистики актов взвешивания"""
    try:
        acts_file = RECORDS_DIR / "weighing_acts.json"
        if not acts_file.exists():
            return {
                "total_acts": 0,
                "total_count": 0,
                "total_weight": 0,
                "avg_weight": 0
            }
        
        with open(acts_file, 'r', encoding='utf-8') as f:
            acts = json.load(f)
        
        # Фильтрация по датам
        if date_from or date_to:
            filtered_acts = []
            for act in acts:
                act_date = act.get('date', '')
                if date_from and act_date < date_from:
                    continue
                if date_to and act_date > date_to:
                    continue
                filtered_acts.append(act)
            acts = filtered_acts
        
        # Фильтрация по stream_id
        if stream_id:
            acts = [act for act in acts if act.get('stream_id') == stream_id]
        
        # Вычисление статистики
        total_acts = len(acts)
        total_count = sum(act.get('total', 0) for act in acts)
        total_weight = sum(act.get('weight', 0) for act in acts)
        avg_weight = round(total_weight / total_count, 2) if total_count > 0 else 0
        
        return {
            "total_acts": total_acts,
            "total_count": total_count,
            "total_weight": round(total_weight, 2),
            "avg_weight": avg_weight
        }
        
    except Exception as e:
        logger.error(f"Error getting weighing stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === API для журнала актов взвешивания ===

@app.get("/api/journal/list")
async def get_journal_acts(
    date_from: str = Query(None),
    date_to: str = Query(None),
    camera: str = Query(None),
    limit: int = Query(100)
):
    """Получить список актов взвешивания с фильтрацией"""
    try:
        acts_file = RECORDS_DIR / "weighing_acts.json"
        if not acts_file.exists():
            return {
                "acts": [],
                "summary": {
                    "total_acts": 0,
                    "total_count": 0,
                    "total_weight": 0,
                    "avg_weight": 0
                }
            }
        
        with open(acts_file, 'r', encoding='utf-8') as f:
            acts = json.load(f)
        
        # Применяем фильтры
        filtered_acts = []
        for act in acts:
            # Фильтр по дате
            if date_from and act.get('date', '') < date_from:
                continue
            if date_to and act.get('date', '') > date_to:
                continue
            # Фильтр по камере/потоку
            if camera and act.get('stream_id', '') != camera:
                continue
            
            filtered_acts.append(act)
        
        # Сортируем по дате (новые сверху)
        filtered_acts.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
        
        # Ограничиваем количество
        filtered_acts = filtered_acts[:limit]
        
        # Добавляем статистику
        total_count = sum(act.get('total', 0) for act in filtered_acts)
        total_weight = sum(act.get('weight', 0) for act in filtered_acts)
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
        logger.error(f"Error getting journal acts: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/journal/export")
async def export_journal_acts(
    format: str = Query("csv", pattern="^(csv|excel)$"),
    date_from: str = Query(None),
    date_to: str = Query(None)
):
    """Экспорт актов взвешивания в CSV или Excel"""
    try:
        acts_file = RECORDS_DIR / "weighing_acts.json"
        if not acts_file.exists():
            return JSONResponse({"error": "Нет данных для экспорта"}, status_code=404)
        
        with open(acts_file, 'r', encoding='utf-8') as f:
            acts = json.load(f)
        
        # Фильтрация по датам
        if date_from or date_to:
            filtered_acts = []
            for act in acts:
                act_date = act.get('date', '')
                if date_from and act_date < date_from:
                    continue
                if date_to and act_date > date_to:
                    continue
                filtered_acts.append(act)
            acts = filtered_acts
        
        if not acts:
            return JSONResponse({"error": "Нет данных в указанном диапазоне дат"}, status_code=404)
        
        # Подготовка данных для экспорта
        export_data = []
        for act in acts:
            export_data.append({
                'Дата': act.get('date', ''),
                'Время': act.get('time', ''),
                'Группа': act.get('group', ''),
                'Количество голов': act.get('total', 0),
                'Общий вес (кг)': act.get('weight', 0),
                'Средний вес (кг)': act.get('avg_weight', 0),
                'Источник': act.get('source', ''),
                'ID потока': act.get('stream_id', '')
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"journal_export_{timestamp}.csv"
        filepath = RECORDS_DIR / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if export_data:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)
        
        return FileResponse(
            filepath,
            filename=filename,
            media_type='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting journal acts: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws/count")
async def websocket_endpoint(ws: WebSocket, id: str = Query(...)):
    await ws.accept()
    await STREAM_MANAGER.register_websocket(id, ws)
    try:
        while True:
            # Keep connection open to receive broadcasts
            await ws.receive_text() 
    except WebSocketDisconnect:
        await STREAM_MANAGER.unregister_websocket(id, ws)


# === API для записей (records) ===

@app.get("/api/records")
async def api_records_list():
    """Получить список всех записей актов взвешивания"""
    try:
        items = []
        for p in sorted(RECORDS_DIR.glob("act_*.json")):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Извлекаем дату и время из имени файла
                    filename = p.stem
                    parsed_date = ""
                    parsed_time = ""
                    
                    # Парсим имя файла: act_file_1757326054497_20250908-131236
                    if "_" in filename:
                        parts = filename.split("_")
                        if len(parts) >= 4:  # act, file, timestamp, date-time
                            # Последняя часть содержит дату-время в формате YYYYMMDD-HHMMSS
                            date_time_part = parts[-1]
                            if "-" in date_time_part and len(date_time_part) == 15:  # YYYYMMDD-HHMMSS
                                try:
                                    date_part = date_time_part[:8]  # YYYYMMDD
                                    time_part = date_time_part[9:]  # HHMMSS
                                    
                                    # Парсим дату
                                    year = int(date_part[:4])
                                    month = int(date_part[4:6])
                                    day = int(date_part[6:8])
                                    
                                    # Парсим время
                                    hour = int(time_part[:2])
                                    minute = int(time_part[2:4])
                                    second = int(time_part[4:6])
                                    
                                    dt = datetime(year, month, day, hour, minute, second)
                                    parsed_date = dt.strftime("%Y-%m-%d")
                                    parsed_time = dt.strftime("%H:%M:%S")
                                except:
                                    # Fallback: используем timestamp из данных
                                    if 'started_at' in data:
                                        try:
                                            dt = datetime.fromtimestamp(data['started_at'])
                                            parsed_date = dt.strftime("%Y-%m-%d")
                                            parsed_time = dt.strftime("%H:%M:%S")
                                        except:
                                            parsed_date = "Неизвестная дата"
                                            parsed_time = ""
                                    else:
                                        parsed_date = "Неизвестная дата"
                                        parsed_time = ""
                            else:
                                # Fallback: используем timestamp из данных
                                if 'started_at' in data:
                                    try:
                                        dt = datetime.fromtimestamp(data['started_at'])
                                        parsed_date = dt.strftime("%Y-%m-%d")
                                        parsed_time = dt.strftime("%H:%M:%S")
                                    except:
                                        parsed_date = "Неизвестная дата"
                                        parsed_time = ""
                                else:
                                    parsed_date = "Неизвестная дата"
                                    parsed_time = ""
                        else:
                            # Fallback: используем timestamp из данных
                            if 'started_at' in data:
                                try:
                                    dt = datetime.fromtimestamp(data['started_at'])
                                    parsed_date = dt.strftime("%Y-%m-%d")
                                    parsed_time = dt.strftime("%H:%M:%S")
                                except:
                                    parsed_date = "Неизвестная дата"
                                    parsed_time = ""
                            else:
                                parsed_date = "Неизвестная дата"
                                parsed_time = ""
                    else:
                        # Fallback: используем timestamp из данных
                        if 'started_at' in data:
                            try:
                                dt = datetime.fromtimestamp(data['started_at'])
                                parsed_date = dt.strftime("%Y-%m-%d")
                                parsed_time = dt.strftime("%H:%M:%S")
                            except:
                                parsed_date = "Неизвестная дата"
                                parsed_time = ""
                        else:
                            parsed_date = "Неизвестная дата"
                            parsed_time = ""
                    
                    # Определяем участок взвешивания из stream_id
                    stream_id = data.get("stream_id", "")
                    weighing_section = "Неизвестный участок"
                    if stream_id.startswith("file_"):
                        # Для файлов используем имя файла как участок
                        weighing_section = stream_id.replace("file_", "Файл ")
                    elif stream_id.startswith("rtsp_"):
                        # Для RTSP используем ID камеры
                        weighing_section = f"Камера {stream_id.replace('rtsp_', '')}"
                    elif stream_id.startswith("demo_"):
                        # Для демо потока
                        weighing_section = "Демо поток"
                    else:
                        weighing_section = stream_id
                    
                    items.append({
                        "name": p.stem,
                        "date": parsed_date,
                        "time": parsed_time,
                        "group": data.get("stream_id", ""),
                        "weighing_section": weighing_section,
                        "total_count": data.get("seen_total", 0),
                        "total_weight": 0,  # В текущем формате нет веса
                        "avg_weight": 0,    # В текущем формате нет веса
                        "duration": data.get("duration_sec", 0),
                        "peak_concurrent": data.get("peak_concurrent", 0)
                    })
            except Exception as e:
                logger.warning(f"Ошибка чтения файла {p}: {e}")
                continue
        return JSONResponse({"records": items})
    except Exception as e:
        logger.error(f"Ошибка получения списка записей: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/records/{act_name}")
async def api_record_details(act_name: str):
    """Получить детали конкретной записи"""
    try:
        # Sanitize filename
        if ".." in act_name or "/" in act_name or "\\" in act_name:
            raise HTTPException(status_code=400, detail="Invalid act name")
        
        file_path = RECORDS_DIR / f"{act_name}.json"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Record not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения деталей записи {act_name}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

