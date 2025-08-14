import os
import cv2
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Generator, Any
import colorsys

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

# PyAV больше не используется в MVP: декодирование вынесено в отдельный процесс OpenCVIsolate
_PYAV_AVAILABLE = False

# --- Bootstrap ---
BASE_DIR = Path(__file__).parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Perf logger to separate file logs/perf.log
LOG_DIR = (Path(__file__).parent.parent / "logs")
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
perf_logger = logging.getLogger("perf")
perf_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('perf.log') for h in perf_logger.handlers):
    try:
        fh = logging.FileHandler(str(LOG_DIR / "perf.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        perf_logger.addHandler(fh)
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app):
    # startup
    try:
        try:
            get_ocv().ping()
            logger.info("OpenCV worker warm-up done")
        except Exception as e:
            logger.warning("OpenCV worker warm-up failed: %s", e)
        yield
    finally:
        # shutdown (performed reliably via lifespan)
        logger.info("Shutting down server, cleaning up resources...")
        # Stop camera streams
        try:
            for cam in list(CAMERAS.cams.values()):
                try:
                    await cam.stop()
                except Exception:
                    pass
        except Exception:
            pass
        # Close file sessions / release caps
        try:
            for sess in list(_file_sessions.values()):
                try:
                    if sess.get("type") == "opencv" and sess.get("cap"):
                        sess["cap"].release()
                    elif sess.get("type") == "pyav" and sess.get("container"):
                        sess["container"].close()
                except Exception:
                    pass
        except Exception:
            pass
        logger.info("All resources cleaned up.")
app = FastAPI(title="PigWeight API (FastAPI)", lifespan=lifespan)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static and models if exist
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
MODELS_DIR = BASE_DIR / "models"
if MODELS_DIR.exists():
    app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")

# --- Config from environment ---
CAM_DEFAULT = os.getenv("CAM_DEFAULT", "rtsp://admin:Qwerty.123@10.15.6.24/1/1")
CAM_URL = os.getenv("CAM_URL", CAM_DEFAULT)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
TARGET_FPS = float(os.getenv("FPS", "12"))
BOUNDARY = "frame"

# Model config (строго .pt из каталога ./models)
DETECTION_MODE = os.getenv("DETECTION_MODE", "pig-only").lower()
PIG_MODEL_PATH = os.getenv("PIG_MODEL_PATH", "models/pig_yolo11-seg.pt")
PIG_CLASS_ID = int(os.getenv("PIG_CLASS_ID", "0"))

# Выбор эффективной модели и классов
if DETECTION_MODE == "pig-only":
    MODEL_PATH = PIG_MODEL_PATH
    TARGET_CLASS_IDS = {PIG_CLASS_ID}
else:
    TARGET_CLASS_IDS = set(map(int, os.getenv("TARGET_CLASS_IDS", "20,17,19").split(",")))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.30"))
AVG_WINDOW = int(os.getenv("AVG_WINDOW", "20"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))

# Balanced counting config
BALANCED_DEFAULT = os.getenv("BALANCED_DEFAULT", "false").lower() == "true"
def _parse_class_weights(s: str) -> Dict[int, float]:
    res: Dict[int, float] = {}
    try:
        for part in (s or "").split(","):
            if not part.strip():
                continue
            k, v = part.split(":")
            res[int(k.strip())] = float(v.strip())
    except Exception:
        pass
    return res or {}
CLASS_WEIGHTS = _parse_class_weights(os.getenv("CLASS_WEIGHTS", "20:1.0,17:1.0,19:1.0"))
BALANCE_ALPHA = float(os.getenv("BALANCE_ALPHA", "0.85"))
BALANCE_BIAS = float(os.getenv("BALANCE_BIAS", "0.0"))
BALANCE_CAP = float(os.getenv("BALANCE_CAP", "150"))

# Auto-calibration (без эталона): держим окно и подгоняем alpha/bias для стабилизации в целевом коридоре
AUTO_CALIBRATE = os.getenv("AUTO_CALIBRATE", "true").lower() == "true"
CALIB_TARGET_MIN = float(os.getenv("CALIB_TARGET_MIN", "70"))
CALIB_TARGET_MAX = float(os.getenv("CALIB_TARGET_MAX", "80"))
CALIB_WINDOW = int(os.getenv("CALIB_WINDOW", "90"))  # ~7-8 сек при 12 FPS и скипе
CALIB_LR_ALPHA = float(os.getenv("CALIB_LR_ALPHA", "0.002"))
CALIB_LR_BIAS = float(os.getenv("CALIB_LR_BIAS", "0.05"))

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Ленивая инициализация изолятора OpenCV (важно для Windows spawn)
OCV: Optional[OpenCVIsolate] = None
def get_ocv() -> OpenCVIsolate:
    global OCV
    if OCV is None:
        OCV = OpenCVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=TARGET_FPS)
    return OCV

# Безопасный вызов методов воркера с авто-респауном
def _ocv_safe_call(method_name: str, *args, **kwargs):
    try:
        ocv = get_ocv()
        method = getattr(ocv, method_name, None)
        if not method:
            raise AttributeError(f"OpenCVIsolate lacks {method_name}")
        return method(*args, **kwargs)
    except Exception:
        # Попытка респаунить воркер и повторить один раз
        try:
            global OCV
            OCV = OpenCVIsolate(jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")), target_fps=TARGET_FPS)
            method = getattr(OCV, method_name, None)
            if not method:
                raise AttributeError(f"OpenCVIsolate lacks {method_name}")
            return method(*args, **kwargs)
        except Exception as e2:
            raise e2

# Адаптер для совместимости с разными реализациями воркера
def ocv_open_rtsp(stream_id: str, url: str) -> Dict[str, Any]:
    # Use slightly longer timeout for RTSP open to balance responsiveness and unstable cameras
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

def encode_image(frame) -> bytes:
    # Оптимизированное JPEG кодирование для экономии трафика
    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Включаем оптимизацию
        cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Прогрессивный JPEG для веба
    ]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        return b""
    return buf.tobytes()

def encode_jpeg(frame, quality: int = None) -> bytes:
    # Оптимизированное JPEG кодирование для файловых операций
    q = quality or JPEG_QUALITY
    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY, q,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Включаем оптимизацию
        cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Прогрессивный JPEG для веба
    ]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        return b""
    return buf.tobytes()

def multipart_chunk(img_bytes: bytes) -> bytes:
    header = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(img_bytes)}\r\n\r\n"
    ).encode("utf-8")
    return header + img_bytes + b"\r\n"

# --- Local OpenCV fallback (для файлов) ---
def _open_file_cap_local(path: str):
    """Простое и стабильное открытие файла локально с выбором бэкенда"""
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
            # Отключаем многопоточность и уменьшаем буфер
            _cap_safe_set(cap, getattr(cv2, 'CAP_PROP_THREADS', 42), 1)
            _cap_safe_set(cap, getattr(cv2, 'CAP_PROP_BUFFERSIZE', 43), 1)

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

def _read_frame_local(cap, t_sec: float = 0.0) -> Optional[np.ndarray]:
    """Более надёжный seek: сначала пробуем POS_FRAMES по индексу, затем POS_MSEC, затем rewind fallback."""
    try:
        if t_sec is None or t_sec < 0:
            t_sec = 0.0
        # Попробуем определить FPS и индекс кадра
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = int(round(max(0.0, float(t_sec)) * max(1.0, float(fps))))
        if total > 0:
            frame_idx = max(0, min(total - 1, frame_idx))

        # Strategy 1: POS_FRAMES
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        except Exception:
            pass

        # Strategy 2: POS_MSEC
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec) * 1000.0))
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        except Exception:
            pass

        # Strategy 3: небольшая перемотка назад и два чтения (для keyframe)
        try:
            rewind = max(0, frame_idx - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, rewind)
            _ = cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        except Exception:
            pass

        return None
    except Exception:
        return None

# --- Camera Manager (single capture per camera with backoff) ---
class CameraStream:
    def __init__(self, cam_id: str, rtsp_url: str):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.opened = False
        self.running = False
        self.last_jpeg: Optional[bytes] = None
        self.last_ts = 0.0
        self.lock = asyncio.Lock()
        self.task: Optional[asyncio.Task] = None
        self.fps_window = []
        self.target_dt = 1.0 / max(1e-3, TARGET_FPS)

        # --- Server-side inference config (single source of truth) ---
        self.model = None
        self.model_loaded = False
        self.frame_idx = 0
        self.frame_skip = FRAME_SKIP
        self.conf_thres = CONF_THRESHOLD
        self.target_class_ids = TARGET_CLASS_IDS
        self.avg_window = AVG_WINDOW
        self.count_window = []  # rolling counts
        self.avg_count = 0.0
        self.last_count = 0

        # статистические агрегаторы для устойчивого отображения
        self.count_window_full = []  # полное окно сырых детекций
        self.stat_count = 0.0        # устойчивое значение (медиана/усреднение)
        self.ema_count = None        # экспоненциальное сглаживание
        self.ema_alpha = float(os.getenv("COUNT_EMA_ALPHA", "0.2"))

        # Cached overlay image to avoid per-frame mask re-blending
        self._overlay_image = None
        self._overlay_shape = None

        # balanced counting state
        self.balanced = BALANCED_DEFAULT

        # auto-calibration state
        self.calib_window = []
        self.balance_alpha = BALANCE_ALPHA
        self.balance_bias = BALANCE_BIAS

        # простой трекер без внешних зависимостей
        self.tracker = SimpleTracker(iou_threshold=float(os.getenv("TRACK_IOU", "0.3")),
                                     max_age=int(os.getenv("TRACK_MAX_AGE", "30")))

        # Serialize any access to underlying VideoCapture to avoid FFmpeg race
        # and disable threaded decoding where possible.
        self.cap_lock = asyncio.Lock()
        # Soft flag to skip model inference if decoding misbehaves
        self.decode_unstable = False

        # Background inference state (decoupled from streaming)
        self._infer_task: Optional[asyncio.Task] = None
        self._infer_running: bool = False
        self._infer_frame = None  # latest BGR frame snapshot for inference
        self._infer_frame_seq = 0
        self._infer_lock = asyncio.Lock()

    def _open(self) -> bool:
        # Открываем RTSP в изолированном процессе
        try:
            ocv_open_rtsp(self.cam_id, self.rtsp_url)
            logger.info("[%s] RTSP opened (worker)", self.cam_id)
        except Exception as e:
            logger.warning("[%s] RTSP open failed: %s", self.cam_id, e)
            return False
        self.opened = True
        # lazy model load on first successful open
        if not self.model_loaded:
            try:
                from ultralytics import YOLO
                # читаем актуальную модель из глобального MODEL_PATH
                model_path = str((MODELS_DIR / MODEL_PATH.split("/")[-1]))
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.model_loaded = True
                    logger.info("[%s] YOLO model loaded: %s", self.cam_id, model_path)
                else:
                    logger.warning("[%s] Model file not found: %s", self.cam_id, model_path)
                    self.model = None
                    self.model_loaded = False
            except Exception as e:
                logger.exception("[%s] Failed to load YOLO model: %s", self.cam_id, e)
                self.model = None
                self.model_loaded = False
        # сброс трекера при переоткрытии
        self.tracker = self.tracker or SimpleTracker(iou_threshold=float(os.getenv("TRACK_IOU", "0.3")),
                                                    max_age=int(os.getenv("TRACK_MAX_AGE", "30")))
        return True

    def _close(self):
        try:
            ocv_close(self.cam_id)
        except Exception:
            pass
        self.opened = False

    async def _infer_loop(self):
        """Run YOLO inference in background on the latest available frame,
        update tracks, counts and cached overlay. Decoupled from streaming FPS."""
        self._infer_running = True
        # Lazy-load the model here to avoid blocking the first frames
        if not self.model_loaded:
            try:
                from ultralytics import YOLO
                model_path = str((MODELS_DIR / MODEL_PATH.split("/")[-1]))
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.model_loaded = True
                    logger.info("[%s] YOLO model loaded (infer loop): %s", self.cam_id, model_path)
            except Exception as e:
                logger.exception("[%s] YOLO model load failed (infer loop): %s", self.cam_id, e)
                self.model = None
                self.model_loaded = False
        while self._infer_running:
            try:
                # Take snapshot
                frame = None
                async with self._infer_lock:
                    if self._infer_frame is not None:
                        frame = self._infer_frame.copy()
                        self._infer_frame = None
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                if not self.model_loaded or frame is None:
                    await asyncio.sleep(0.005)
                    continue
                # Predict
                results = self.model.predict(
                    frame,
                    imgsz=640,
                    conf=self.conf_thres,
                    verbose=False,
                    retina_masks=True
                )
                r = results[0] if results else None
                det_bboxes = []
                det_idx_map = []
                if r is not None and hasattr(r, "boxes") and r.boxes is not None:
                    xyxy = r.boxes.xyxy
                    cls = r.boxes.cls
                    conf = r.boxes.conf
                    if hasattr(xyxy, "cpu"): xyxy = xyxy.cpu().numpy()
                    if hasattr(cls, "cpu"): cls = cls.cpu().numpy()
                    if hasattr(conf, "cpu"): conf = conf.cpu().numpy()
                    for i, b in enumerate(xyxy):
                        c = int(cls[i]) if i < len(cls) else -1
                        cf = float(conf[i]) if i < len(conf) else 0.0
                        if (c in self.target_class_ids) and (cf >= self.conf_thres):
                            x1, y1, x2, y2 = map(float, b)
                            det_bboxes.append([x1, y1, x2, y2])
                            det_idx_map.append(i)
                tracks = self.tracker.update([det_bboxes[k] + [det_idx_map[k]] for k in range(len(det_bboxes))]) if det_bboxes else []
                det_count = len(tracks)

                # build cached overlay once per inference
                try:
                    h, w = frame.shape[:2]
                    overlay = np.zeros_like(frame)
                    mask_data = getattr(r, 'masks', None)
                    if mask_data is not None:
                        mask_data = mask_data.data if hasattr(mask_data, 'data') else mask_data
                        if hasattr(mask_data, 'cpu'):
                            mask_data = mask_data.cpu().numpy()
                        for tr in tracks:
                            tid = tr['id']
                            mi = tr.get('det_index', -1)
                            if 0 <= mi < len(mask_data):
                                mask = (mask_data[mi] > 0.5).astype(np.uint8)
                                if mask.shape[:2] != (h, w):
                                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                color = _pastel_color_for_id(int(tid))
                                overlay[mask > 0] = color
                    else:
                        # Fallback: if no masks available, draw simple bounding boxes for tracks
                        try:
                            for tr in tracks:
                                tid = tr.get('id')
                                bbox = tr.get('bbox') or tr.get('bbox', None)
                                if bbox and len(bbox) >= 4:
                                    x1, y1, x2, y2 = map(int, bbox[:4])
                                    color = _pastel_color_for_id(int(tid))
                                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
                        except Exception:
                            pass

                    self._overlay_image = overlay
                    self._overlay_shape = (h, w)
                    # Публикацию оверлея в live отключаем: оставляем только внутреннее хранение
                    # для возможной диагностики/метрик. JPEG с оверлеем не пишем в self.last_jpeg.
                    try:
                        _ = overlay  # оставим как артефакт для отладки, но не кодируем в JPEG
                    except Exception:
                        pass
                except Exception:
                    pass

                # update rolling stats
                if self.balanced:
                    det_count = min(BALANCE_CAP, det_count * self.balance_alpha + self.balance_bias)
                self.count_window.append(float(det_count))
                if len(self.count_window) > self.avg_window:
                    self.count_window.pop(0)
                self.avg_count = sum(self.count_window) / max(1, len(self.count_window))
                self.last_count = int(round(self.avg_count))
                # robust stat window + ema
                try:
                    self.count_window_full.append(float(det_count))
                    max_full = max(self.avg_window * 3, 60)
                    if len(self.count_window_full) > max_full:
                        self.count_window_full.pop(0)
                    arr = sorted(self.count_window_full)
                    n = len(arr)
                    if n >= 5:
                        k = max(1, int(n * 0.1))
                        trimmed = arr[k:-k] if (2*k) < n else arr
                        m = trimmed[(len(trimmed)-1)//2]
                    else:
                        m = arr[n//2] if n else 0.0
                    if self.ema_count is None:
                        self.ema_count = float(m)
                    else:
                        self.ema_count = float(self.ema_alpha * m + (1.0 - self.ema_alpha) * self.ema_count)
                    self.stat_count = float(self.ema_count)
                except Exception:
                    self.stat_count = float(det_count)

                # auto-calibration
                try:
                    if self.balanced and AUTO_CALIBRATE:
                        self.calib_window.append(self.last_count)
                        if len(self.calib_window) > CALIB_WINDOW:
                            self.calib_window.pop(0)
                        if len(self.calib_window) >= max(10, CALIB_WINDOW // 3):
                            avg = sum(self.calib_window) / len(self.calib_window)
                            avg = max(0.0, min(BALANCE_CAP, avg))
                            target_mid = (CALIB_TARGET_MIN + CALIB_TARGET_MAX) * 0.5
                            err = target_mid - avg
                            self.balance_bias += CALIB_LR_BIAS * err
                            self.balance_bias = float(max(-50.0, min(50.0, self.balance_bias)))
                            self.balance_alpha += CALIB_LR_ALPHA * (1.0 if err > 0 else -1.0)
                            self.balance_alpha = float(max(0.5, min(1.5, self.balance_alpha)))
                except Exception:
                    pass

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.01)
                continue
        self._infer_running = False

    async def _loop(self):
        self.running = True
        backoff = 0.5
        max_back = 5.0
        while self.running:
            if not self.opened:
                if not self._open():
                    logger.warning("[%s] open failed, retry in %.2fs", self.cam_id, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(max_back, backoff * 2)
                    continue
                backoff = 0.5
            # Читаем JPEG из воркера и публикуем его как есть (минимальная задержка).
            # Декодируем только по необходимости для фонового инференса.
            jpeg = ocv_read_jpeg(self.cam_id, timeout=1.0)
            if not jpeg:
                # mark unstable, and force close-reopen
                self.decode_unstable = True
                self._close()
                logger.warning("[%s] read failed, reconnecting...", self.cam_id)
                await asyncio.sleep(backoff)
                backoff = min(max_back, max(0.5, backoff * 2))
                continue

            now = time.time()
            # publish raw jpeg for minimal-latency streaming (no decode/encode cycle)
            try:
                async with self.lock:
                    # Всегда публикуем сырой JPEG для максимально гладкого стрима
                    self.last_jpeg = jpeg
                    if self.last_ts > 0:
                        inst = 1.0 / max(1e-6, now - self.last_ts)
                        self.fps_window.append(inst)
                        if len(self.fps_window) > 30:
                            self.fps_window.pop(0)
                    self.last_ts = now
            except Exception:
                pass

            # Для инференса декодируем только иногда (frame_skip) — это экономит CPU на основном потоке
            try:
                decode_for_infer = (self.model_loaded and (self.frame_idx % max(1, self.frame_skip) == 0))
                if decode_for_infer:
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame_for_infer = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame_for_infer is not None:
                        async with self._infer_lock:
                            if self._infer_frame is None:
                                self._infer_frame = frame_for_infer.copy()
                                self._infer_frame_seq += 1
            except Exception:
                pass

            # Обновим индекс кадра и состояние decode_unstable как раньше, затем продолжим цикл
            self.frame_idx += 1
            if self.decode_unstable and (self.frame_idx % 15 == 0):
                self.decode_unstable = False
            # Пропускаем дальнейшую отрисовку/кодирование в основном loop — стримим сырой JPEG
            continue

            # provide latest frame to infer loop (non-blocking)
            try:
                if frame is not None:
                    async with self._infer_lock:
                        if self._infer_frame is None:
                            self._infer_frame = frame.copy()
                            self._infer_frame_seq += 1
            except Exception:
                pass

            # throttle
            elapsed = now - self.last_ts
            if elapsed < self.target_dt:
                await asyncio.sleep(self.target_dt - elapsed)
                now = time.time()

            # clear unstable flag occasionally when we pass decode/read stage
            self.frame_idx += 1
            if self.decode_unstable and (self.frame_idx % 15 == 0):
                self.decode_unstable = False

            # draw overlays and HUD
            try:
                h, w = frame.shape[:2]
                y_line = int(h * 0.5)
                cv2.line(frame, (0, y_line), (w, y_line), (0, 180, 255), 2)
                # apply cached overlay if present
                try:
                    if self._overlay_image is not None and self._overlay_image.shape[:2] == (h, w):
                        frame = cv2.addWeighted(self._overlay_image, 0.35, frame, 0.65, 0)
                except Exception:
                    pass
                # HUD: current/stat counts from infer loop
                cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Count(cur): {int(self.last_count)}", (18, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Count(stat): {int(round(getattr(self, 'stat_count', self.last_count)))}", (18, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass

            # Encode with perf timing and log
            t_enc0 = time.time()
            img_bytes = encode_image(frame)
            enc_ms = (time.time() - t_enc0) * 1000.0
            if img_bytes:
                try:
                    perf_logger.info(json.dumps({
                        "phase": "live_frame",
                        "camera": self.cam_id,
                        "encode_ms": float(enc_ms)
                    }, ensure_ascii=False))
                except Exception:
                    pass
                async with self.lock:
                    self.last_jpeg = img_bytes
                    if self.last_ts > 0:
                        inst = 1.0 / max(1e-6, now - self.last_ts)
                        self.fps_window.append(inst)
                        if len(self.fps_window) > 30:
                            self.fps_window.pop(0)
                    self.last_ts = now

        self._close()

    async def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._loop())
        if self._infer_task is None or self._infer_task.done():
            self._infer_task = asyncio.create_task(self._infer_loop())

    async def stop(self):
        self.running = False
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=2.0)
            except Exception:
                pass
            self.task = None
        # stop infer task
        if self._infer_task:
            try:
                self._infer_running = False
                self._infer_task.cancel()
            except Exception:
                pass
            self._infer_task = None
        self._close()

    async def get_last_jpeg(self) -> Optional[bytes]:
        async with self.lock:
            return self.last_jpeg

    def fps(self) -> float:
        if not self.fps_window:
            return 0.0
        return sum(self.fps_window) / len(self.fps_window)

class CameraManager:
    def __init__(self):
        self.cams: Dict[str, CameraStream] = {}

    def get_or_create(self, cam_id: str, rtsp_url: str) -> CameraStream:
        cs = self.cams.get(cam_id)
        if cs is None:
            cs = CameraStream(cam_id, rtsp_url)
            self.cams[cam_id] = cs
            # Запуск будет через BackgroundTasks
        return cs

    def get(self, cam_id: str) -> Optional[CameraStream]:
        return self.cams.get(cam_id)

    async def start_camera(self, cam_id: str):
        """Запускает камеру асинхронно"""
        cs = self.cams.get(cam_id)
        if cs:
            await cs.start()

CAMERAS = CameraManager()
DEFAULT_CAM_ID = "cam1"

def mjpeg_multicast(cam: CameraStream) -> Generator[bytes, None, None]:
    # sync generator, uses last_jpeg snapshot
    while True:
        jpeg = cam.last_jpeg
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield multipart_chunk(jpeg)
        time.sleep(max(0.0, (1.0 / max(1.0, TARGET_FPS)) * 0.5))

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<!doctype html><title>PigWeight</title><h1>static/index.html not found</h1>", status_code=200)

@app.get("/api/video_feed")
async def video_feed(mode: str = Query(default="server", pattern="^(server|client)$"),
               camera: str = Query(default=DEFAULT_CAM_ID)):
    cam = CAMERAS.get_or_create(camera, CAM_URL)
    # Запускаем камеру асинхронно, не блокируя ответ (камера может быть недоступна)
    try:
        asyncio.create_task(CAMERAS.start_camera(camera))
    except Exception:
        pass
    return StreamingResponse(mjpeg_multicast(cam), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

@app.get("/api/snapshot")
async def api_snapshot(camera: str = Query(default=DEFAULT_CAM_ID)):
    """Возвращает последний кадр камеры (JPEG) с оверлеем.
    Если камера не запущена — запускает её. Если кадра ещё нет, ждём короткое время (до ~300мс),
    чтобы не создавать на фронте частые пустые ответы и визуальные паузы.
    """
    try:
        CAMERAS.get_or_create(camera, CAM_URL)
        # Не зависаем, если камера недоступна: ограничим время запуска
        try:
            await asyncio.wait_for(CAMERAS.start_camera(camera), timeout=0.25)
        except Exception:
            pass
    except Exception:
        pass
    jpeg = None
    # подождём немного появления кадра
    t0 = time.time()
    deadline = t0 + 0.30
    while time.time() < deadline and not jpeg:
        try:
            existing = CAMERAS.get(camera)
            if existing is not None:
                jpeg = existing.last_jpeg
        except Exception:
            jpeg = None
        if jpeg:
            break
        await asyncio.sleep(0.02)
    if not jpeg:
        return Response(status_code=204, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    return Response(content=jpeg, media_type="image/jpeg", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })

@app.websocket("/ws/count")
async def ws_count(ws: WebSocket):
    await ws.accept()
    try:
        q = ws.query_params or {}
        sess_id = q.get('id')
        camera = q.get('camera', DEFAULT_CAM_ID)
        if sess_id:
            # файловая сессия: стримим счётчик из _file_sessions[id]
            while True:
                await asyncio.sleep(0.2)
                sess = _file_sessions.get(sess_id)
                if not sess:
                    await ws.send_text(json.dumps({
                        "type": "count_update",
                        "file_id": sess_id,
                        "count": 0,
                        "avg": 0.0
                    }))
                    continue
                await ws.send_text(json.dumps({
                    "type": "count_update",
                    "file_id": sess_id,
                    "count": int(sess.get("last_count", 0) or 0),
                    "avg": float(sess.get("avg_count", 0.0) or 0.0)
                }))
        else:
            # режим камеры (как раньше)
            cam = CAMERAS.get_or_create(camera, CAM_URL)
            while True:
                await asyncio.sleep(0.2)
                await ws.send_text(json.dumps({
                    "type": "count_update",
                    "camera": camera,
                    "count": int(cam.last_count),
                    "fps": round(cam.fps(), 2),
                    "debug": {"avg": cam.avg_count, "classes": getattr(cam, 'last_debug_classes', []), "ids": getattr(cam, 'last_debug_ids', []), "bboxes": getattr(cam, 'last_debug_bboxes', [])}
                }))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        try:
            await ws.close()
        except Exception:
            pass
        logger.exception("WebSocket error: %s", e)

# WebSocket-стрим кадров для файла: latest-only
@app.websocket("/ws/video_file")
async def ws_video_file(ws: WebSocket):
    await ws.accept()
    try:
        q = ws.query_params or {}
        sess_id = q.get('id')
        if not sess_id:
            await ws.close()
            return
        # Запуск фона, если ещё не запущен
        async with _file_sessions_lock:
            sess = _file_sessions.get(sess_id)
            if not sess:
                await ws.close()
                return
            if not sess.get("task") or sess["task"].done():
                sess["task"] = asyncio.create_task(_run_file_play_loop(sess_id))
        # Цикл отправки latest-only
        while True:
            await asyncio.sleep(0.02)
            sess = _file_sessions.get(sess_id)
            if not sess:
                break
            img = sess.get("latest_jpeg")
            if not img:
                continue
            try:
                await ws.send_bytes(img)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("/ws/video_file error: %s", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
# --- Video file endpoints (через изолированный OpenCV воркер) ---

_file_sessions: Dict[str, dict] = {}
# Глобальная блокировка доступа к сессиям файлов и их cap,
# чтобы исключить конкурентное чтение/seek/close из разных потоков ASGI.
_file_sessions_lock = asyncio.Lock()

# Кэш серверной модели для кадров из файлов
FILE_MODEL = None
FILE_MODEL_PATH = ""

# Фоновая петля воспроизведения файла для WS/H TTP источников: обновляет latest_jpeg в сессии
async def _run_file_play_loop(sess_id: str):
    """Latest-only: кадр всегда перезаписывается в sess['latest_jpeg'].
    Оверлеи и счёт обновляются как в HTTP-потоке, инференс по FRAME_SKIP."""
    global FILE_MODEL, FILE_MODEL_PATH, MODEL_PATH
    frame_idx_local = 0
    try:
        while True:
            sess = _file_sessions.get(sess_id)
            if not sess or sess.get("type") != "local":
                break
            cap = sess.get("cap")
            if cap is None:
                await asyncio.sleep(0.02)
                continue

            fps = max(5.0, float(sess.get("fps", 25.0) or 25.0))
            t0 = time.time()

            # Чтение кадра под сессионным lock, чтобы не конфликтовать с seek из /frame
            lock: asyncio.Lock = sess.get("lock")
            async with lock:
                ok, frame = cap.read()
            if not ok or frame is None:
                await asyncio.sleep(max(0.005, 1.0 / fps))
                continue

            do_infer = (frame_idx_local % max(1, FRAME_SKIP) == 0)
            det_count = int(sess.get("last_count", 0) or 0)
            r = None
            tracks = []
            if do_infer:
                try:
                    from ultralytics import YOLO
                    try:
                        target_model_path = str((MODELS_DIR / MODEL_PATH.split("/")[-1]))
                    except Exception:
                        target_model_path = os.getenv("MODEL_PATH", "models/yolo11n-seg.pt")
                    if (FILE_MODEL is None) or (FILE_MODEL_PATH != target_model_path):
                        logger.info("[file_ws] Loading YOLO model: %s", target_model_path)
                        FILE_MODEL = YOLO(target_model_path)
                        FILE_MODEL_PATH = target_model_path
                    results = FILE_MODEL.predict(
                        frame,
                        imgsz=640,
                        conf=CONF_THRESHOLD,
                        verbose=False,
                        retina_masks=True
                    )
                    if results and len(results) > 0:
                        r = results[0]
                        det_bboxes = []
                        det_idx_map = []
                        if hasattr(r, "boxes") and r.boxes is not None:
                            xyxy = r.boxes.xyxy
                            cls = r.boxes.cls
                            conf = r.boxes.conf
                            if hasattr(xyxy, "cpu"): xyxy = xyxy.cpu().numpy()
                            if hasattr(cls, "cpu"): cls = cls.cpu().numpy()
                            if hasattr(conf, "cpu"): conf = conf.cpu().numpy()
                            for i, b in enumerate(xyxy):
                                c = int(cls[i]) if i < len(cls) else -1
                                cf = float(conf[i]) if i < len(conf) else 0.0
                                if (c in TARGET_CLASS_IDS) and (cf >= CONF_THRESHOLD):
                                    x1, y1, x2, y2 = map(float, b)
                                    det_bboxes.append([x1, y1, x2, y2])
                                    det_idx_map.append(i)
                        tracker = sess.get("tracker")
                        tracks = tracker.update([det_bboxes[k] + [det_idx_map[k]] for k in range(len(det_bboxes))]) if tracker else []
                        det_count = len(tracks)
                except Exception:
                    pass

            # Рисуем маски по последнему результату (или актуальному)
            try:
                mask_data = None
                if r is not None and hasattr(r, "masks") and r.masks is not None:
                    mask_data = r.masks.data
                    if hasattr(mask_data, "cpu"):
                        mask_data = mask_data.cpu().numpy()
                    sess["_last_masks"] = r.masks
                    sess["_last_tracks"] = tracks
                else:
                    # использовать предыдущее
                    lm = sess.get("_last_masks")
                    if lm is not None:
                        mask_data = lm.data if hasattr(lm, 'data') else lm
                        if hasattr(mask_data, 'cpu'):
                            mask_data = mask_data.cpu().numpy()
                    tracks = sess.get("_last_tracks") or []
                if mask_data is not None and tracks:
                    try:
                        h, w = frame.shape[:2]
                        overlay = np.zeros_like(frame)
                        label_pts = []
                        for tr in tracks:
                            tid = tr['id']
                            mi = tr.get('det_index', -1)
                            if 0 <= mi < len(mask_data):
                                mask = (mask_data[mi] > 0.5).astype(np.uint8)
                                if mask.shape[:2] != (h, w):
                                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                color = _pastel_color_for_id(int(tid))
                                overlay[mask > 0] = color
                                # центроид для нумерации
                                try:
                                    mu = cv2.moments((mask*255).astype(np.uint8))
                                    if mu['m00'] > 1e-3:
                                        cx = int(mu['m10']/mu['m00'])
                                        cy = int(mu['m01']/mu['m00'])
                                        label_pts.append((cx, cy))
                                except Exception:
                                    pass
                        sess["overlay_image"] = overlay
                        sess["overlay_shape"] = (h, w)
                        sess["overlay_labels"] = label_pts
                        try:
                            perf_logger.info(json.dumps({"phase":"overlay_cached","id": sess_id, "h": int(h), "w": int(w)}, ensure_ascii=False))
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

            # Применяем кэшированный оверлей (если есть)
            try:
                h, w = frame.shape[:2]
                ov = sess.get("overlay_image")
                sh = sess.get("overlay_shape")
                if ov is not None and sh == (h, w):
                    # Смягчаем края маски лёгким блюром перед смешиванием
                    try:
                        ovb = cv2.GaussianBlur(ov, (11, 11), 0)
                    except Exception:
                        ovb = ov
                    frame = cv2.addWeighted(ovb, 0.35, frame, 0.65, 0)
                    # Рисуем номера слева-направо
                    try:
                        pts = list(sess.get("overlay_labels") or [])
                        pts.sort(key=lambda p: p[0])
                        for idx, (cx, cy) in enumerate(pts, start=1):
                            label = str(idx)
                            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                    except Exception:
                        pass
            except Exception:
                pass

            # HUD и статистики
            try:
                sess["last_count"] = int(det_count)
                cw = sess.setdefault("count_window", [])
                cw.append(float(det_count))
                if len(cw) > AVG_WINDOW:
                    cw.pop(0)
                # Средневзвешенное в сторону максимума: квадратично-взвешенное среднее
                if cw:
                    s1 = sum(cw)
                    s2 = sum(x*x for x in cw)
                    sess["avg_count"] = (s2 / s1) if s1 > 0 else 0.0
                else:
                    sess["avg_count"] = 0.0
                # HUD
                cv2.rectangle(frame, (10, 10), (320, 80), (0, 0, 0), -1)
                cur_cnt = int(sess.get('last_count', 0) or 0)
                avg_val = float(sess.get('avg_count', 0.0) or 0.0)
                # ceil без импорта math
                avg_int = int(avg_val)
                if avg_val > avg_int:
                    avg_int += 1
                cv2.putText(frame, f"Count(cur): {cur_cnt}", (18, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Count(avg): {avg_int}", (18, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

            # Кодирование и публикация latest-only
            try:
                img = encode_jpeg(frame)
                if img:
                    sess["latest_jpeg"] = img
            except Exception:
                pass

            # pacing
            spent = time.time() - t0
            target = max(0.005, (1.0 / fps))
            await asyncio.sleep(max(0.0, target - spent))
            frame_idx_local += 1
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("[file_ws] play loop error")

@app.post("/api/video_file/open")
async def api_video_file_open(camera: str = Form(default="cam_file1"),
                              id: str = Form(default="file1"),
                              file: UploadFile = File(...)):
    # Закрыть предыдущую сессию с тем же id (если была), чтобы освободить cap и файловые блокировки
    async with _file_sessions_lock:
        old = _file_sessions.pop(id, None)
        if old:
            try:
                cap_old = old.get("cap")
                if cap_old is not None:
                    cap_old.release()
            except Exception:
                pass
            try:
                if old.get("type") == "opencv":
                    get_ocv().close(id)
            except Exception:
                pass
    try:
        # Кеширование по исходному имени файла в каталоге uploads (без раздувания)
        # 1) Санитизируем имя: оставляем буквы/цифры/._- и ограничиваем длину
        raw_name = file.filename or "upload.bin"
        raw_name = os.path.basename(raw_name)
        p = Path(raw_name)
        stem, suffix = (p.stem or "upload"), (p.suffix or ".bin")
        # Санитация stem
        import re as _re
        safe_stem = _re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:120] or "upload"
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        safe_name = f"{safe_stem}{safe_suffix}"
        dst = UPLOAD_DIR / safe_name

        # 2) Если файл уже существует — переиспользуем (кеш)
        if dst.exists():
            try:
                await file.close()
            except Exception:
                pass
            logger.info("[file_open] reuse cached file %s", dst)
        else:
            # 3) Пишем во временный файл и атомарно переименовываем
            tmp_path = dst.with_suffix(dst.suffix + ".part")
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            try:
                await file.close()
            except Exception:
                pass
            os.replace(tmp_path, dst)
            try:
                sz = os.path.getsize(dst)
            except Exception:
                sz = -1
            logger.info("[file_open] saved to %s, size=%s bytes", dst, sz)
    except PermissionError as e:
        return JSONResponse({"error": f"upload failed: {e}"}, status_code=500)
    except Exception as e:
        # Всегда JSON-ответ об ошибке загрузки
        return JSONResponse({"error": f"upload failed: {e}"}, status_code=500)

    # Открываем файл локально (просто и стабильно)
    try:
        logger.info("[file_open] local opening id=%s path=%s", id, dst)
        cap, meta_local = _open_file_cap_local(str(dst))
        if cap is None:
            return JSONResponse({"error": f"local open failed: {meta_local.get('error', 'unknown')}"}, status_code=500)
        logger.info("[file_open] local opened id=%s meta=%s", id, meta_local)

        # Регистрируем локальную сессию
        async with _file_sessions_lock:
            _file_sessions[id] = {
                "type": "local",
                "cap": cap,
                "path": str(dst),
                "camera": camera,
                "fps": float(meta_local.get("fps", 25.0)),
                "frame_count": int(meta_local.get("frame_count", 0)),
                "duration": float(meta_local.get("duration", 0.0)),
                "balanced": BALANCED_DEFAULT,
                "tracker": SimpleTracker(iou_threshold=float(os.getenv("TRACK_IOU", "0.3")),
                                          max_age=int(os.getenv("TRACK_MAX_AGE", "30"))),
                "lock": asyncio.Lock(),
                "seq": 0,
            }

        # Перф-лог: завершили open
        try:
            perf_logger.info(json.dumps({
                "phase": "file_open_done",
                "id": id,
                "path": str(dst),
                "fps": _file_sessions[id]["fps"],
                "frame_count": _file_sessions[id]["frame_count"],
                "duration": _file_sessions[id]["duration"]
            }, ensure_ascii=False))
        except Exception:
            pass

        # Запускаем фоновую play-loop, чтобы /play и WS сразу начали выдавать latest_jpeg (маски)
        try:
            sess = _file_sessions.get(id)
            if sess is not None:
                if not sess.get("task") or sess["task"].done():
                    sess["task"] = asyncio.create_task(_run_file_play_loop(id))
        except Exception:
            pass

        return JSONResponse({
            "id": id,
            "camera": camera,
            "path": str(dst),
            "fps": _file_sessions[id]["fps"],
            "frame_count": _file_sessions[id]["frame_count"],
            "duration": _file_sessions[id]["duration"],
            "balanced": BALANCED_DEFAULT,
            "backend": "local"
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"file open failed: {e}"}, status_code=500)

@app.get("/api/video_file/close")
async def api_video_file_close(id: str = Query(default="file1")):
    async with _file_sessions_lock:
        sess = _file_sessions.pop(id, None)
        if not sess:
            return {"status": "noop"}
        try:
            cap = sess.get("cap")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if sess.get("type") == "opencv":
                with contextlib.suppress(Exception):
                    get_ocv().close(id)
        finally:
            return {"status": "closed", "id": id}

@app.post("/api/video_file/balance_mode")
async def api_video_file_balance_mode(id: str = Query(default="file1"),
                                      balanced: bool = Query(default=True),
                                      reset_calibration: bool = Query(default=False)):
    async with _file_sessions_lock:
        sess = _file_sessions.get(id)
        if not sess:
            return JSONResponse({"error": "file session not opened"}, status_code=400)
        sess["balanced"] = bool(balanced)
        # зарезервировано: если в будущем добавим калибровку для файлов
        if reset_calibration:
            sess["calib_window"] = []
            sess["balance_alpha"] = BALANCE_ALPHA
            sess["balance_bias"] = BALANCE_BIAS
        return {
            "id": id,
            "balanced": sess["balanced"],
            "alpha": float(sess.get("balance_alpha", BALANCE_ALPHA)),
            "bias": float(sess.get("balance_bias", BALANCE_BIAS)),
        }

@app.get("/api/models")
def api_models():
    """
    Возвращает список доступных .pt в каталоге ./models (только имена файлов).
    """
    try:
        if not MODELS_DIR.exists():
            return {"models": []}
        items = [p.name for p in MODELS_DIR.iterdir() if p.is_file() and p.suffix.lower()==".pt"]
        return {"models": items}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/video_file/snapshot")
async def api_video_file_snapshot(id: str = Query(default="file1")):
    """Возвращает последний кадр файловой сессии (JPEG) без плеера.
    Если latest_jpeg ещё не готов, пытается прочитать один кадр из cap и закодировать его.
    """
    try:
        async with _file_sessions_lock:
            sess = _file_sessions.get(id)
            if not sess:
                return JSONResponse({"error": "session not found"}, status_code=404)
            jpeg = sess.get("latest_jpeg")
            cap = sess.get("cap")
        if not jpeg and cap is not None:
            # Пытаемся синхронно прочитать один кадр
            ok, frame = cap.read()
            if ok and frame is not None:
                jpeg = encode_jpeg(frame)
                # Кэшируем как latest
                try:
                    async with _file_sessions_lock:
                        sess = _file_sessions.get(id)
                        if sess is not None:
                            sess["latest_jpeg"] = jpeg
                except Exception:
                    pass
        if not jpeg:
            return Response(status_code=204)
        return Response(content=jpeg, media_type="image/jpeg")
    except Exception as e:
        logger.exception("/api/video_file/snapshot error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/balance_mode")
def api_get_balance_mode(camera: str = Query(default=DEFAULT_CAM_ID)):
    try:
        cs = CAMERAS.get_or_create(camera, CAM_URL)
        return {"camera": camera, "balanced": bool(getattr(cs, "balanced", False))}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/balance_mode")
async def api_set_balance_mode(camera: str = Query(default=DEFAULT_CAM_ID),
                               balanced: bool = Query(default=True),
                               reset_calibration: bool = Query(default=False)):
    try:
        cs = CAMERAS.get_or_create(camera, CAM_URL)
        cs.balanced = bool(balanced)
        if reset_calibration:
            cs.calib_window.clear()
            cs.balance_alpha = BALANCE_ALPHA
            cs.balance_bias = BALANCE_BIAS
        return {
            "camera": camera,
            "balanced": cs.balanced,
            "alpha": getattr(cs, "balance_alpha", BALANCE_ALPHA),
            "bias": getattr(cs, "balance_bias", BALANCE_BIAS),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/video_config")
def api_video_config(camera: str = Query(default=DEFAULT_CAM_ID),
                     seg_model_path: str = Query(default="")):
    """
    Применение серверной модели:
    - Принимает относительный path внутри ./models (например, 'models/yolo11n.pt' или 'yolo11n.pt')
    - Обновляет глобальный MODEL_PATH для последующих загрузок.
    """
    global MODEL_PATH
    try:
        # нормализуем путь: разрешаем как 'models/xxx.pt', так и 'xxx.pt'
        fname = seg_model_path.replace("\\", "/").split("/")[-1]
        candidate = MODELS_DIR / fname
        if not candidate.exists():
            return JSONResponse({"error": f"Model not found: {candidate}"}, status_code=404)
        MODEL_PATH = f"models/{fname}"
        logger.info("MODEL_PATH set to %s", MODEL_PATH)
        return {"status": "ok", "model": MODEL_PATH, "camera": camera}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/video_file/frame")
async def api_video_file_frame(id: str = Query(default="file1"),
                         camera: str = Query(default="cam_file1"),
                         t: float = Query(default=0.0),
                         req: int = Query(default=0)):
    """
    Безопасное извлечение одиночного кадра из загруженного видеофайла с защитой от гонок,
    валидацией параметров и подробными логами. Исключения не роняют процесс — всегда JSON/JPEG.
    """
    try:
        # 1) Чтение и базовая валидация под глобальной блокировкой сессий
        async with _file_sessions_lock:
            sess = _file_sessions.get(id)
            if not sess:
                logger.warning("[file_frame] no session id=%s", id)
                return JSONResponse({"error": "file session not opened"}, status_code=400)

            duration = float(sess.get("duration", 0.0) or 0.0)
            fps = float(sess.get("fps", 0.0) or 0.0)
            backend_type = sess.get("type", "opencv")

            # Ограничим t в допустимый диапазон, чтобы не уходить за края
            if t < 0:
                t = 0.0
            if duration > 0 and t > duration:
                t = max(0.0, duration - (1.0 / max(fps, 25.0)))

            logger.info("[file_frame] id=%s camera=%s t=%.3f dur=%.3f fps=%.3f backend=%s",
                        id, camera, t, duration, fps, backend_type)

            frame = None

            # 2) Извлечение кадра через локальный OpenCV (просто и стабильно)
            sess = _file_sessions.get(id)
            if not sess or sess.get("type") != "local" or sess.get("cap") is None:
                return JSONResponse({"error": "file session not opened or invalid"}, status_code=400)

            # отмена устаревших запросов: увеличиваем seq и сравниваем
            lock: asyncio.Lock = sess.get("lock")
            async with lock:
                sess["seq"] = int(sess.get("seq", 0)) + 1
                cur_seq = sess["seq"]
                cap = sess.get("cap")
                t_seek0 = time.time()
                frame = _read_frame_local(cap, t)
                seek_ms = (time.time() - t_seek0) * 1000.0
                sess["last_seek_ms"] = float(seek_ms)
                # если за время seek пришёл новый запрос — прерываемся
                if cur_seq != sess.get("seq"):
                    return JSONResponse({"error": "superseded"}, status_code=409)
            if frame is None:
                return JSONResponse({"error": "Failed to read frame (local)"}, status_code=500)

            # 3) Инференс и отрисовка оверлеев для одиночного запроса (чтобы seek отдавал маски)
            global FILE_MODEL, FILE_MODEL_PATH, MODEL_PATH
            det_count = int(sess.get("last_count", 0) or 0)
            r = None
            tracks = []
            try:
                from ultralytics import YOLO
                try:
                    target_model_path = str((MODELS_DIR / MODEL_PATH.split("/")[-1]))
                except Exception:
                    target_model_path = os.getenv("MODEL_PATH", "models/yolo11n-seg.pt")
                if (FILE_MODEL is None) or (FILE_MODEL_PATH != target_model_path):
                    logger.info("[file_frame] Loading YOLO model: %s", target_model_path)
                    FILE_MODEL = YOLO(target_model_path)
                    FILE_MODEL_PATH = target_model_path

                t_inf0 = time.time()
                results = FILE_MODEL.predict(
                    frame,
                    imgsz=640,
                    conf=CONF_THRESHOLD,
                    verbose=False,
                    retina_masks=True
                )
                infer_ms = (time.time() - t_inf0) * 1000.0
                sess["last_infer_ms"] = float(infer_ms)
                if results and len(results) > 0:
                    r = results[0]
                    det_bboxes = []
                    det_idx_map = []
                    if hasattr(r, "boxes") and r.boxes is not None:
                        xyxy = r.boxes.xyxy
                        cls = r.boxes.cls
                        conf = r.boxes.conf
                        if hasattr(xyxy, "cpu"): xyxy = xyxy.cpu().numpy()
                        if hasattr(cls, "cpu"): cls = cls.cpu().numpy()
                        if hasattr(conf, "cpu"): conf = conf.cpu().numpy()
                        for i, b in enumerate(xyxy):
                            c = int(cls[i]) if i < len(cls) else -1
                            cf = float(conf[i]) if i < len(conf) else 0.0
                            if (c in TARGET_CLASS_IDS) and (cf >= CONF_THRESHOLD):
                                x1, y1, x2, y2 = map(float, b)
                                det_bboxes.append([x1, y1, x2, y2])
                                det_idx_map.append(i)
                    tracker = sess.get("tracker")
                    tracks = tracker.update([det_bboxes[k] + [det_idx_map[k]] for k in range(len(det_bboxes))]) if tracker else []
                    det_count = len(tracks)
            except Exception as e:
                logger.debug("[file_frame] infer failed: %s", e)
                infer_ms = float(sess.get("last_infer_ms", 0.0))

            # Отрисовка масок/оверлея по результатам инференса (или по кэшу)
            try:
                mask_data = None
                if r is not None and hasattr(r, "masks") and r.masks is not None:
                    mask_data = r.masks.data
                    if hasattr(mask_data, "cpu"):
                        mask_data = mask_data.cpu().numpy()
                    sess["_last_masks"] = r.masks
                    sess["_last_tracks"] = tracks
                else:
                    lm = sess.get("_last_masks")
                    if lm is not None:
                        mask_data = lm.data if hasattr(lm, 'data') else lm
                        if hasattr(mask_data, 'cpu'):
                            mask_data = mask_data.cpu().numpy()
                    tracks = sess.get("_last_tracks") or []

                if mask_data is not None and tracks:
                    try:
                        h, w = frame.shape[:2]
                        overlay = np.zeros_like(frame)
                        for tr in tracks:
                            tid = tr['id']
                            mi = tr.get('det_index', -1)
                            if 0 <= mi < len(mask_data):
                                mask = (mask_data[mi] > 0.5).astype(np.uint8)
                                if mask.shape[:2] != (h, w):
                                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                color = _pastel_color_for_id(int(tid))
                                overlay[mask > 0] = color
                        # Наложение оверлея непосредственно на кадр, чтобы вернуть пользователю видимые маски
                        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                        sess["overlay_image"] = overlay
                        sess["overlay_shape"] = (h, w)
                        try:
                            perf_logger.info(json.dumps({"phase":"overlay_cached_file_frame","file_id": id, "h": int(h), "w": int(w)}, ensure_ascii=False))
                        except Exception:
                            pass
                    except Exception:
                        pass

            except Exception:
                pass

            # update rolling avg/stat
            try:
                sess["last_count"] = int(det_count)
                cw = sess.setdefault("count_window", [])
                cw.append(float(det_count))
                if len(cw) > AVG_WINDOW:
                    cw.pop(0)
                sess["avg_count"] = (sum(cw) / len(cw)) if cw else 0.0
            except Exception:
                pass

    except Exception as e:
        logger.exception("[file_frame] unexpected pre-infer error: %s", e)
        return JSONResponse({"error": f"internal error: {e}"}, status_code=500)

    # 4) Быстрая валидация кадра
    try:
        if frame is None:
            return JSONResponse({"error": "Empty frame"}, status_code=500)
        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            return JSONResponse({"error": "Invalid frame shape"}, status_code=500)
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return JSONResponse({"error": "Zero-size frame"}, status_code=500)
        if frame.dtype != np.uint8:
            try:
                frame = frame.astype(np.uint8)
            except Exception:
                return JSONResponse({"error": "Unsupported frame dtype"}, status_code=500)
    except Exception as e:
        logger.exception("[file_frame] frame validation error: %s", e)
        return JSONResponse({"error": f"frame validation error: {e}"}, status_code=500)

    # 5) JPEG кодирование (заменено BMP на JPEG — экономия трафика и совместимость)
    try:
        t_enc0 = time.time()
        img_bytes = encode_jpeg(frame)
        if not img_bytes:
            logger.error("[file_frame] jpeg encode failed")
            return JSONResponse({"error": "Failed to encode frame"}, status_code=500)
        enc_ms = (time.time() - t_enc0) * 1000.0
        try:
            sess = _file_sessions.get(id)
            if sess is not None:
                sess["last_encode_ms"] = float(enc_ms)
                total_ms = float(sess.get("last_seek_ms", 0.0)) + float(sess.get("last_infer_ms", 0.0)) + float(enc_ms)
                sess["last_total_ms"] = total_ms
        except Exception:
            pass
        headers = {
            "X-Seek-Ms": str(int(round(sess.get("last_seek_ms", 0.0) if 'sess' in locals() and sess else 0))),
            "X-Infer-Ms": str(int(round(sess.get("last_infer_ms", 0.0) if 'sess' in locals() and sess else 0))),
            "X-Encode-Ms": str(int(round(enc_ms)))
        }
        try:
            # обновим latest_jpeg чтобы /play и WS сразу увидели результат seek
            sess_update = _file_sessions.get(id)
            if sess_update is not None:
                sess_update["latest_jpeg"] = img_bytes
        except Exception:
            pass
        return Response(img_bytes, media_type="image/jpeg", headers=headers)
    except Exception as e:
        logger.exception("[file_frame] encoding error: %s", e)
        return JSONResponse({"error": f"Failed to encode frame: {e}"}, status_code=500)

def _gen_file_mjpeg_minimal(sess_id: str, rate: float):
    """Минималистичный MJPEG-генератор для файлов — отдаёт sess['latest_jpeg'] без инференса/оверлеев."""
    import time as _time
    while True:
        sess = _file_sessions.get(sess_id)
        if not sess or sess.get("type") != "local":
            break
        img = sess.get("latest_jpeg")
        if not img:
            _time.sleep(0.02)
            continue
        try:
            yield multipart_chunk(img)
        except GeneratorExit:
            break
        except Exception:
            _time.sleep(0.02)
            continue
        # небольшой sleep, чтобы клиентский браузер получал обновления с разумной частотой
        try:
            _time.sleep(max(0.05, 1.0 / max(1.0, float(TARGET_FPS))))
        except Exception:
            _time.sleep(0.05)
            continue
def _gen_file_mjpeg(sess_id: str, rate: float):
    """
    Простой генератор MJPEG для проигрывания файла через локальный OpenCV
    с непрерывным серверным инференсом, оверлеями и усреднением счёта.
    """
    import time as _time
    global FILE_MODEL, FILE_MODEL_PATH, MODEL_PATH
    frame_idx_local = 0
    while True:
        try:
            sess = _file_sessions.get(sess_id)
            if not sess or sess.get("type") != "local":
                break
            
            cap = sess.get("cap")
            if cap is None:
                _time.sleep(0.02)
                continue

            fps = max(5.0, float(sess.get("fps", 25.0) or 25.0))
            t0 = _time.time()
            # измерим стадии: seek (для play это read), infer, encode, total
            t_seek0 = _time.time()
            frame = _read_frame_local(cap)
            seek_ms = (time.time() - t_seek0) * 1000.0
            if frame is None:
                _time.sleep(max(0.005, 1.0 / fps))
                continue

            do_infer = (frame_idx_local % max(1, FRAME_SKIP) == 0)
            det_count = int(sess.get("last_count", 0) or 0)
            if do_infer:
                # ensure model
                from ultralytics import YOLO
                try:
                    target_model_path = str((MODELS_DIR / MODEL_PATH.split("/")[-1]))
                except Exception:
                    target_model_path = os.getenv("MODEL_PATH", "models/yolo11n-seg.pt")
                if (FILE_MODEL is None) or (FILE_MODEL_PATH != target_model_path):
                    logger.info("[file_play] Loading YOLO model: %s", target_model_path)
                    FILE_MODEL = YOLO(target_model_path)
                    FILE_MODEL_PATH = target_model_path
                # predict
                t_inf0 = time.time()
                results = FILE_MODEL.predict(
                    frame,
                    imgsz=640,
                    conf=CONF_THRESHOLD,
                    verbose=False,
                    retina_masks=True
                )
                infer_ms = (time.time() - t_inf0) * 1000.0
                if results and len(results) > 0:
                    r = results[0]
                    det_bboxes = []
                    det_idx_map = []
                    if hasattr(r, "boxes") and r.boxes is not None:
                        xyxy = r.boxes.xyxy
                        cls = r.boxes.cls
                        conf = r.boxes.conf
                        if hasattr(xyxy, "cpu"):
                            xyxy = xyxy.cpu().numpy()
                        if hasattr(cls, "cpu"):
                            cls = cls.cpu().numpy()
                        if hasattr(conf, "cpu"):
                            conf = conf.cpu().numpy()
                        for i, b in enumerate(xyxy):
                            c = int(cls[i]) if i < len(cls) else -1
                            cf = float(conf[i]) if i < len(conf) else 0.0
                            if (c in TARGET_CLASS_IDS) and (cf >= CONF_THRESHOLD):
                                x1, y1, x2, y2 = map(float, b)
                                det_bboxes.append([x1, y1, x2, y2])
                                det_idx_map.append(i)
                tracker = sess.get("tracker")
                tracks = tracker.update([det_bboxes[k] + [det_idx_map[k]] for k in range(len(det_bboxes))]) if tracker else []
                det_count = len(tracks)
                # draw masks
                mask_data = None
                if hasattr(r, "masks") and r.masks is not None:
                    mask_data = r.masks.data
                    if hasattr(mask_data, "cpu"):
                        mask_data = mask_data.cpu().numpy()
                    h, w = frame.shape[:2]
                    overlay = np.zeros_like(frame)
                    for tr in tracks:
                        tid = tr['id']
                        mi = tr.get('det_index', -1)
                        if 0 <= mi < len(mask_data):
                            mask = (mask_data[mi] > 0.5).astype(np.uint8)
                            if mask.shape[:2] != (h, w):
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            color = _pastel_color_for_id(int(tid))
                            overlay[mask > 0] = color
                    sess["overlay_image"] = overlay
                    sess["overlay_shape"] = (h, w)
                    # optional ID labels kept minimal to reduce cost
                    try:
                        ys, xs = np.where(mask > 0)
                        if len(xs) > 0:
                            cx, cy = int(xs.mean()), int(ys.mean())
                            cv2.putText(frame, str(int(tid)), (max(0, cx-10), max(12, cy-8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                # cache
                sess["_last_tracks"] = tracks if do_infer else sess.get("_last_tracks")
                sess["_last_masks"] = getattr(r, 'masks', None) if do_infer else sess.get("_last_masks")
                # update rolling avg
                sess["last_count"] = int(det_count)
                cw = sess.setdefault("count_window", [])
                cw.append(float(det_count))
                if len(cw) > AVG_WINDOW:
                    cw.pop(0)
                sess["avg_count"] = (sum(cw) / len(cw)) if cw else 0.0
            else:
                # overlay last
                mask_data = sess.get("_last_masks")
                if mask_data is not None:
                    try:
                        if hasattr(mask_data, 'cpu'):
                            mask_data = mask_data.cpu().numpy()
                        tracks = sess.get("_last_tracks") or []
                        h, w = frame.shape[:2]
                        overlay = np.zeros_like(frame)
                        for tr in tracks:
                            tid = tr['id']
                            mi = tr.get('det_index', -1)
                            if 0 <= mi < len(mask_data):
                                mask = (mask_data[mi] > 0.5).astype(np.uint8)
                                if mask.shape[:2] != (h, w):
                                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                color = _pastel_color_for_id(int(tid))
                                overlay[mask > 0] = color
                        sess["overlay_image"] = overlay
                        sess["overlay_shape"] = (h, w)
                    except Exception:
                        pass
            # HUD: стадии обработки в секундах
            try:
                seek_s = (float(sess.get('last_seek_ms', 0.0)))/1000.0
                infer_s = (float(sess.get('last_infer_ms', 0.0)))/1000.0
                enc_s = (float(sess.get('last_encode_ms', 0.0)))/1000.0
                tot_s = (float(sess.get('last_total_ms', 0.0)))/1000.0
            except Exception:
                seek_s = infer_s = enc_s = tot_s = 0.0
            cv2.rectangle(frame, (10, 10), (360, 110), (0, 0, 0), -1)
            cv2.putText(frame, f"Count(cur): {int(sess.get('last_count', 0) or 0)}", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Count(avg): {int(round(sess.get('avg_count', 0.0) or 0.0))}", (18, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Seek: {seek_s:.3f}s  Infer: {infer_s:.3f}s  Enc: {enc_s:.3f}s  Tot: {tot_s:.3f}s", (18, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
            
            # Кодирование и отправка кадра
            t_enc0 = time.time()
            img_bytes = encode_jpeg(frame)
            enc_ms = (time.time() - t_enc0) * 1000.0
            
            # Сохраним метрики для HUD
            sess["last_seek_ms"] = seek_ms
            sess["last_infer_ms"] = infer_ms if do_infer else sess.get("last_infer_ms", 0.0)
            sess["last_encode_ms"] = enc_ms
            sess["last_total_ms"] = (time.time() - t0) * 1000.0
            
            # Сохраним последний JPEG для minimal-генератора
            sess["latest_jpeg"] = img_bytes
            
            yield multipart_chunk(img_bytes)
            
            # Пауза для поддержания FPS
            spent = time.time() - t0
            target = max(0.005, (1.0 / fps))
            time.sleep(max(0.0, target - spent))
            frame_idx_local += 1
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("[file_play] error")
        finally:
            # Освобождаем ресурсы при выходе из генератора
            pass

# Override old heavy generator with minimal one (keeps file but ensures minimal behavior is used)
def _gen_file_mjpeg(sess_id: str, rate: float):
    """Alias to minimal generator to avoid heavy per-frame processing."""
    return _gen_file_mjpeg_minimal(sess_id, rate)

@app.get("/api/video_file/play")
async def api_video_file_play(id: str = Query(default="file1"),
                        camera: str = Query(default="cam_file1"),
                        rate: float = Query(default=1.0)):
    # Только проверка существования под блокировкой, генератор остаётся синхронным
    async with _file_sessions_lock:
        if id not in _file_sessions:
            return JSONResponse({"error": "file session not opened"}, status_code=400)
    return StreamingResponse(_gen_file_mjpeg_minimal(id, rate), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

# /api/video_file/last_count removed (WS provides updates)
# --- Shutdown handled by lifespan() ---

# --- Tracking utils ---
def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return float(inter / max(1e-6, union))

class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.tracks = {}  # id -> {bbox: [x1,y1,x2,y2], age: int}
        self.next_id = 1
    
    def update(self, detections: list[list[float]]):
        """
        detections: list of [x1,y1,x2,y2, det_index]
        Returns: list of dict {id, bbox, det_index}
        """
        assigned = set()
        # Compute IoU matrix track_id -> det_index
        matches = []
        for tid, st in self.tracks.items():
            best_iou = 0.0
            best_j = -1
            for j, det in enumerate(detections):
                if j in assigned:
                    continue
                iou = _iou(st['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= self.iou_threshold:
                # assign
                assigned.add(best_j)
                self.tracks[tid]['bbox'] = detections[best_j][:4]
                self.tracks[tid]['age'] = 0
                matches.append({'id': tid, 'bbox': detections[best_j][:4], 'det_index': int(detections[best_j][4])})
            else:
                # age up
                self.tracks[tid]['age'] += 1
        # remove old
        to_del = [tid for tid, st in self.tracks.items() if st['age'] > self.max_age]
        for tid in to_del:
            self.tracks.pop(tid, None)
        # create for unassigned detections
        for j, det in enumerate(detections):
            if j in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {'bbox': det[:4], 'age': 0}
            matches.append({'id': tid, 'bbox': det[:4], 'det_index': int(det[4])})
        return matches

# --- Color utils for overlays ---
def _pastel_color_for_id(track_id: int) -> tuple[int, int, int]:
    if track_id is None or track_id <= 0:
        return (200, 200, 200)
    # Knuth multiplicative hash для равномерного распределения цвета по кругу
    hseed = ((int(track_id) * 2654435761) & 0xFFFFFFFF) / 4294967296.0  # [0,1)
    # Независимая вариация насыщенности из второго хеша
    sseed = ((int(track_id) ^ 0x9E3779B9) * 2246822519 & 0xFFFFFFFF) / 4294967296.0
    h = hseed  # тон по всему кругу
    s = 0.45 + 0.20 * sseed  # 0.45..0.65 — пастель, но с вариацией
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))

# Ультра-быстрые эндпоинты
try:
    from .ultra_fast_endpoints import add_ultra_fast_endpoints
    add_ultra_fast_endpoints(app)
    logger.info("Ultra-fast endpoints added successfully")
except Exception as e:
    logger.warning(f"Failed to add ultra-fast endpoints: {e}")

# Глобально ограничим потоки OpenCV для стабильности
try:
    import cv2 as _cv2_for_threads
    if hasattr(_cv2_for_threads, 'setNumThreads'):
        _cv2_for_threads.setNumThreads(1)
except Exception:
    pass

# Безопасная установка свойств VideoCapture
def _cap_safe_set(cap, prop, value):
    try:
        cap.set(prop, value)
    except Exception:
        return False
    return True

@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    try:
        q = ws.query_params or {}
        sess_id = q.get('id')
        camera = q.get('camera', DEFAULT_CAM_ID)
        if sess_id:
            # file session: latest-only frames
            while True:
                await asyncio.sleep(0.02)
                sess = _file_sessions.get(sess_id)
                if not sess:
                    break
                img = sess.get("latest_jpeg")
                if not img:
                    continue
                try:
                    await ws.send_bytes(img)
                except Exception:
                    break
        else:
            # live camera: latest-only frames from CameraStream
            cam = CAMERAS.get_or_create(camera, CAM_URL)
            await CAMERAS.start_camera(camera)
            while True:
                await asyncio.sleep(0.02)
                try:
                    jpeg = await cam.get_last_jpeg()
                except Exception:
                    jpeg = None
                if not jpeg:
                    continue
                try:
                    await ws.send_bytes(jpeg)
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("/ws/video error: %s", e)
    finally:
        with contextlib.suppress(Exception):
            await ws.close()
