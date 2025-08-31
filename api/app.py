import os
import logging
import cv2
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional, Generator, Any, List
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

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RECORDS_DIR = BASE_DIR / "records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# --- Helper Functions ---

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

class SimpleTracker:
    def __init__(self, iou_threshold=0.5, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        # This is a simplified tracker. A real implementation would be more complex.
        # For now, we just return the detections as tracks.
        tracks = []
        for i, det in enumerate(detections):
            tracks.append({**det, 'id': i})
        return tracks

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
        self.last_masks = []
        self.tracker = SimpleTracker()
        self._infer_task: Optional[asyncio.Task] = None
        self._stream_task: Optional[asyncio.Task] = None

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

    async def _infer_loop(self):
        from ultralytics import YOLO
        self.model = YOLO(MODEL_PATH)
        self.model_loaded = True
        while self.running:
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
                    if r and hasattr(r, "masks") and r.masks is not None:
                        self.last_count = len(r.masks.xy)
                        # Normalize masks to original full frame 0-1 range, accounting for vertical crop offset
                        hc, wc = proc.shape[:2]
                        # Map each polygon point (x,y) from cropped coords to original normalized coords
                        mapped: list[list[tuple[float, float]]] = []
                        for m in r.masks.xy:
                            pts = []
                            for p in m:
                                x = float(p[0])
                                y = float(p[1]) + float(y0)
                                pts.append((x / float(W0), y / float(H0)))
                            mapped.append(pts)
                        self.last_masks = mapped
                    else:
                        self.last_count = 0
                        self.last_masks = []
                    await STREAM_MANAGER.broadcast(self.stream_id, {
                        "type": "count_update",
                        "count": self.last_count,
                        "debug": {"masks": self.last_masks}
                    })
            await asyncio.sleep(1)

    async def start(self):
        if not self.running:
            self.running = True
            self._stream_task = asyncio.create_task(self._stream_loop())
            self._infer_task = asyncio.create_task(self._infer_loop())

    async def stop(self):
        if self.running:
            self.running = False
            if self._stream_task:
                self._stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._stream_task
            if self._infer_task:
                self._infer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._infer_task

    @abc.abstractmethod
    async def _stream_loop(self):
        pass

    async def get_jpeg(self) -> Optional[bytes]:
        async with self.lock:
            return self.last_jpeg

class RtspStream(VideoStream):
    def __init__(self, stream_id: str, rtsp_url: str):
        super().__init__(stream_id)
        self.rtsp_url = rtsp_url

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

class FileStream(VideoStream):
    def __init__(self, stream_id: str, file_path: str):
        super().__init__(stream_id)
        self.file_path = file_path
        self.duration = 0.0
        self.fps = 0.0
        self._seek_event = asyncio.Event()
        self._seek_time: Optional[float] = None
        self.current_time = 0.0

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

class StreamManager:
    def __init__(self):
        self.streams: Dict[str, VideoStream] = {}
        self.websockets: Dict[str, List[WebSocket]] = {}

    async def get_or_create_stream(self, stream_id: str, source_uri: str) -> VideoStream:
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._-")
        dst = UPLOAD_DIR / safe_name
        with open(dst, "wb") as buffer:
            buffer.write(await file.read())
        return {"file_path": str(dst)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/stream/start")
async def api_stream_start(stream_id: str, source_uri: str):
    stream = await STREAM_MANAGER.get_or_create_stream(stream_id, source_uri)
    await stream.start()
    return {"status": "started", "stream_id": stream_id}

@app.get("/api/stream/{stream_id}/stop")
async def api_stream_stop(stream_id: str):
    await STREAM_MANAGER.stop_stream(stream_id)
    return {"status": "stopped", "stream_id": stream_id}

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

@app.get("/api/stream/{stream_id}/info")
async def api_stream_info(stream_id: str):
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream:
        return JSONResponse({"error": "stream not found"}, status_code=404)
    if isinstance(stream, FileStream):
        return {
            "type": "file",
            "duration": float(stream.duration or 0.0),
            "current_time": float(stream.current_time or 0.0),
            "fps": float(stream.fps or 0.0),
        }
    else:
        return {"type": "rtsp", "duration": None}

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
