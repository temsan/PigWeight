import os
import cv2
import json
import time
import asyncio
import logging
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

# --- Bootstrap ---
BASE_DIR = Path(__file__).parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
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
                    results = self.model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False, retina_masks=True)
                    r = results[0] if results else None
                    if r and hasattr(r, "masks") and r.masks is not None:
                        self.last_count = len(r.masks.xy)
                        # Normalize masks to 0-1 range
                        h, w, _ = frame.shape
                        self.last_masks = [[(p[0]/w, p[1]/h) for p in m] for m in r.masks.xy]
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
        self.cap: Optional[cv2.VideoCapture] = None
        self.duration = 0.0
        self.fps = 0.0

    async def _stream_loop(self):
        self.cap, meta = _open_file_cap_local(self.file_path)
        if self.cap:
            self.duration = meta.get("duration", 0.0)
            self.fps = meta.get("fps", 25.0)
            while self.running and self.cap:
                ok, frame = self.cap.read()
                if not ok:
                    break
                jpeg = encode_jpeg(frame)
                async with self.lock:
                    self.last_jpeg = jpeg
                await asyncio.sleep(1.0 / self.fps)
        self.running = False

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

app = FastAPI(title="PigWeight API (FastAPI)")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.websocket("/ws/count")
async def ws_count(ws: WebSocket, id: str):
    await ws.accept()
    STREAM_MANAGER.register_websocket(id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        STREAM_MANAGER.unregister_websocket(id, ws)

# ... (the rest of the API endpoints)