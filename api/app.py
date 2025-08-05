import os
import cv2
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Generator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from norfair import Detection, Tracker, Video, draw_tracked_objects
import numpy as np

# --- Bootstrap ---
BASE_DIR = Path(__file__).parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="PigWeight API (FastAPI)")

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

# --- Config ---
CAM_DEFAULT = os.getenv("CAM_DEFAULT", "rtsp://admin:Qwerty.123@10.15.6.24/1/1")
CAM_URL = os.getenv("CAM_URL", CAM_DEFAULT)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
TARGET_FPS = float(os.getenv("FPS", "12"))
BOUNDARY = "frame"

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def encode_jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
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

# --- Camera Manager (single capture per camera with backoff) ---
class CameraStream:
    def __init__(self, cam_id: str, rtsp_url: str):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.last_jpeg: Optional[bytes] = None
        self.last_ts = 0.0
        self.lock = asyncio.Lock()
        self.task: Optional[asyncio.Task] = None
        self.fps_window = []
        self.target_dt = 1.0 / max(1e-3, TARGET_FPS)

        # --- Server-side inference config (cow counting) ---
        self.model = None
        self.model_loaded = False
        self.frame_idx = 0
        # Инференс с пропуском промежуточных кадров — только на "крайнем" кадре из окна:
        # реализуем как запуск инференса раз в frame_skip кадров
        self.frame_skip = 3
        self.target_class_ids = {20, 17, 19}  # cow, sheep, horse
        self.conf_thres = 0.30
        self.avg_window = 20
        self.count_window = []  # последние N детектов для усреднения
        self.avg_count = 0.0
        self.last_count = 0

        # --- Server inference config ---
        self.model = None
        self.model_loaded = False
        self.frame_idx = 0
        self.frame_skip = 3  # as requested
        self.conf_thres = 0.40
        self.target_class_ids = {18}  # pig
        self.avg_window = 10
        self.count_window = []  # rolling counts
        self.avg_count = 0.0
        self.last_count = 0

        self.tracker = None

    def _open(self) -> bool:
        cap = cv2.VideoCapture(self.rtsp_url)
        for prop, val in ((cv2.CAP_PROP_BUFFERSIZE, 1),):
            try:
                cap.set(prop, val)
            except Exception:
                pass
        if not cap or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return False
        self.cap = cap
        logger.info("[%s] RTSP opened", self.cam_id)
        # lazy model load on first successful open
        if not self.model_loaded:
            try:
                from ultralytics import YOLO
                model_path = str((MODELS_DIR / "yolo11n-seg.pt"))
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.model_loaded = True
                    logger.info("[%s] YOLO model loaded: %s", self.cam_id, model_path)
                else:
                    logger.warning("[%s] Model file not found: %s", self.cam_id, model_path)
            except Exception as e:
                logger.exception("[%s] Failed to load YOLO model: %s", self.cam_id, e)
                self.model = None
                self.model_loaded = False
        if not self.tracker:
            self.tracker = Tracker(distance_function=norfair.detections.iou, distance_threshold=0.5)
        return True

    def _close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    async def _loop(self):
        self.running = True
        backoff = 0.5
        max_back = 5.0
        while self.running:
            if self.cap is None:
                if not self._open():
                    logger.warning("[%s] open failed, retry in %.2fs", self.cam_id, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(max_back, backoff * 2)
                    continue
                backoff = 0.5
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            now = time.time()
            if not ok or frame is None:
                self._close()
                logger.warning("[%s] read failed, reconnecting...", self.cam_id)
                await asyncio.sleep(backoff)
                backoff = min(max_back, max(0.5, backoff * 2))
                continue

            # throttle
            elapsed = now - self.last_ts
            if elapsed < self.target_dt:
                await asyncio.sleep(self.target_dt - elapsed)
                now = time.time()

            # --- Server-side inference with rolling average ---
            do_infer = (self.model_loaded and (self.frame_idx % max(1, self.frame_skip) == 0))
            cur_count = self.last_count
            if do_infer:
                try:
                    # Inference
                    results = self.model.predict(
                        frame,
                        imgsz=640,
                        conf=self.conf_thres,
                        verbose=False
                    )
                    det_count = 0
                    detections = []
                    det_classes = []
                    det_ids = []
                    det_bboxes = []
                    det_masks = []
                    try:
                        r = results[0]
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
                                if c in self.target_class_ids and cf >= self.conf_thres:
                                    x1, y1, x2, y2 = b
                                    detections.append(Detection(points=np.array([[x1, y1], [x2, y2]])))
                                    det_classes.append(c)
                                    det_bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                        # --- Instance tracking ---
                        tracked = self.tracker.update(detections=detections)
                        det_count = len(tracked)
                        det_ids = [t.id for t in tracked]
                        # --- Overlay tracked objects ---
                        for t in tracked:
                            x1, y1 = t.estimate[0]
                            x2, y2 = t.estimate[1]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                            cv2.putText(frame, f"ID:{t.id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                        # --- Overlay masks (если есть) ---
                        if hasattr(r, "masks") and r.masks is not None:
                            masks = r.masks.data.cpu().numpy()
                            for mask in masks:
                                mask = (mask > 0.5).astype(np.uint8) * 255
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(frame, contours, -1, (0, 180, 255), 2)
                    except Exception:
                        det_count = 0
                        det_ids = []
                        det_classes = []
                        det_bboxes = []
                    # update rolling window
                    self.count_window.append(det_count)
                    if len(self.count_window) > self.avg_window:
                        self.count_window.pop(0)
                    self.avg_count = sum(self.count_window) / max(1, len(self.count_window))
                    cur_count = int(round(self.avg_count))
                    self.last_count = cur_count
                    self.last_debug_classes = det_classes
                    self.last_debug_ids = det_ids
                    self.last_debug_bboxes = det_bboxes
                except Exception:
                    # keep previous count on failure
                    pass

            self.frame_idx += 1

            # draw overlays: line + count box
            try:
                h, w = frame.shape[:2]
                # line (optional visual indicator)
                y_line = int(h * 0.5)
                cv2.line(frame, (0, y_line), (w, y_line), (0, 180, 255), 2)
                # count box
                cv2.rectangle(frame, (10, 10), (200, 55), (0, 0, 0), -1)
                cv2.putText(frame, f"Count: {cur_count}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                pass

            jpeg = encode_jpeg(frame)
            if jpeg:
                async with self.lock:
                    self.last_jpeg = jpeg
                    if self.last_ts > 0:
                        inst = 1.0 / max(1e-6, now - self.last_ts)
                        self.fps_window.append(inst)
                        if len(self.fps_window) > 30:
                            self.fps_window.pop(0)
                    self.last_ts = now

        self._close()

    def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=1.0)
            except Exception:
                pass
            self.task = None
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
            cs.start()
        return cs

    def get(self, cam_id: str) -> Optional[CameraStream]:
        return self.cams.get(cam_id)

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
def video_feed(mode: str = Query(default="server", pattern="^(server|client)$"),
               camera: str = Query(default=DEFAULT_CAM_ID)):
    cam = CAMERAS.get_or_create(camera, CAM_URL)
    return StreamingResponse(mjpeg_multicast(cam), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

@app.websocket("/ws/count")
async def ws_count(ws: WebSocket, camera: str = DEFAULT_CAM_ID):
    await ws.accept()
    cam = CAMERAS.get_or_create(camera, CAM_URL)
    try:
        await ws.send_text(json.dumps({"type": "status", "text": "connected"}))
        while True:
            await asyncio.sleep(1.0)
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
            await ws.send_text(json.dumps({"type": "error", "text": f"{e}"}))
        except Exception:
            pass

# --- Video file endpoints (ported) ---
def _open_file_cap(path: str):
    """
    Безопасное открытие файла (Windows-friendly):
    - Не используем CAP_FFMPEG (провоцирует libavcodec/pthread_frame asserts).
    - Форсируем однопоточность декодера и минимальную буферизацию, где поддерживается.
    - Все операции с cap обёрнуты в try/except.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(path)  # без CAP_FFMPEG
        if not cap or not cap.isOpened():
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
            return None, "Cannot open video file"
        try:
            cap.set(cv2.CAP_PROP_THREADS, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return cap, {"fps": float(fps), "frame_count": frame_count, "duration": float(duration)}
    except Exception as e:
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        return None, f"Exception: {e}"

_file_sessions: Dict[str, dict] = {}

@app.post("/api/video_file/open")
async def api_video_file_open(camera: str = Form(default="cam_file1"),
                              id: str = Form(default="file1"),
                              file: UploadFile = File(...)):
    try:
        dst = UPLOAD_DIR / (file.filename or "upload.bin")
        with open(dst, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        return JSONResponse({"error": f"upload failed: {e}"}, status_code=500)

    cap, meta = _open_file_cap(str(dst))
    if cap is None:
        return JSONResponse({"error": meta}, status_code=400)
    _file_sessions[id] = {
        "cap": cap,
        "path": str(dst),
        "camera": camera,
        "fps": meta["fps"],
        "frame_count": meta["frame_count"],
        "duration": meta["duration"],
    }
    return {"id": id, "camera": camera, "path": str(dst), **meta}

@app.get("/api/video_file/close")
def api_video_file_close(id: str = Query(default="file1")):
    sess = _file_sessions.pop(id, None)
    if not sess:
        return {"status": "noop"}
    try:
        if sess.get("cap"):
            sess["cap"].release()
    finally:
        return {"status": "closed", "id": id}

@app.get("/api/video_file/frame")
def api_video_file_frame(id: str = Query(default="file1"),
                         camera: str = Query(default="cam_file1"),
                         t: float = Query(default=0.0)):
    sess = _file_sessions.get(id)
    if not sess:
        return JSONResponse({"error": "file session not opened"}, status_code=400)
    cap = sess["cap"]
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t) * 1000.0))
    ok, frame = cap.read()
    if not ok or frame is None:
        return Response(b"", media_type="image/jpeg")
    # --- Инференс и overlay ---
    det_count = 0
    try:
        from ultralytics import YOLO
        model_path = str((MODELS_DIR / "yolo11n-seg.pt"))
        if not hasattr(api_video_file_frame, "model"):
            api_video_file_frame.model = YOLO(model_path)
        model = api_video_file_frame.model
        target_class_ids = {20, 17, 19}  # cow, sheep, horse
        conf_thres = 0.30
        results = model.predict(frame, imgsz=640, conf=conf_thres, verbose=False)
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for i, b in enumerate(xyxy):
                c = int(cls[i])
                cf = float(conf[i])
                if c in target_class_ids and cf >= conf_thres:
                    det_count += 1
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        if hasattr(r, "masks") and r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                mask = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 180, 255), 2)
        # Overlay count
        cv2.rectangle(frame, (10, 10), (200, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"Count: {det_count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    except Exception as e:
        pass
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        return Response(b"", media_type="image/jpeg")
    return Response(buf.tobytes(), media_type="image/jpeg")

def _gen_file_mjpeg(sess_id: str, rate: float):
    """
    Crash-safe генератор MJPEG для проигрывания файла:
    - Без CAP_FFMPEG
    - На каждой итерации пытаемся выставить безопасные параметры
    - Любые исключения подавляются, чтобы процесс не падал
    """
    import time as _time
    while True:
        try:
            sess = _file_sessions.get(sess_id)
            if not sess:
                break
            cap = sess.get("cap")
            if cap is None:
                _time.sleep(0.05)
                continue

            try:
                cap.set(cv2.CAP_PROP_THREADS, 1)
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            ok, frame = cap.read()
            if not ok or frame is None:
                _time.sleep(0.03)
                continue

            img = encode_jpeg(frame)
            if not img:
                _time.sleep(0.02)
                continue

            yield multipart_chunk(img)

            fps = sess.get("fps", 25.0) or 25.0
            delay = max(0.01, (1.0 / fps) / max(0.05, float(rate or 1.0)))
            _time.sleep(delay)
        except GeneratorExit:
            break
        except Exception:
            _time.sleep(0.05)
            continue

@app.get("/api/video_file/play")
def api_video_file_play(id: str = Query(default="file1"),
                        camera: str = Query(default="cam_file1"),
                        rate: float = Query(default=1.0)):
    if id not in _file_sessions:
        return JSONResponse({"error": "file session not opened"}, status_code=400)
    return StreamingResponse(_gen_file_mjpeg(id, rate), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


# -----------------------------
# Video file endpoints (compat)
# -----------------------------
_file_sessions = {}  # id -> dict(cap, fps, frame_count, duration, camera)

def _open_file_cap(path: str):
    """
    Открытие видеофайла с приоритетом CAP_FFMPEG и безопасными настройками декодера,
    чтобы минимизировать риск assert в libavcodec/pthread_frame на Windows.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    except Exception:
        cap = None
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(path)
    if not cap or not cap.isOpened():
        return None, "Cannot open video file"

    # Безопасные настройки (игнорируются, если backend не поддерживает)
    try:
        cap.set(cv2.CAP_PROP_THREADS, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return cap, {"fps": float(fps), "frame_count": frame_count, "duration": float(duration)}

@app.post("/api/video_file/open")
async def api_video_file_open(
    camera: str = Form(default="cam_file1"),
    id: str = Form(default="file1"),
    file: UploadFile = File(...)
):
    try:
        dst = UPLOAD_DIR / file.filename
        with open(dst, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        return JSONResponse({"error": f"upload failed: {e}"}, status_code=500)

    cap, meta = _open_file_cap(str(dst))
    if cap is None:
        return JSONResponse({"error": meta}, status_code=400)
    _file_sessions[id] = {
        "cap": cap,
        "path": str(dst),
        "camera": camera,
        "fps": meta["fps"],
        "frame_count": meta["frame_count"],
        "duration": meta["duration"],
    }
    return {
        "id": id,
        "camera": camera,
        "path": str(dst),
        "fps": meta["fps"],
        "frame_count": meta["frame_count"],
        "duration": meta["duration"],
    }

@app.get("/api/video_file/frame")
def api_video_file_frame(id: str = Query(default="file1"),
                         camera: str = Query(default="cam_file1"),
                         t: float = Query(default=0.0)):
    sess = _file_sessions.get(id)
    if not sess:
        return JSONResponse({"error": "file session not opened"}, status_code=400)
    cap = sess["cap"]
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t) * 1000.0))
        ok, frame = cap.read()
    except Exception:
        ok, frame = False, None
    if not ok or frame is None:
        return Response(b"", media_type="image/jpeg")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        return Response(b"", media_type="image/jpeg")
    return Response(buf.tobytes(), media_type="image/jpeg")

def _gen_file_mjpeg(sess_id: str, rate: float):
    import time as _time
    while True:
        sess = _file_sessions.get(sess_id)
        if not sess:
            break
        cap = sess["cap"]
        ok, frame = cap.read()
        if not ok or frame is None:
            _time.sleep(0.03)
            continue
        img = encode_jpeg(frame)
        if not img:
            continue
        yield multipart_chunk(img)
        fps = sess.get("fps", 25.0) or 25.0
        delay = max(0.005, (1.0 / fps) / max(0.01, float(rate or 1.0)))
        _time.sleep(delay)

@app.get("/api/video_file/play")
def api_video_file_play(id: str = Query(default="file1"),
                        camera: str = Query(default="cam_file1"),
                        rate: float = Query(default=1.0)):
    if id not in _file_sessions:
        return JSONResponse({"error": "file session not opened"}, status_code=400)
    return StreamingResponse(_gen_file_mjpeg(id, rate), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")
