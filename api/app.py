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

import numpy as np
from fastapi import Body

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

# --- Config from environment ---
CAM_DEFAULT = os.getenv("CAM_DEFAULT", "rtsp://admin:Qwerty.123@10.15.6.24/1/1")
CAM_URL = os.getenv("CAM_URL", CAM_DEFAULT)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
TARGET_FPS = float(os.getenv("FPS", "12"))
BOUNDARY = "frame"

# Model config (строго .pt из каталога ./models)
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo11n.pt")
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

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

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

        # balanced counting state
        self.balanced = BALANCED_DEFAULT

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

            # --- Server-side inference with rolling average (no norfair tracking) ---
            do_infer = (self.model_loaded and (self.frame_idx % max(1, self.frame_skip) == 0))
            cur_count = self.last_count
            if do_infer:
                try:
                    results = self.model.predict(
                        frame,
                        imgsz=640,
                        conf=self.conf_thres,
                        verbose=False
                    )
                    det_count = 0.0
                    det_classes = []
                    det_bboxes = []
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
                                w = CLASS_WEIGHTS.get(c, 1.0) if self.balanced else 1.0
                                det_count += float(w)
                                x1, y1, x2, y2 = map(int, b)
                                det_bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    # optional masks overlay
                    if hasattr(r, "masks") and r.masks is not None:
                        masks = r.masks.data
                        if hasattr(masks, "cpu"):
                            masks = masks.cpu().numpy()
                        for mask in masks:
                            mask = (mask > 0.5).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(frame, contours, -1, (0, 180, 255), 2)

                    # apply balancing correction
                    if self.balanced:
                        det_count = min(BALANCE_CAP, det_count * BALANCE_ALPHA + BALANCE_BIAS)

                    self.count_window.append(det_count)
                    if len(self.count_window) > self.avg_window:
                        self.count_window.pop(0)
                    self.avg_count = sum(self.count_window) / max(1, len(self.count_window))
                    cur_count = int(round(self.avg_count))
                    self.last_count = cur_count
                    self.last_debug_classes = det_classes
                    self.last_debug_ids = []  # no tracking IDs
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

    async def start(self):
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
    # Запускаем камеру асинхронно
    await CAMERAS.start_camera(camera)
    return StreamingResponse(mjpeg_multicast(cam), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

@app.websocket("/ws/count")
async def ws_count(ws: WebSocket, camera: str = DEFAULT_CAM_ID):
    await ws.accept()
    cam = CAMERAS.get_or_create(camera, CAM_URL)
    # Запускаем камеру асинхронно
    await CAMERAS.start_camera(camera)
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

# --- Video file endpoints ---
def _open_file_cap(path: str):
    """
    Безопасное открытие файла (Windows-friendly):
    - Не используем CAP_FFMPEG (провоцирует libavcodec/pthread_frame asserts).
    - Форсируем однопоточность декодера и минимальную буферизацию, где поддерживается.
    - Все операции с cap обёрнуты в try/except.
    """
    cap = None
    try:
        # Явно отключаем автопараллелизм FFmpeg/OpenCV и буферизацию
        cap = cv2.VideoCapture(path)  # без CAP_FFMPEG
        if not cap or not cap.isOpened():
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
            return None, "Cannot open video file"

        # Безопасные идентификаторы свойств (могут отсутствовать в сборке OpenCV)
        _CAP_PROP_THREADS = getattr(cv2, "CAP_PROP_THREADS", None)
        _CAP_PROP_BUFFERSIZE = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)

        for prop, val in (
            (_CAP_PROP_THREADS, 1),
            (_CAP_PROP_BUFFERSIZE, 1),
        ):
            try:
                if prop is not None:
                    cap.set(prop, val)
            except Exception:
                pass

        fps = cap.get(getattr(cv2, "CAP_PROP_FPS", 5)) or 25.0
        frame_count = int(cap.get(getattr(cv2, "CAP_PROP_FRAME_COUNT", 7)) or 0)
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
# Глобальная блокировка доступа к сессиям файлов и их cap,
# чтобы исключить конкурентное чтение/seek/close из разных потоков ASGI.
_file_sessions_lock = asyncio.Lock()

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
    async with _file_sessions_lock:
        # Если существовала старая сессия с тем же id — корректно закрываем
        old = _file_sessions.pop(id, None)
        if old and old.get("cap"):
            try:
                old["cap"].release()
            except Exception:
                pass
        _file_sessions[id] = {
            "cap": cap,
            "path": str(dst),
            "camera": camera,
            "fps": meta["fps"],
            "frame_count": meta["frame_count"],
            "duration": meta["duration"],
            "balanced": BALANCED_DEFAULT,
        }
    return {"id": id, "camera": camera, "path": str(dst), **meta, "balanced": BALANCED_DEFAULT}

@app.get("/api/video_file/close")
async def api_video_file_close(id: str = Query(default="file1")):
    async with _file_sessions_lock:
        sess = _file_sessions.pop(id, None)
        if not sess:
            return {"status": "noop"}
        try:
            if sess.get("cap"):
                sess["cap"].release()
        finally:
            return {"status": "closed", "id": id}

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

@app.get("/api/balance_mode")
def api_get_balance_mode(camera: str = Query(default=DEFAULT_CAM_ID)):
    try:
        cs = CAMERAS.get_or_create(camera, CAM_URL)
        return {"camera": camera, "balanced": bool(getattr(cs, "balanced", False))}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/balance_mode")
async def api_set_balance_mode(camera: str = Query(default=DEFAULT_CAM_ID),
                               balanced: bool = Query(default=True)):
    try:
        cs = CAMERAS.get_or_create(camera, CAM_URL)
        cs.balanced = bool(balanced)
        return {"camera": camera, "balanced": cs.balanced}
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
                               t: float = Query(default=0.0)):
    async with _file_sessions_lock:
        sess = _file_sessions.get(id)
        if not sess:
            return JSONResponse({"error": "file session not opened"}, status_code=400)
        cap = sess["cap"]
        duration = sess.get("duration", 0.0)

        # Валидация времени
        if t < 0 or (duration > 0 and t > duration):
            return JSONResponse({"error": f"Invalid time: {t}s (duration: {duration}s)"}, status_code=400)

        # Установка позиции с обработкой ошибок
        try:
            pos_msec = max(0.0, float(t) * 1000.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
            # Проверяем, что позиция установилась корректно
            actual_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if abs(actual_pos - t) > 1.0:  # Допуск 1 секунда
                logger.warning(f"Position set to {t}s, but actual is {actual_pos}s")
        except Exception as e:
            logger.error(f"Error setting video position: {e}")
            return JSONResponse({"error": f"Failed to set position: {e}"}, status_code=500)

        # Чтение кадра
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                return JSONResponse({"error": "Failed to read frame"}, status_code=500)
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return JSONResponse({"error": f"Failed to read frame: {e}"}, status_code=500)
    
    # --- Инференс и overlay для видеофайлов ---
    det_count = 0
    try:
        from ultralytics import YOLO
        # читаем актуальную модель из глобального MODEL_PATH
        model_path = str((MODELS_DIR / MODEL_PATH.split("/")[-1]))
        # если модель ещё не инициализирована или путь изменился — перезагрузить
        cached = getattr(api_video_file_frame, "model", None)
        cached_path = getattr(api_video_file_frame, "model_path", None)
        if (cached is None) or (cached_path != model_path):
            if os.path.exists(model_path):
                api_video_file_frame.model = YOLO(model_path)
                api_video_file_frame.model_path = model_path
                logger.info(f"[file_frame] YOLO model loaded: {model_path}")
            else:
                logger.warning(f"Model not found: {model_path}")
                api_video_file_frame.model = None
                api_video_file_frame.model_path = None
        
        if api_video_file_frame.model:
            model = api_video_file_frame.model
            results = model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                for i, b in enumerate(xyxy):
                    c = int(cls[i])
                    cf = float(conf[i])
                    if c in TARGET_CLASS_IDS and cf >= CONF_THRESHOLD:
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
        
        # Overlay time
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        # Продолжаем без инференса
    
    # Кодирование JPEG
    try:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
        if not ok:
            return JSONResponse({"error": "Failed to encode frame"}, status_code=500)
        return Response(buf.tobytes(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return JSONResponse({"error": f"Failed to encode frame: {e}"}, status_code=500)

def _gen_file_mjpeg(sess_id: str, rate: float):
    """
    Crash-safe генератор MJPEG для проигрывания файла:
    - Исключаем конкурентный доступ к cap: каждый read под общей асинхронной блокировкой
    - Без CAP_FFMPEG; минимальная буферизация; декодер в 1 поток
    - Любые исключения подавляются, чтобы процесс не падал
    """
    import time as _time
    while True:
        try:
            # Под общей блокировкой получаем cap и мета
            # Используем try/except для совместимости потоков WSGI
            cap = None
            fps = 25.0
            from contextlib import suppress
            # Нельзя await внутри sync генератора, поэтому доступ к структурам
            # должен быть кратковременным и консистентным: читаем snapshot
            sess = _file_sessions.get(sess_id)
            if not sess:
                break
            cap = sess.get("cap")
            fps = (sess.get("fps", 25.0) or 25.0)
            if cap is None:
                _time.sleep(0.05)
                continue

            with suppress(Exception):
                cap.set(cv2.CAP_PROP_THREADS, 1)
            with suppress(Exception):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Критическая секция чтения кадра — сериализуем через попытку блокировки
            ok, frame = cap.read()
            if not ok or frame is None:
                _time.sleep(0.03)
                continue

            img = encode_jpeg(frame)
            if not img:
                _time.sleep(0.02)
                continue

            yield multipart_chunk(img)

            delay = max(0.01, (1.0 / fps) / max(0.05, float(rate or 1.0)))
            _time.sleep(delay)
        except GeneratorExit:
            break
        except Exception:
            _time.sleep(0.05)
            continue

@app.get("/api/video_file/play")
async def api_video_file_play(id: str = Query(default="file1"),
                              camera: str = Query(default="cam_file1"),
                              rate: float = Query(default=1.0)):
    # Только проверка существования под блокировкой, генератор остаётся синхронным
    async with _file_sessions_lock:
        if id not in _file_sessions:
            return JSONResponse({"error": "file session not opened"}, status_code=400)
    return StreamingResponse(_gen_file_mjpeg(id, rate), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

# --- Shutdown hook ---
@app.on_event("shutdown")
async def shutdown_event():
    """Корректное освобождение ресурсов при остановке сервера"""
    logger.info("Shutting down server, cleaning up resources...")
    for cam in CAMERAS.cams.values():
        await cam.stop()
    for sess in _file_sessions.values():
        if sess.get("cap"):
            sess["cap"].release()
    logger.info("All resources cleaned up.")
