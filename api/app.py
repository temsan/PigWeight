from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import logging
import sys
import time
from pathlib import Path
from threading import Lock

# Add the root directory to the Python path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from core.config import MODEL_PATH, ONNX_PATH  # keep existing health check
from services.rtsp_manager import RTSPManager   # keep existing HLS endpoints

# Lazy import heavy deps
try:
    import cv2
except Exception:
    cv2 = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder=None)  # Disable default static folder
CORS(app)

# Define directories
STATIC_DIR = BASE_DIR / 'static'
MODELS_DIR = BASE_DIR / 'models'
STREAM_DIR = BASE_DIR / 'stream'

# Ensure all directories exist
for directory in [STATIC_DIR, MODELS_DIR, STREAM_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Configure static file serving
app.add_url_rule('/static/<path:filename>', 'static', build_only=True)
app.add_url_rule('/models/<path:filename>', 'model_file', build_only=True)
app.add_url_rule('/stream/<path:filename>', 'stream_file', build_only=True)

# Initialize RTSP manager
# If you don't need HLS via ffmpeg anymore, you can disable it by commenting the next line.
rtsp_manager = RTSPManager()

# Global MJPEG config/state (multi-camera capable)
# cameras: {camera_id: {rtsp_url, cy1, offset, frame_skip, conf_thres, seg_model_path}}
_cfg = {
    'cameras': {
        'cam1': {
            'rtsp_url': 'rtsp://admin:Qwerty.123@10.15.6.24/1/1',
            'cy1': 487,
            'offset': 6,
            'frame_skip': 3,
            'conf_thres': 0.25,
            'seg_model_path': str((BASE_DIR / 'models' / 'yolo11n-seg.pt'))
        }
    }
}
_state_lock = Lock()
# counts per camera
_last_counts = {}  # {camera_id: int}
# File video sessions for playback/seek
_file_sessions = {}  # {file_id: {'cap': cv2.VideoCapture, 'path': str, 'fps': float, 'frame_count': int, 'duration': float, 'camera': str, 'model_path': str}}
_file_lock = Lock()

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

# Serve model files with proper MIME type
@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(MODELS_DIR, filename, mimetype='application/octet-stream')

# Serve stream files with proper MIME type
@app.route('/stream/<path:filename>')
def serve_stream(filename):
    if filename.endswith('.m3u8'):
        return send_from_directory(STREAM_DIR, filename, mimetype='application/vnd.apple.mpegurl')
    elif filename.endswith('.ts'):
        return send_from_directory(STREAM_DIR, filename, mimetype='video/MP2T')
    return send_from_directory(STREAM_DIR, filename)

# NOTE: If HLS (ffmpeg) is not required, prefer using /api/video_config to set RTSP URL and consume /api/video_feed directly.
@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """
    Start or register a camera configuration.
    body: { camera_id: "cam1", rtsp_url: "...", cy1?, offset?, frame_skip?, conf_thres?, seg_model_path? }
    returns { processed_stream_url: f"/api/video_feed?camera=cam1" }
    """
    data = request.json or {}
    cam_id = data.get('camera_id', 'cam1')
    rtsp_url = data.get('rtsp_url', '')
    with _state_lock:
        cam = _cfg['cameras'].get(cam_id, {})
        if rtsp_url:
            cam['rtsp_url'] = rtsp_url
        # Ensure defaults so further code doesn't KeyError
        cam.setdefault('cy1', 487)
        cam.setdefault('offset', 6)
        cam.setdefault('frame_skip', 3)
        cam.setdefault('conf_thres', 0.25)
        cam.setdefault('seg_model_path', str((BASE_DIR / 'models' / 'yolo11n-seg.pt')))
        _cfg['cameras'][cam_id] = cam
    return jsonify({'processed_stream_url': f'/api/video_feed?camera={cam_id}'})

# Stop doesn't need to control ffmpeg when relying on MJPEG; keep as no-op for compatibility.
@app.route('/api/stream/stop/<camera_id>', methods=['POST'])
def stop_stream(camera_id):
    # No ffmpeg to stop; keep camera registered. Optionally could deregister here.
    return jsonify({'status': 'noop'})

@app.route('/api/health')
def health_check():
    with _state_lock:
        cams = list(_cfg['cameras'].keys())
        models_ok = {}
        # Report seg model presence for first camera (and overall default path)
        for cid, cfg in _cfg['cameras'].items():
            models_ok[cid] = os.path.exists(cfg.get('seg_model_path', ''))
    return jsonify({
        'status': 'ok',
        'cameras': cams,
        'models': {
            'yolo11n.pt': os.path.exists(MODEL_PATH),
            'yolo11n.onnx': os.path.exists(ONNX_PATH),
            'per_camera_seg': models_ok
        }
    })

# Configure MJPEG processing settings
@app.route('/api/video_config', methods=['GET','POST'])
def video_config():
    """
    Update camera config.
    Supports:
      - POST body JSON
      - GET query params for convenience: /api/video_config?camera_id=cam1&seg_model_path=models/yolo11n-seg.pt&rtsp_url=...
    body/params: { camera_id: "cam1", rtsp_url?, cy1?, offset?, frame_skip?, conf_thres?, seg_model_path? }
    """
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args.to_dict(flat=True)
    cam_id = data.get('camera_id') or data.get('camera') or 'cam1'
    with _state_lock:
        cam = _cfg['cameras'].get(cam_id, {})
        if 'rtsp_url' in data: cam['rtsp_url'] = data['rtsp_url']
        if 'seg_model_path' in data: cam['seg_model_path'] = data['seg_model_path']
        if 'cy1' in data: cam['cy1'] = int(data['cy1'])
        if 'offset' in data: cam['offset'] = int(data['offset'])
        if 'frame_skip' in data: cam['frame_skip'] = max(1, int(data['frame_skip']))
        if 'conf_thres' in data: cam['conf_thres'] = float(data['conf_thres'])
        if not cam:
            return jsonify({'error': f'Unknown camera {cam_id}'}), 400
        _cfg['cameras'][cam_id] = cam
        out = {cid: {**cfg, 'seg_model_path': cfg.get('seg_model_path', '')} for cid, cfg in _cfg['cameras'].items()}
    return jsonify({'status': 'ok', 'config': out})

@app.route('/api/video_count', methods=['GET'])
def video_count():
    """
    Query param:
      camera: camera id, default 'cam1'
    Returns JSON with count for the selected camera and optional per-camera summary.
    """
    cam_id = request.args.get('camera', 'cam1')
    with _state_lock:
        cnt = _last_counts.get(cam_id, 0)
        summary = dict(_last_counts)
    return jsonify({'camera': cam_id, 'count': cnt, 'all': summary})

def _load_model(path):
    from ultralytics import YOLO
    return YOLO(path)

def _draw(frame, line_y, count, dets):
    # dets: list of (x1,y1,x2,y2,id,conf)
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    for (x1,y1,x2,y2,ident,conf) in dets:
        x1=int(max(0,x1)); y1=int(max(0,y1)); x2=int(min(w-1,x2)); y2=int(min(h-1,y2))
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,64,255),2)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        txt = f"id {ident} {conf:.2f}"
        cv2.putText(frame, txt, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    # counter box
    cv2.rectangle(frame,(10,10),(200,55),(0,0,0),-1)
    cv2.putText(frame, f"Count: {count}", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return frame

def _update_tracker(state, boxes):
    # state contains dict with 'next_id', 'tracks': {id:(cx,cy)}
    # boxes: list of (x1,y1,x2,y2,conf)
    import math
    new_tracks = {}
    assigned = set()
    # compute centers
    centers = []
    for (x1,y1,x2,y2,conf) in boxes:
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        centers.append((cx,cy,conf,x1,y1,x2,y2))
    # match by nearest
    for tid,(pcx,pcy) in state['tracks'].items():
        best_i, best_d = -1, 1e9
        for i,(cx,cy,conf,x1,y1,x2,y2) in enumerate(centers):
            if i in assigned: continue
            d = math.hypot(pcx-cx,pcy-cy)
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d < 80:
            cx,cy,conf,x1,y1,x2,y2 = centers[best_i]
            new_tracks[tid] = (cx,cy)
            assigned.add(best_i)
        # else track drops
    # assign new ids to remaining
    for i,(cx,cy,conf,x1,y1,x2,y2) in enumerate(centers):
        if i in assigned: continue
        tid = state['next_id']
        state['next_id'] += 1
        new_tracks[tid] = (cx,cy)
        assigned.add(i)
    state['tracks'] = new_tracks
    # return list with ids
    out = []
    idx_map = {}
    # map centers back to ids by nearest
    for tid,(cx,cy) in state['tracks'].items():
        idx_map[tid] = (cx,cy)
    for (x1,y1,x2,y2,conf) in boxes:
        # find matching id by nearest center
        best_id, best_d = None, 1e9
        for tid,(cx,cy) in idx_map.items():
            d = (abs((x1+x2)/2 - cx) + abs((y1+y2)/2 - cy))
            if d < best_d:
                best_d, best_id = d, tid
        out.append((x1,y1,x2,y2,best_id,conf))
    return out

def _gen_mjpeg(cam_id: str):
    if cv2 is None:
        yield b''
        return
    # lazy import ultralytics
    try:
        with _state_lock:
            cam_cfg = _cfg['cameras'][cam_id]
        model = _load_model(cam_cfg['seg_model_path'])
    except Exception as e:
        logger.error(f"Failed to load model for {cam_id}: {e}")
        yield b''
        return
    # RTSP open
    cap = None
    try:
        rtsp_url = cam_cfg.get('rtsp_url', '')
        if not rtsp_url:
            logger.error(f"RTSP URL is empty for camera {cam_id}")
            yield b''
            return
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"Cannot open RTSP for camera {cam_id}: {rtsp_url}")
            yield b''
            return
        # state
        track_state = {'next_id':1, 'tracks':{}}
        counted_ids = set()
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame_idx += 1
            # frame skip
            if frame_idx % max(1, cam_cfg['frame_skip']) != 0:
                # still emit raw frame to keep MJPEG alive
                ret, buf = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                continue
            # inference
            try:
                results = model.predict(frame, imgsz=640, conf=cam_cfg['conf_thres'], verbose=False)
            except Exception as e:
                logger.error(f"Inference error [{cam_id}]: {e}")
                ret, buf = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                continue
            # parse cows only (COCO id 20 is 'cow' in some versions, YOLOv8 uses 20? classic coco has cow index 20 zero-based; older texts use 18)
            # ultralytics results: r.boxes.xyxy, r.boxes.cls, r.boxes.conf
            det_boxes = []
            try:
                r = results[0]
                if hasattr(r, 'boxes') and r.boxes is not None:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else []
                    conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
                    for i, b in enumerate(xyxy):
                        x1,y1,x2,y2 = b
                        c = int(cls[i]) if i < len(cls) else -1
                        cf = float(conf[i]) if i < len(conf) else 0.0
                        # pig custom model not provided; user asks to count pigs but we only have a generic seg model path name.
                        # Until custom pig class available, filter for 'cow'-like (commonly class id 20 in COCO; sometimes 18 in older mappings).
                        if c in (18, 19, 20, 21):  # sheep(19), cow(20) with 0-based; include nearby classes for robustness
                            det_boxes.append((float(x1), float(y1), float(x2), float(y2), cf))
            except Exception as e:
                logger.warning(f"Parsing detections failed: {e}")
            # tracking
            dets_with_ids = _update_tracker(track_state, det_boxes)
            # counting on line cross
            cy1 = cam_cfg.get('cy1', 487); off = cam_cfg.get('offset', 6)
            for (x1,y1,x2,y2,tid,cf) in dets_with_ids:
                cy = (y1+y2)/2.0
                if cy1 - off <= cy <= cy1 + off:
                    if tid not in counted_ids:
                        counted_ids.add(tid)
            with _state_lock:
                _last_counts[cam_id] = len(counted_ids)
            # overlay
            frame = _draw(frame, cy1, _last_counts.get(cam_id, 0), dets_with_ids)
            # encode and yield
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        if cap is not None:
            cap.release()

# -------------------------
# Models listing endpoint
# -------------------------
@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        files = []
        for name in os.listdir(MODELS_DIR):
            p = MODELS_DIR / name
            if p.is_file() and name.lower().endswith('.pt'):
                files.append(name)
        return jsonify({'models': sorted(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------
# Video file playback endpoints
# -------------------------
from werkzeug.utils import secure_filename

UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.m4v', '.webm'}

def _open_cap_for_file(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 'Cannot open video file'
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
    return cap, {'fps': float(fps), 'frame_count': frame_count, 'duration': float(duration)}

@app.route('/api/video_file/open', methods=['POST'])
def open_video_file():
    """
    Multipart upload: camera, id, file (binary)
    Returns: {id, camera, path, fps, frame_count, duration}
    Notes:
      - Server-side uploads to ./uploads without UI раздутия; один input в верхней панели.
    """
    if cv2 is None:
        return jsonify({'error': 'cv2 not available'}), 500
    cam_id = request.form.get('camera', 'cam_file1')
    file_id = request.form.get('id', 'file1')
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'file required'}), 400
    filename = secure_filename(file.filename)
    ext = (Path(filename).suffix or '').lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({'error': f'unsupported extension {ext}'}), 400
    dst = UPLOAD_DIR / filename
    try:
        file.save(dst)
    except Exception as e:
        return jsonify({'error': f'upload failed: {e}'}), 500

    path = str(dst)
    with _file_lock:
        # close existing
        sess = _file_sessions.get(file_id)
        if sess and sess.get('cap'):
            try: sess['cap'].release()
            except: pass
        # Prefer FFmpeg backend
        try:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        except Exception:
            cap = None
        if cap is None or not cap.isOpened():
            cap, meta = _open_cap_for_file(path)
            if cap is None:
                return jsonify({'error': meta}), 400
            fps = meta['fps']; frame_count = meta['frame_count']; duration = meta['duration']
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
        # ensure camera config exists
        with _state_lock:
            cam_cfg = _cfg['cameras'].get(cam_id, {
                'rtsp_url': '',
                'cy1': 487,
                'offset': 6,
                'frame_skip': 1,
                'conf_thres': 0.25,
                'seg_model_path': str((BASE_DIR / 'models' / 'yolo11n-seg.pt'))
            })
            _cfg['cameras'][cam_id] = cam_cfg
        _file_sessions[file_id] = {
            'cap': cap,
            'path': path,
            'camera': cam_id,
            'fps': float(fps),
            'frame_count': int(frame_count),
            'duration': float(duration),
            'pos_frame': 0
        }
    return jsonify({'id': file_id, 'camera': cam_id, 'path': path, 'fps': float(fps), 'frame_count': int(frame_count), 'duration': float(duration)})

@app.route('/api/video_file/close', methods=['GET'])
def close_video_file():
    file_id = request.args.get('id', 'file1')
    with _file_lock:
        sess = _file_sessions.pop(file_id, None)
        if not sess:
            return jsonify({'status': 'noop'})
        try:
            if sess.get('cap'):
                sess['cap'].release()
        finally:
            return jsonify({'status': 'closed', 'id': file_id})

def _infer_and_draw_for_cam(cam_id: str, frame):
    try:
        with _state_lock:
            cam_cfg = _cfg['cameras'][cam_id]
        model = _load_model(cam_cfg['seg_model_path'])
    except Exception as e:
        logger.error(f"Failed to load model for {cam_id}: {e}")
        return frame
    # inference
    try:
        results = model.predict(frame, imgsz=640, conf=cam_cfg['conf_thres'], verbose=False)
    except Exception as e:
        logger.error(f"Inference error (file) [{cam_id}]: {e}")
        return frame
    # parse
    det_boxes = []
    try:
        r = results[0]
        if hasattr(r, 'boxes') and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else []
            conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
            for i, b in enumerate(xyxy):
                x1,y1,x2,y2 = b
                c = int(cls[i]) if i < len(cls) else -1
                cf = float(conf[i]) if i < len(conf) else 0.0
                if c in (18,19,20,21):
                    det_boxes.append((float(x1), float(y1), float(x2), float(y2), cf))
    except Exception as e:
        logger.warning(f"Parsing detections failed (file): {e}")
    # simple tracker per frame (no persistent id for file seek); still count line-cross in-session
    track_state = {'next_id':1, 'tracks':{}}
    dets_with_ids = _update_tracker(track_state, det_boxes)
    cy1 = cam_cfg.get('cy1', 487); off = cam_cfg.get('offset', 6)
    counted = 0
    for (x1,y1,x2,y2,tid,cf) in dets_with_ids:
        cy = (y1+y2)/2.0
        if cy1 - off <= cy <= cy1 + off:
            counted += 1
    with _state_lock:
        _last_counts[cam_id] = counted
    frame = _draw(frame, cy1, counted, dets_with_ids)
    return frame

@app.route('/api/video_file/frame', methods=['GET'])
def video_file_frame():
    """
    Single frame by time (seconds): /api/video_file/frame?id=file1&camera=camFile&t=12.34
    """
    if cv2 is None:
        return Response(b'', mimetype='image/jpeg')
    file_id = request.args.get('id', 'file1')
    cam_id = request.args.get('camera', 'cam_file1')
    t = float(request.args.get('t', '0'))
    with _file_lock:
        sess = _file_sessions.get(file_id)
        if not sess:
            return jsonify({'error': 'file session not opened'}), 400
        cap = sess['cap']
        # seek
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t*1000.0))
        ok, frame = cap.read()
    if not ok or frame is None:
        return Response(b'', mimetype='image/jpeg')
    frame = _infer_and_draw_for_cam(cam_id, frame)
    ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        return Response(b'', mimetype='image/jpeg')
    return Response(buf.tobytes(), mimetype='image/jpeg')

def _gen_file_mjpeg(file_id: str, cam_id: str, rate: float):
    if cv2 is None:
        yield b''
        return
    with _file_lock:
        sess = _file_sessions.get(file_id)
        if not sess:
            yield b''
            return
        cap = sess['cap']
        fps = sess.get('fps', 25.0) or 25.0
    delay = max(0.005, (1.0 / fps) / max(0.01, float(rate or 1.0)))
    while True:
        with _file_lock:
            sess = _file_sessions.get(file_id)
            if not sess:
                break
            cap = sess['cap']
            ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        frame = _infer_and_draw_for_cam(cam_id, frame)
        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(delay)

@app.route('/api/video_file/play', methods=['GET'])
def video_file_play():
    """
    MJPEG streaming playback: /api/video_file/play?id=file1&camera=camFile&rate=1.0
    """
    file_id = request.args.get('id', 'file1')
    cam_id = request.args.get('camera', 'cam_file1')
    rate = float(request.args.get('rate', '1.0'))
    return Response(_gen_file_mjpeg(file_id, cam_id, rate), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/video_feed')
def video_feed():
    cam_id = request.args.get('camera', 'cam1')
    with _state_lock:
        if cam_id not in _cfg['cameras']:
            return jsonify({'error': f'Unknown camera {cam_id}'}), 400
        # Ensure all default keys exist to avoid KeyError later
        cam = _cfg['cameras'][cam_id]
        cam.setdefault('rtsp_url', '')
        cam.setdefault('cy1', 487)
        cam.setdefault('offset', 6)
        cam.setdefault('frame_skip', 3)
        cam.setdefault('conf_thres', 0.25)
        cam.setdefault('seg_model_path', str((BASE_DIR / 'models' / 'yolo11n-seg.pt')))
        rtsp = cam.get('rtsp_url', '')
    if not rtsp:
        return jsonify({'error': f'RTSP URL is not configured for {cam_id}'}), 400
    return Response(_gen_mjpeg(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')
