"""
Streaming endpoints
"""

import asyncio
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse, Response

# Import dependencies from shared module
from api.dependencies import STREAM_MANAGER, TARGET_FPS, FileStream, perf_logger, av_meta

router = APIRouter(prefix="/api", tags=["stream"])

logger = logging.getLogger(__name__)

async def mjpeg_generator(stream):
    """MJPEG stream generator"""
    while stream.running:
        jpeg = await stream.get_jpeg()
        if jpeg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        # Убрана задержка для максимальной производительности
        # await asyncio.sleep(1.0 / TARGET_FPS)

@router.post("/stream/start")
async def api_stream_start(stream_id: str, source_uri: str):
    """Запуск потока обработки"""
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

@router.get("/stream/{stream_id}/stop")
async def api_stream_stop(stream_id: str):
    """Остановка потока обработки"""
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

@router.get("/stream/{stream_id}/snapshot")
async def api_stream_snapshot(stream_id: str):
    """Получение снимка из потока"""
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream or not stream.running:
        return JSONResponse({"error": "stream not found or not running"}, status_code=404)
    
    jpeg = await stream.get_jpeg()
    if not jpeg:
        return JSONResponse({"error": "no frame"}, status_code=404)
    
    return Response(content=jpeg, media_type="image/jpeg")

@router.get("/stream/{stream_id}/feed")
async def api_stream_feed(stream_id: str):
    """MJPEG поток"""
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

@router.get("/stream/{stream_id}/info")
async def api_stream_info(stream_id: str):
    """Информация о потоке"""
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

@router.get("/stream/{stream_id}/seek")
async def api_stream_seek(stream_id: str, t: float = Query(...)):
    """Перемотка потока"""
    stream = STREAM_MANAGER.streams.get(stream_id)
    if not stream:
        return JSONResponse({"error": "stream not found"}, status_code=404)
    await stream.seek(t)
    return {"status": "ok", "current_time": float(stream.current_time)}

@router.post("/stream/{stream_id}/optimize")
async def api_stream_optimize(stream_id: str, transport: str = Query("mjpeg")):
    """Оптимизация настроек потока"""
    try:
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse({"error": "stream not found"}, status_code=404)

        logger.info(f"Optimizing stream {stream_id} for WebRTC transport")
        
        # WebRTC optimization logic would go here
        # For now, just return success
        return {
            "status": "optimized",
            "transport": transport,
            "stream_id": stream_id
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.post("/stream/{stream_id}/line_positions")
async def api_set_line_positions(stream_id: str, positions: Dict[str, Any] = Body(...)):
    """Сохранить позиции линий для конкретного видеофайла"""
    try:
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse({"error": f"Stream {stream_id} not found"}, status_code=404)

        # Save line positions logic would go here
        # For now, just return success
        return {
            "status": "ok",
            "stream_id": stream_id,
            "positions": positions
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)