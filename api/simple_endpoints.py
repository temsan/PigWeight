"""
Упрощенные эндпоинты без лишних конверсий для максимальной производительности
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Query, UploadFile, File, Form, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .lightweight_processor import (
    LightweightVideoProcessor,
    StreamingVideoProcessor,
    encode_frame_fast,
    FrameResult
)

import logging
logger = logging.getLogger(__name__)

# Глобальное хранилище процессоров
processors: Dict[str, LightweightVideoProcessor] = {}
streaming_processors: Dict[str, StreamingVideoProcessor] = {}

router = APIRouter(prefix="/api/simple", tags=["simple"])

@router.post("/video/upload")
async def upload_video_simple(
    id: str = Form(...),
    file: UploadFile = File(...)
):
    """Загрузка видео без лишней обработки"""
    try:
        # Очистка старого процессора если есть
        if id in processors:
            processors[id].close()
            del processors[id]
            
        # Сохраняем файл
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Создаем процессор
        processor = LightweightVideoProcessor(str(file_path))
        if not processor.open():
            raise HTTPException(status_code=400, detail="Cannot open video file")
            
        processors[id] = processor
        
        return {
            "id": id,
            "filename": file.filename,
            "fps": processor.fps,
            "duration": processor.duration,
            "total_frames": processor.total_frames
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}/frame")
async def get_frame_simple(
    video_id: str,
    t: float = Query(..., description="Timestamp in seconds"),
    inference: bool = Query(default=False),
    quality: int = Query(default=85, ge=10, le=100)
):
    """Сверхбыстрое получение кадра для максимальной отзывчивости"""
    if video_id not in processors:
        raise HTTPException(status_code=404, detail="Video not found")
        
    processor = processors[video_id]
    
    try:
        start_time = time.time()
        
        # По умолчанию используем быстрый режим без инференса
        if inference:
            # Инференс только при явном запросе
            if not processor.inference_enabled:
                processor.enable_inference("models/yolo11n.pt")  
            result = processor.get_frame_with_inference(t)
            if not result:
                raise HTTPException(status_code=404, detail="Frame not found")
            frame = result.frame
            count = result.count
            inference_time = result.inference_time_ms
        else:
            # Быстрый режим - только кадр
            frame = processor.get_frame_fast(t)
            if frame is None:
                raise HTTPException(status_code=404, detail="Frame not found")
            count = 0
            inference_time = 0.0
                
        # Максимально быстрое кодирование
        jpeg_bytes = encode_frame_fast(frame, quality)
        if not jpeg_bytes:
            raise HTTPException(status_code=500, detail="Encoding failed")
            
        total_time = (time.time() - start_time) * 1000
        
        headers = {
            "X-Processing-Time": f"{total_time:.1f}ms",
            "X-Frame-Time": f"{t:.3f}",
            "X-Inference": str(inference).lower(),
            "X-Count": str(count),
            "X-Inference-Time": f"{inference_time:.1f}ms",
            "Cache-Control": "no-cache"  # Для отзывчивости
        }
            
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}/stream")
async def start_stream_simple(
    video_id: str,
    inference: bool = Query(default=False),
    fps: Optional[float] = Query(default=None)
):
    """Запуск простого стрима без лишних накладок"""
    if video_id not in processors:
        raise HTTPException(status_code=404, detail="Video not found")
        
    # Создаем стрим процессор
    base_processor = processors[video_id]
    stream_processor = StreamingVideoProcessor(base_processor.video_path)
    
    model_path = "models/yolo11n.pt" if inference else None
    success = await stream_processor.start_streaming(inference, model_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="Cannot start streaming")
        
    streaming_processors[video_id] = stream_processor
    
    target_fps = fps or base_processor.fps
    frame_interval = 1.0 / target_fps
    
    async def generate_frames():
        """Генератор кадров для стрима"""
        boundary = "frame_boundary"
        
        try:
            while stream_processor.is_playing:
                frame_start = time.time()
                
                result = await stream_processor.get_frame_stream()
                if not result:
                    break
                    
                # Быстрое кодирование
                jpeg_data = encode_frame_fast(result.frame, quality=70)
                if not jpeg_data:
                    continue
                    
                # Формируем multipart chunk
                chunk = (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg_data)}\r\n"
                    f"X-Timestamp: {stream_processor.current_time:.3f}\r\n"
                ).encode() + b"\r\n" + jpeg_data + b"\r\n"
                
                yield chunk
                
                # Контроль FPS
                elapsed = time.time() - frame_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            if video_id in streaming_processors:
                streaming_processors[video_id].stop()
                del streaming_processors[video_id]
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame_boundary"
    )

@router.post("/video/{video_id}/seek")
async def seek_simple(
    video_id: str,
    timestamp: float = Query(...)
):
    """Простой seek без лишних проверок"""
    if video_id in streaming_processors:
        streaming_processors[video_id].seek(timestamp)
        return {"status": "ok", "timestamp": timestamp}
    else:
        raise HTTPException(status_code=404, detail="No active stream")

@router.post("/video/{video_id}/stop")
async def stop_stream_simple(video_id: str):
    """Остановка стрима"""
    if video_id in streaming_processors:
        streaming_processors[video_id].stop()
        del streaming_processors[video_id]
        return {"status": "stopped"}
    else:
        raise HTTPException(status_code=404, detail="No active stream")

@router.get("/video/{video_id}/info")
async def get_video_info_simple(video_id: str):
    """Информация о видео"""
    if video_id not in processors:
        raise HTTPException(status_code=404, detail="Video not found")
        
    processor = processors[video_id]
    
    # Информация о стриме если активен
    stream_info = None
    if video_id in streaming_processors:
        stream_proc = streaming_processors[video_id]
        stream_info = {
            "playing": stream_proc.is_playing,
            "current_time": stream_proc.current_time,
            "speed": stream_proc.play_speed
        }
    
    return {
        "id": video_id,
        "fps": processor.fps,
        "duration": processor.duration,
        "total_frames": processor.total_frames,
        "cache_size": len(processor.frame_cache),
        "inference_enabled": processor.inference_enabled,
        "stream": stream_info
    }

@router.delete("/video/{video_id}")
async def delete_video_simple(video_id: str):
    """Удаление видео и очистка ресурсов"""
    # Останавливаем стрим если есть
    if video_id in streaming_processors:
        streaming_processors[video_id].stop()
        del streaming_processors[video_id]
        
    # Закрываем процессор
    if video_id in processors:
        processors[video_id].close()
        del processors[video_id]
        
    return {"status": "deleted", "id": video_id}

@router.get("/stats")
async def get_simple_stats():
    """Статистика упрощенной системы"""
    return {
        "active_videos": len(processors),
        "active_streams": len(streaming_processors),
        "video_ids": list(processors.keys()),
        "stream_ids": list(streaming_processors.keys())
    }

# Функции для интеграции с основным приложением
def cleanup_all():
    """Очистка всех ресурсов при shutdown"""
    for processor in processors.values():
        processor.close()
    processors.clear()
    
    for stream_processor in streaming_processors.values():
        stream_processor.stop()
    streaming_processors.clear()

def get_processor(video_id: str) -> Optional[LightweightVideoProcessor]:
    """Получение процессора по ID"""
    return processors.get(video_id)
