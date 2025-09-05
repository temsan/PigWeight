"""
Упрощенные API endpoints с использованием единого процессора
Заменяет все legacy endpoints
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np

# Импортируем единый процессор
try:
    from core.processor import get_processor, ProcessingOptions
    HAVE_UNIFIED_PROCESSOR = True
except ImportError:
    HAVE_UNIFIED_PROCESSOR = False
    logging.error("❌ Unified processor not available")

logger = logging.getLogger(__name__)

# Конфигурация
VIDEO_DIR = Path("uploads")
RECORDS_DIR = Path("records")

# Глобальный экземпляр процессора
_processor = None

def get_video_processor():
    """Получение процессора с автоконфигурацией"""
    global _processor
    if _processor is None and HAVE_UNIFIED_PROCESSOR:
        model_path = os.getenv("MODEL_PATH", "models/pig_yolo11-seg.v4.pt")
        if not Path(model_path).exists():
            # Пробуем найти любую доступную модель
            models_dir = Path("models")
            for model_file in models_dir.glob("*.pt"):
                model_path = str(model_file)
                break
        
        options = ProcessingOptions(
            confidence_threshold=float(os.getenv("CONF_THRESHOLD", "0.3")),
            img_size=int(os.getenv("IMG_SIZE", "640")),
            device=os.getenv("DEVICE", "auto"),
            batch_size=int(os.getenv("BATCH_SIZE", "1"))
        )
        
        try:
            _processor = get_processor(model_path, options)
            logger.info(f"✅ Video processor initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize processor: {e}")
            _processor = None
    
    return _processor

def setup_endpoints(app: FastAPI):
    """Настройка упрощенных endpoints"""
    
    @app.get("/api/video/process")
    async def process_video_endpoint(
        video_id: str = Query(..., description="ID видеофайла"),
        frame_skip: int = Query(1, description="Обрабатывать каждый N-й кадр"),
        start_sec: float = Query(0, description="Начальное время в секундах"),
        end_sec: Optional[float] = Query(None, description="Конечное время в секундах")
    ):
        """Обработка видеофайла с детекцией свиней"""
        try:
            # Проверяем процессор
            processor = get_video_processor()
            if not processor:
                return JSONResponse(
                    {"error": "Video processor not available"}, 
                    status_code=503
                )
            
            # Находим видеофайл
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                candidate = VIDEO_DIR / f"{video_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if not video_path:
                return JSONResponse(
                    {"error": f"Video {video_id} not found"}, 
                    status_code=404
                )
            
            logger.info(f"🎬 Processing video: {video_path}")
            
            # Обрабатываем видео
            results = await processor.process_video(
                video_path,
                frame_skip=frame_skip
            )
            
            # Сохраняем результаты
            output_file = RECORDS_DIR / f"{video_id}_analysis.json"
            RECORDS_DIR.mkdir(exist_ok=True)
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"✅ Analysis saved to: {output_file}")
            
            return {
                "status": "success",
                "video_id": video_id,
                "video_path": str(video_path),
                "output_file": str(output_file),
                "processor_stats": processor.get_stats(),
                **results
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing video {video_id}: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @app.get("/api/video/frame")
    async def process_frame_endpoint(
        video_id: str = Query(..., description="ID видеофайла"),
        frame_number: int = Query(..., description="Номер кадра"),
        timestamp: Optional[float] = Query(None, description="Временная метка")
    ):
        """Обработка одного кадра"""
        try:
            processor = get_video_processor()
            if not processor:
                return JSONResponse(
                    {"error": "Video processor not available"}, 
                    status_code=503
                )
            
            # Находим видеофайл
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                candidate = VIDEO_DIR / f"{video_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
                    
            if not video_path:
                return JSONResponse(
                    {"error": f"Video {video_id} not found"}, 
                    status_code=404
                )
            
            # Извлекаем кадр
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return JSONResponse(
                    {"error": "Cannot open video"}, 
                    status_code=500
                )
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return JSONResponse(
                    {"error": "Cannot read frame"}, 
                    status_code=404
                )
            
            # Обрабатываем кадр
            if timestamp is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                timestamp = frame_number / fps
            
            result = await processor.process_frame(frame, frame_number, timestamp)
            
            return {
                "status": "success",
                "video_id": video_id,
                "frame_number": frame_number,
                "timestamp": timestamp,
                "result": {
                    "pig_count": result.pig_count,
                    "detections": result.detections,
                    "confidence_scores": result.confidence_scores,
                    "processing_time_ms": result.processing_time_ms
                }
            }
                    
        except Exception as e:
            logger.error(f"❌ Error processing frame: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @app.get("/api/system/status")
    async def system_status():
        """Статус системы и процессора"""
        processor = get_video_processor()
        
        status = {
            "unified_processor_available": HAVE_UNIFIED_PROCESSOR,
            "processor_initialized": processor is not None,
            "video_directory": str(VIDEO_DIR),
            "records_directory": str(RECORDS_DIR),
            "available_videos": []
        }
        
        # Список доступных видео
        if VIDEO_DIR.exists():
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                for video_file in VIDEO_DIR.glob(f"*{ext}"):
                    status["available_videos"].append({
                        "id": video_file.stem,
                        "filename": video_file.name,
                        "size_mb": round(video_file.stat().st_size / 1024 / 1024, 2)
                    })
        
        # Статистика процессора
        if processor:
            status["processor_stats"] = processor.get_stats()
        
        return status
    
    @app.post("/api/video/upload")
    async def upload_video(file: UploadFile = File(...)):
        """Загрузка видеофайла"""
        try:
            # Проверяем тип файла
            if not file.content_type or not file.content_type.startswith('video/'):
                return JSONResponse(
                    {"error": "Only video files are allowed"}, 
                    status_code=400
                )
            
            # Создаем директорию если нужно
            VIDEO_DIR.mkdir(exist_ok=True)
            
            # Сохраняем файл
            file_path = VIDEO_DIR / file.filename
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"📁 Video uploaded: {file_path}")
            
            return {
                "status": "success",
                "filename": file.filename,
                "video_id": file_path.stem,
                "size_mb": round(len(content) / 1024 / 1024, 2),
                "path": str(file_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Error uploading video: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    logger.info("✅ Simplified endpoints configured")