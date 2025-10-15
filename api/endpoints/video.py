"""
Video processing endpoints
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Body
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["video"])

# Получаем конфигурацию из основного приложения
try:
    from core.config import get_config
    config = get_config()
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except ImportError:
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_video_file(file: UploadFile = File(...)):
    """Загрузка видеофайла для обработки"""
    try:
        # Валидация имени файла
        if not file.filename:
            return JSONResponse(
                {"error": "Имя файла не указано"}, 
                status_code=400
            )
        
        # Валидация типа файла
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.flv'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            return JSONResponse(
                {"error": f"Неподдерживаемый формат файла: {file_extension}. Поддерживаемые: {', '.join(allowed_extensions)}"},
                status_code=400
            )
        
        # Чтение содержимого файла
        file_content = await file.read()
        
        # Проверка размера файла (максимум 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        if len(file_content) > max_size:
            return JSONResponse(
                {"error": f"Файл слишком большой: {len(file_content)/1024/1024:.1f}MB. Максимум: 500MB"},
                status_code=413
            )
        
        if len(file_content) == 0:
            return JSONResponse(
                {"error": "Файл пустой"}, 
                status_code=400
            )
        
        # Создание безопасного имени файла без timestamp префикса
        # Сохраняем оригинальное имя файла для предотвращения искажения
        import re
        import secrets
        
        # Очищаем имя файла от потенциально опасных символов
        safe_filename = re.sub(r'[^\w\-_\.]', '_', file.filename)
        
        # Используем оригинальное имя файла без добавления суффиксов
        # Если файл существует, он будет перезаписан
        file_path = UPLOAD_DIR / safe_filename
        
        # Сохранение файла
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"📁 Video uploaded: {safe_filename}, size: {len(file_content)/1024/1024:.1f}MB")
        
        # Попробуем получить метаданные видео
        try:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
            cap.release()
        except Exception as e:
            logger.warning(f"Could not get video metadata: {e}")
            fps = 25.0
            duration = 0.0
        
        return {
            "status": "success",
            "filename": file.filename,
            "safe_filename": safe_filename,
            "video_id": file_path.stem,
            "size_mb": round(len(file_content) / 1024 / 1024, 2),
            "file_path": str(file_path),  # Фронтенд ожидает file_path
            "path": str(file_path),       # Оставляем для совместимости
            "fps": fps,
            "duration": duration,
            "message": "Файл успешно загружен"
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading video: {e}", exc_info=True)
        
        # Более информативные сообщения об ошибках
        error_msg = str(e)
        if "Permission denied" in error_msg:
            error_msg = "Нет прав для сохранения файла. Проверьте права доступа к папке uploads."
        elif "No space left" in error_msg:
            error_msg = "Недостаточно места на диске для сохранения файла."
        elif "File too large" in error_msg:
            error_msg = "Файл слишком большой для загрузки."
        else:
            error_msg = f"Ошибка при загрузке файла: {error_msg}"
        
        return JSONResponse({"error": error_msg}, status_code=500)

@router.post("/lines")
async def api_set_lines(data: Dict[str, float] = Body(...)):
    """Установка позиций линий подсчета"""
    try:
        # Здесь будет логика установки линий
        # Пока возвращаем заглушку
        return {
            "status": "ok", 
            "left_x": data.get("left_x", 0.25), 
            "right_x": data.get("right_x", 0.75)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)