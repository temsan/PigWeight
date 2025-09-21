"""
Video processing endpoints
"""

import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Body
from fastapi.responses import JSONResponse

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
        # Валидация типа файла
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            return JSONResponse(
                {"error": f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"},
                status_code=400
            )
        
        # Проверка размера файла (максимум 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        file_content = await file.read()
        
        if len(file_content) > max_size:
            return JSONResponse(
                {"error": "Файл слишком большой. Максимальный размер: 500MB"},
                status_code=400
            )
        
        # Создание уникального имени файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Сохранение файла
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return {
            "status": "success",
            "filename": safe_filename,
            "path": str(file_path),
            "size": len(file_content),
            "message": "Файл успешно загружен"
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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