"""
File upload endpoints
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["files"])

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_video_file(file: UploadFile = File(...)):
    """Загрузка видеофайла для обработки"""
    logger.info(f"🎬 FILES ENDPOINT: Upload request received for file: {file.filename}")
    try:
        # Валидация файла
        if not file.filename:
            return JSONResponse(
                {"error": "Имя файла не указано"}, 
                status_code=400
            )
        
        # Проверяем расширение файла
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return JSONResponse(
                {"error": f"Неподдерживаемый формат файла: {file_ext}. Поддерживаемые: {', '.join(allowed_extensions)}"}, 
                status_code=400
            )
        
        # Читаем содержимое файла
        content = await file.read()
        
        # Проверяем размер файла (максимум 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        if len(content) > max_size:
            return JSONResponse(
                {"error": f"Файл слишком большой: {len(content)/1024/1024:.1f}MB. Максимум: 500MB"}, 
                status_code=413
            )
        
        if len(content) == 0:
            return JSONResponse(
                {"error": "Файл пустой"}, 
                status_code=400
            )
        
        # Создаем безопасное имя файла без timestamp префикса
        # Сохраняем оригинальное имя файла для предотвращения искажения
        import re
        
        # Очищаем имя файла от потенциально опасных символов
        safe_filename = re.sub(r'[^\w\-_\.]', '_', file.filename)
        
        # Если файл с таким именем уже существует, добавляем уникальный суффикс
        file_path = UPLOAD_DIR / safe_filename
        if file_path.exists():
            name_part = file_path.stem
            extension = file_path.suffix
            counter = 1
            while file_path.exists():
                safe_filename = f"{name_part}_{counter}{extension}"
                file_path = UPLOAD_DIR / safe_filename
                counter += 1
        
        # Сохраняем файл
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Получаем метаданные видео
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
        
        logger.info(f"📁 Video uploaded: {safe_filename}, size: {len(content)/1024/1024:.1f}MB, duration: {duration:.1f}s")
        
        return {
            "status": "success",
            "filename": file.filename,
            "safe_filename": safe_filename,
            "file_path": str(file_path),  # Фронтенд ожидает file_path
            "path": str(file_path),       # Оставляем для совместимости
            "size": len(content),
            "size_mb": round(len(content) / 1024 / 1024, 2),
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