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

# @router.post("/upload")  # ОТКЛЮЧЕН: используется endpoint в video.py
# async def upload_video_file(file: UploadFile = File(...)):
#     """Загрузка видеофайла для обработки - ПЕРЕНЕСЕНО В video.py"""
#     pass