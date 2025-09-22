"""
CORS middleware configuration
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

def setup_cors(app: FastAPI):
    """Настройка CORS middleware"""
    
    # Получаем настройки CORS из переменных окружения
    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    allowed_methods = os.getenv("CORS_METHODS", "*").split(",")
    allowed_headers = os.getenv("CORS_HEADERS", "*").split(",")
    allow_credentials = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    
    # В продакшене предупреждаем о небезопасных настройках
    if "*" in allowed_origins and os.getenv("ENVIRONMENT", "development") == "production":
        logger.warning("⚠️ CORS configured to allow all origins in production environment!")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
    )
    
    logger.info(f"✅ CORS configured: origins={allowed_origins}, credentials={allow_credentials}")