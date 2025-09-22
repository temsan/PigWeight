"""
Request logging middleware
"""

import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import Response

logger = logging.getLogger("api.requests")

def setup_request_logging(app: FastAPI):
    """Настройка логирования запросов"""
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Middleware для логирования HTTP запросов"""
        start_time = time.time()
        
        # Логируем входящий запрос
        logger.info(f"→ {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
        
        # Выполняем запрос
        response = await call_next(request)
        
        # Вычисляем время выполнения
        process_time = time.time() - start_time
        
        # Логируем ответ
        status_emoji = "✅" if response.status_code < 400 else "❌"
        logger.info(f"← {status_emoji} {response.status_code} {request.method} {request.url.path} ({process_time:.3f}s)")
        
        # Добавляем заголовок с временем выполнения
        response.headers["X-Process-Time"] = str(process_time)
        
        return response