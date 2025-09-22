"""
Security middleware
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

def setup_security_headers(app: FastAPI):
    """Настройка заголовков безопасности"""
    
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Middleware для добавления заголовков безопасности"""
        response = await call_next(request)
        
        # Основные заголовки безопасности
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy (базовый)
        csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' ws: wss:;"
        response.headers["Content-Security-Policy"] = csp
        
        # HTTPS-only в продакшене
        if os.getenv("ENVIRONMENT", "development") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    logger.info("✅ Security headers middleware configured")