"""
Error handling middleware
"""

import logging
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

def setup_error_handling(app: FastAPI):
    """Настройка обработки ошибок"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Обработка HTTP ошибок"""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail, 
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Обработка Starlette HTTP ошибок"""
        logger.warning(f"Starlette HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail or "HTTP Error", 
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Обработка ошибок валидации"""
        logger.warning(f"Validation error: {exc.errors()} - {request.method} {request.url}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error", 
                "details": exc.errors(),
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Обработка общих ошибок"""
        error_id = f"ERR-{hash(str(exc)) % 10000:04d}"
        logger.error(f"[{error_id}] Unhandled exception in {request.method} {request.url}: {exc}", exc_info=True)
        
        # В режиме разработки возвращаем детали ошибки
        import os
        debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        
        content = {
            "error": "Internal server error",
            "error_id": error_id,
            "path": str(request.url.path)
        }
        
        if debug_mode:
            content["debug"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc()
            }
        
        return JSONResponse(
            status_code=500,
            content=content
        )