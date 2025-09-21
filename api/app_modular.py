"""
Модульное FastAPI приложение для PigWeight
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Импорт модульных компонентов
from .endpoints import (
    health_router,
    video_router,
    stream_router,
    websocket_router,
    files_router
)
from .middleware import setup_cors, setup_error_handling

# Настройка логирования
logger = logging.getLogger(__name__)

# Директории
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("🚀 PigWeight API starting up...")
    
    # Здесь можно добавить инициализацию:
    # - Подключение к базе данных
    # - Загрузка моделей ML
    # - Инициализация воркеров
    
    yield
    
    # Shutdown
    logger.info("🛑 PigWeight API shutting down...")
    
    # Здесь можно добавить очистку:
    # - Закрытие соединений
    # - Остановка воркеров
    # - Сохранение состояния

# Создание приложения
app = FastAPI(
    title="PigWeight API",
    description="API для системы видеообработки и подсчета свиней",
    version="3.0.0",
    lifespan=lifespan
)

# Настройка middleware
setup_cors(app)
setup_error_handling(app)

# Подключение роутеров
app.include_router(health_router)
app.include_router(video_router)
app.include_router(stream_router)
app.include_router(websocket_router)
app.include_router(files_router)

# Статические файлы
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Основные страницы
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Главная страница"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        except Exception as e:
            logger.error(f"Cannot read index.html: {e}")
            return HTMLResponse(
                content="<h1>PigWeight</h1><p>Error loading interface</p>",
                status_code=500
            )
    else:
        return HTMLResponse(
            content="<h1>PigWeight API</h1><p>Interface not found</p>",
            status_code=404
        )

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard():
    """Страница дашборда"""
    dashboard_path = STATIC_DIR / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        return HTMLResponse(
            content="<h1>Dashboard</h1><p>Dashboard not found</p>",
            status_code=404
        )

# Дополнительные endpoints для совместимости
@app.get("/api/cameras")
async def api_cameras():
    """Получение списка доступных камер"""
    # Здесь будет логика получения камер из переменных окружения
    cameras = {}
    
    # Поиск камер в переменных окружения
    for key, value in os.environ.items():
        if key.startswith("CAM_CH") and value:
            cam_id = key[6:]  # Убираем "CAM_CH"
            if cam_id.isdigit():
                cameras[f"cam{cam_id}"] = value
    
    # Fallback камера
    if not cameras:
        default_cam = os.getenv("CAM_URL") or os.getenv("CAM_DEFAULT")
        if default_cam:
            cameras["cam101"] = default_cam
    
    return cameras

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app_modular:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )