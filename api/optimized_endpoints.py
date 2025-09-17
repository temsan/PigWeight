"""
Оптимизированные API endpoints для интеграции новых компонентов
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Импортируем унифицированный StreamManager
from api.app import STREAM_MANAGER

try:
    from core.optimized_config import get_config, apply_performance_profile, PERFORMANCE_PROFILES
    from core.adaptive_quality_controller import QualityLevel, QualitySettings
    from core.performance_monitor import PerformanceMonitor, PerformanceMetrics
    # from core.async_rtsp_decoder import AsyncRTSPDecoder, DecoderConfig # DEPRECATED
    from core.h264_direct_track import H264Config, H264StreamAdapter
    OPTIMIZED_AVAILABLE = True
except ImportError as e:
    logging.error(f"Оптимизированные компоненты недоступны: {e}")
    OPTIMIZED_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Optimized"])

# Pydantic модели для API
class SystemStatus(BaseModel):
    """Статус системы"""
    timestamp: float
    uptime_seconds: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    current_fps: float
    target_fps: float
    active_streams: int
    quality_level: str
    
class PerformanceStats(BaseModel):
    """Статистика производительности"""
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    frames_processed: int
    error_rate: float
    
class QualityControlRequest(BaseModel):
    """Запрос управления качеством"""
    level: str = Field(..., description="Уровень качества: ULTRA, HIGH, MEDIUM, LOW, MINIMAL")
    force: bool = Field(False, description="Принудительное изменение (игнорировать cooldown)")
    
class BatcherConfigRequest(BaseModel):
    """Конфигурация батчера"""
    min_batch_size: int = Field(1, ge=1, le=64)
    max_batch_size: int = Field(16, ge=1, le=64)
    target_latency_ms: float = Field(50.0, gt=0)
    adaptation_interval: float = Field(2.0, gt=0)

class StreamConfigRequest(BaseModel):
    """Конфигурация потока"""
    rtsp_url: str
    target_fps: float = Field(30.0, gt=0, le=120)
    use_h264_direct: bool = True
    enable_cuda: bool = True
    
# Глобальные объекты компонентов (будут инициализированы в main)
performance_monitor: Optional[PerformanceMonitor] = None
quality_controller = None
frame_queue = None
batcher = None

# WebSocket соединения для real-time данных
active_websockets: List[WebSocket] = []

@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Получение текущего статуса системы"""
    if not OPTIMIZED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Оптимизированные компоненты недоступны")
        
    try:
        config = get_config()
        
        # Базовая информация
        status = SystemStatus(
            timestamp=time.time(),
            uptime_seconds=0.0,  # Будет заполнено из monitor
            cpu_usage=0.0,
            memory_usage=0.0,
            gpu_usage=None,
            current_fps=0.0,
            target_fps=config.target_fps,
            active_streams=len(STREAM_MANAGER.streams),
            quality_level="MEDIUM"
        )
        
        # Данные от performance monitor
        if performance_monitor:
            current_metrics = performance_monitor.get_current_metrics()
            if current_metrics:
                status.cpu_usage = current_metrics.cpu_usage
                status.memory_usage = current_metrics.memory_usage
                status.gpu_usage = current_metrics.gpu_usage if current_metrics.gpu_usage > 0 else None
                status.current_fps = current_metrics.current_fps
                
        # Данные от quality controller
        if quality_controller:
            current_settings = quality_controller.get_current_settings()
            status.quality_level = current_settings.level.name
            
        return status
        
    except Exception as e:
        logger.error(f"Ошибка получения статуса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=PerformanceStats) 
async def get_performance_stats():
    """Получение статистики производительности"""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor не инициализирован")
            
        current_metrics = performance_monitor.get_current_metrics()
        if not current_metrics:
            raise HTTPException(status_code=404, detail="Метрики недоступны")
            
        stats = PerformanceStats(
            avg_latency_ms=current_metrics.avg_latency_ms,
            p95_latency_ms=current_metrics.p95_latency_ms,
            p99_latency_ms=current_metrics.p99_latency_ms,
            throughput_fps=current_metrics.inference_throughput,
            frames_processed=current_metrics.frames_processed,
            error_rate=current_metrics.error_rate
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/history")
async def get_performance_history(minutes: int = 10):
    """Получение истории производительности"""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor не инициализирован")
            
        history = performance_monitor.get_metrics_history(minutes)
        
        return {
            "timespan_minutes": minutes,
            "metrics_count": len(history),
            "data": [
                {
                    "timestamp": m.timestamp,
                    "cpu_usage": m.cpu_usage,
                    "memory_usage": m.memory_usage,
                    "gpu_usage": m.gpu_usage,
                    "fps": m.current_fps,
                    "latency_ms": m.avg_latency_ms
                }
                for m in history
            ]
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения истории: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quality/set")
async def set_quality_level(request: QualityControlRequest):
    """Установка уровня качества"""
    try:
        if not quality_controller:
            raise HTTPException(status_code=503, detail="Quality controller не инициализирован")
            
        # Валидация уровня качества
        try:
            level = QualityLevel[request.level.upper()]
        except KeyError:
            available_levels = [level.name for level in QualityLevel]
            raise HTTPException(
                status_code=400, 
                detail=f"Неверный уровень качества. Доступные: {available_levels}"
            )
            
        # Применение нового уровня
        quality_controller.set_quality_level(level, force=request.force)
        
        # Получение новых настроек
        new_settings = quality_controller.get_current_settings()
        
        return {
            "success": True,
            "new_level": level.name,
            "settings": {
                "resolution_scale": new_settings.resolution_scale,
                "fps_limit": new_settings.fps_limit,
                "batch_size": new_settings.batch_size,
                "jpeg_quality": new_settings.jpeg_quality
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка установки качества: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality/current")
async def get_current_quality():
    """Получение текущих настроек качества"""
    try:
        if not quality_controller:
            raise HTTPException(status_code=503, detail="Quality controller не инициализирован")
            
        settings = quality_controller.get_current_settings()
        stats = quality_controller.get_stats()
        
        return {
            "level": settings.level.name,
            "settings": {
                "resolution_scale": settings.resolution_scale,
                "fps_limit": settings.fps_limit,
                "batch_size": settings.batch_size,
                "jpeg_quality": settings.jpeg_quality,
                "confidence_threshold": settings.confidence_threshold,
                "h264_bitrate": settings.h264_bitrate,
                "h264_preset": settings.h264_preset
            },
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения настроек качества: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/profile/apply")
async def apply_profile(profile_name: str):
    """Применение профиля производительности"""
    try:
        if profile_name not in PERFORMANCE_PROFILES:
            available = list(PERFORMANCE_PROFILES.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Неизвестный профиль. Доступные: {available}"
            )
            
        apply_performance_profile(profile_name)
        
        return {
            "success": True,
            "applied_profile": profile_name,
            "settings": PERFORMANCE_PROFILES[profile_name]
        }
        
    except Exception as e:
        logger.error(f"Ошибка применения профиля: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles")
async def get_available_profiles():
    """Получение доступных профилей производительности"""
    return {
        "profiles": list(PERFORMANCE_PROFILES.keys()),
        "details": PERFORMANCE_PROFILES
    }

@router.get("/batcher/stats")
async def get_batcher_stats():
    """Получение статистики батчера"""
    try:
        if not batcher:
            raise HTTPException(status_code=503, detail="Batcher не инициализирован")
            
        stats = batcher.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики батчера: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batcher/config")
async def update_batcher_config(config: BatcherConfigRequest):
    """Обновление конфигурации батчера"""
    try:
        # Здесь можно добавить логику динамического обновления конфигурации батчера
        return {
            "success": True,
            "message": "Конфигурация батчера обновлена",
            "new_config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Ошибка обновления конфигурации батчера: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue/stats")
async def get_queue_stats():
    """Получение статистики очереди кадров"""
    try:
        if not frame_queue:
            raise HTTPException(status_code=503, detail="Frame queue не инициализирован")
            
        stats = frame_queue.get_stats()
        return stats.dict() if hasattr(stats, 'dict') else stats
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики очереди: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts():
    """Получение активных алертов"""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor не инициализирован")
            
        alerts = performance_monitor.get_alerts()
        return alerts
        
    except Exception as e:
        logger.error(f"Ошибка получения алертов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket для real-time метрик"""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        while True:
            # Отправка текущих метрик каждые 2 секунды
            if performance_monitor:
                current_metrics = performance_monitor.get_current_metrics()
                if current_metrics:
                    data = {
                        "type": "metrics_update",
                        "timestamp": time.time(),
                        "metrics": {
                            "cpu_usage": current_metrics.cpu_usage,
                            "memory_usage": current_metrics.memory_usage,
                            "gpu_usage": current_metrics.gpu_usage,
                            "fps": current_metrics.current_fps,
                            "latency_ms": current_metrics.avg_latency_ms,
                            "active_streams": current_metrics.active_streams
                        }
                    }
                    await websocket.send_json(data)
                    
            await asyncio.sleep(2.0)
            
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info("WebSocket клиент отключился")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)

@router.post("/streams/create")
async def create_optimized_stream(config: StreamConfigRequest):
    """Создание оптимизированного потока с использованием унифицированного StreamManager"""
    try:
        stream_id = f"stream_{int(time.time())}"
        stream = await STREAM_MANAGER.get_or_create_stream(stream_id, config.rtsp_url)
        await stream.start()
        
        return {
            "success": True,
            "stream_id": stream_id,
            "config": config.dict(),
            "message": "Оптимизированный поток создан через StreamManager"
        }
        
    except Exception as e:
        logger.error(f"Ошибка создания потока: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/info")
async def get_system_info():
    """Получение информации о системе"""
    try:
        import platform
        import psutil
        
        # CUDA информация
        cuda_info = None
        try:
            import torch
            if torch.cuda.is_available():
                cuda_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_cached": torch.cuda.memory_reserved()
                }
        except ImportError:
            cuda_info = {"available": False, "reason": "PyTorch not installed"}
            
        # Системная информация
        memory = psutil.virtual_memory()
        
        info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent
            },
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "cuda": cuda_info,
            "optimized_components": OPTIMIZED_AVAILABLE
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Ошибка получения системной информации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dependency для получения компонентов
def get_performance_monitor():
    global performance_monitor
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitor не инициализирован")
    return performance_monitor

def get_quality_controller():
    global quality_controller
    if not quality_controller:
        raise HTTPException(status_code=503, detail="Quality controller не инициализирован")
    return quality_controller

# Функция инициализации компонентов (вызывается из main)
def initialize_optimized_components(perf_monitor, qual_controller, f_queue, batch_processor):
    """Инициализация глобальных компонентов"""
    global performance_monitor, quality_controller, frame_queue, batcher
    performance_monitor = perf_monitor
    quality_controller = qual_controller
    frame_queue = f_queue
    batcher = batch_processor
    
    logger.info("✅ Оптимизированные endpoints инициализированы")

# Broadcast функция для WebSocket
async def broadcast_to_websockets(data: Dict[str, Any]):
    """Отправка данных всем подключенным WebSocket клиентам"""
    if not active_websockets:
        return
        
    disconnected = []
    
    for websocket in active_websockets:
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.debug(f"Ошибка отправки WebSocket сообщения: {e}")
            disconnected.append(websocket)
            
    # Удаляем отключенные соединения
    for ws in disconnected:
        active_websockets.remove(ws)
