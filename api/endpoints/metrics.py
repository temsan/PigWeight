"""
Metrics endpoints for mobile dashboard
Real-time pig counting statistics

ОБНОВЛЕНО: 8 ноября 2025
- Миграция с STREAM_MANAGER на DatabaseManager
- Получение данных из PostgreSQL/Supabase вместо in-memory
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

# Import dependencies
from api.dependencies import STREAM_MANAGER, get_database_manager

router = APIRouter(prefix="/api", tags=["metrics"])
logger = logging.getLogger(__name__)


@router.get("/metrics/current")
async def get_current_metrics(stream_id: Optional[str] = None):
    """
    Получить текущие показатели по активному потоку/акту
    
    ОБНОВЛЕНО: Использует DatabaseManager для получения данных из PostgreSQL
    
    Для мобильного дашборда показывает:
    - Количество свиней (текущее)
    - Общий вес
    - Средний вес
    - Статус автоматической фиксации акта
    
    Response:
    {
        "stream_id": "cam101",
        "current_count": 14,
        "total_weight": 1850.5,
        "avg_weight": 132.2,
        "left_count": 25,
        "right_count": 23,
        "active_act": {
            "id": 1,
            "started_at": "2025-11-06T12:30:00",
            "duration_sec": 45.5,
            "pig_ids": [1, 2, 3, ...]
        },
        "auto_manual": "auto",  # 'auto' или 'manual'
        "timestamp": "2025-11-06T12:45:30"
    }
    """
    try:
        # Получаем DatabaseManager
        db = get_database_manager()
        
        # Если stream_id не указан, пытаемся получить из STREAM_MANAGER (fallback)
        if not stream_id and STREAM_MANAGER:
            streams = list(STREAM_MANAGER.streams.keys())
            if streams:
                stream_id = streams[0]
        
        # Если stream_id всё ещё не указан, используем дефолтное значение
        if not stream_id:
            stream_id = "default"
        
        # Получаем статистику из БД
        try:
            stats = db.get_stats_summary(stream_id=stream_id)
        except Exception as db_error:
            logger.warning(f"⚠️ Ошибка получения данных из БД: {db_error}")
            # Graceful degradation: возвращаем пустые данные
            return JSONResponse(
                {
                    "stream_id": stream_id,
                    "current_count": 0,
                    "total_weight": 0.0,
                    "avg_weight": 0.0,
                    "left_count": 0,
                    "right_count": 0,
                    "active_act": None,
                    "auto_manual": "auto",
                    "timestamp": datetime.now().isoformat(),
                    "warning": "Database unavailable, showing empty data"
                },
                status_code=200
            )
        
        # Проверяем наличие активного акта в STREAM_MANAGER (для real-time данных)
        active_act = None
        auto_manual = "auto"
        
        if STREAM_MANAGER and stream_id in STREAM_MANAGER.streams:
            stream = STREAM_MANAGER.streams[stream_id]
            
            # Получаем режим фиксации
            auto_manual = "auto" if getattr(stream, 'auto_fix_enabled', True) else "manual"
            
            # Получаем активный акт из потока (real-time)
            if hasattr(stream, 'current_act') and stream.current_act:
                act = stream.current_act
                duration = (datetime.now() - act.get('started_at')).total_seconds()
                active_act = {
                    "id": act.get('id'),
                    "started_at": act.get('started_at').isoformat() if act.get('started_at') else None,
                    "duration_sec": duration,
                    "pig_ids": act.get('pig_ids', [])
                }
        
        # Формируем ответ на основе данных из БД
        return {
            "stream_id": stream_id,
            "current_count": stats.get("total_pigs", 0),
            "total_weight": round(stats.get("total_weight", 0.0), 1),
            "avg_weight": round(stats.get("avg_weight", 0.0), 2),
            "left_count": stats.get("left_count", 0),
            "right_count": stats.get("right_count", 0),
            "active_act": active_act,
            "auto_manual": auto_manual,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения метрик: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e), "details": "Internal server error"},
            status_code=500
        )


@router.post("/metrics/toggle-auto-fix")
async def toggle_auto_fix(stream_id: Optional[str] = None, enabled: bool = True):
    """
    Переключить режим автоматической фиксации акта
    
    Query params:
    - stream_id: ID потока (опционально, по умолчанию первый активный)
    - enabled: True (автоматический) или False (ручной)
    
    Response:
    {
        "stream_id": "cam101",
        "auto_manual": "auto",
        "message": "Auto-fix mode enabled"
    }
    """
    try:
        # Если stream_id не указан, берём первый активный
        if not stream_id:
            streams = list(STREAM_MANAGER.streams.keys())
            if not streams:
                return JSONResponse(
                    {"error": "No active streams"},
                    status_code=404
                )
            stream_id = streams[0]
        
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse(
                {"error": f"Stream {stream_id} not found"},
                status_code=404
            )
        
        # Устанавливаем режим
        stream.auto_fix_enabled = enabled
        auto_manual = enabled and 'auto' or 'manual'
        
        message = f"Auto-fix mode {'enabled' if enabled else 'disabled'}"
        logger.info(f"📌 {stream_id}: {message}")
        
        return {
            "stream_id": stream_id,
            "auto_manual": auto_manual,
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Error toggling auto-fix: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@router.post("/metrics/manual-fix")
async def manual_fix_act(stream_id: Optional[str] = None):
    """
    Ручная фиксация текущего акта (завершение)
    
    Используется когда режим установлен на 'manual'
    
    Response:
    {
        "stream_id": "cam101",
        "act_fixed": {
            "id": 1,
            "started_at": "2025-11-06T12:30:00",
            "ended_at": "2025-11-06T12:45:30",
            "duration_sec": 45.5,
            "count": 48,
            "total_weight": 1850.5
        },
        "message": "Act fixed and saved"
    }
    """
    try:
        # Если stream_id не указан, берём первый активный
        if not stream_id:
            streams = list(STREAM_MANAGER.streams.keys())
            if not streams:
                return JSONResponse(
                    {"error": "No active streams"},
                    status_code=404
                )
            stream_id = streams[0]
        
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse(
                {"error": f"Stream {stream_id} not found"},
                status_code=404
            )
        
        # Завершаем текущий акт
        if not hasattr(stream, 'current_act') or not stream.current_act:
            return JSONResponse(
                {"error": "No active act to fix"},
                status_code=400
            )
        
        act = stream.current_act
        act['ended_at'] = datetime.now()
        duration = (act['ended_at'] - act['started_at']).total_seconds()
        
        # Формируем ответ
        fixed_act = {
            "id": act.get('id'),
            "started_at": act['started_at'].isoformat(),
            "ended_at": act['ended_at'].isoformat(),
            "duration_sec": duration,
            "count": len(act.get('pig_ids', [])),
            "total_weight": act.get('total_weight', 0.0)
        }
        
        # Очищаем текущий акт
        stream.current_act = None
        
        logger.info(f"📌 Ручная фиксация акта для {stream_id}: {fixed_act}")
        
        return {
            "stream_id": stream_id,
            "act_fixed": fixed_act,
            "message": "Act fixed and saved"
        }
        
    except Exception as e:
        logger.error(f"Error fixing act manually: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )



@router.get("/metrics/latest-act")
async def get_latest_act():
    """
    Получить последний завершенный акт взвешивания из БД
    
    Response:
    {
        "act": {
            "id": 1,
            "started_at": "2025-11-06T12:30:00",
            "ended_at": "2025-11-06T12:45:30",
            "duration_sec": 930,
            "left_count": 25,
            "right_count": 23,
            "peak_count": 14,
            "total_weight": 1850.5,
            "avg_weight": 132.2
        },
        "timestamp": "2025-11-06T13:00:00"
    }
    """
    try:
        db = get_database_manager()
        
        # Получаем последний акт
        acts = db.get_acts_by_period(
            start_date=datetime.now().replace(hour=0, minute=0, second=0),
            end_date=datetime.now()
        )
        
        # Берем только последний
        if acts:
            acts = [acts[-1]]
        
        if not acts:
            return JSONResponse(
                {
                    "act": None,
                    "message": "No acts found today",
                    "timestamp": datetime.now().isoformat()
                },
                status_code=200
            )
        
        act = acts[0]
        
        return {
            "act": {
                "id": act.get("id"),
                "started_at": act.get("started_at"),
                "ended_at": act.get("ended_at"),
                "duration_sec": act.get("duration_sec", 0),
                "left_count": act.get("left_count", 0),
                "right_count": act.get("right_count", 0),
                "peak_count": act.get("peak_count", 0),
                "total_weight": round(act.get("total_weight", 0.0), 1),
                "avg_weight": round(act.get("avg_weight", 0.0), 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting latest act: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )



# ============================================================================
# НОВЫЕ СТАНДАРТИЗИРОВАННЫЕ ENDPOINTS (8 ноября 2025)
# ============================================================================

@router.get("/stats/current")
async def get_current_stats(stream_id: Optional[str] = None):
    """
    Стандартизированный endpoint для получения текущей статистики
    
    Это новое название для /api/metrics/current согласно спецификации
    
    Response: см. get_current_metrics()
    """
    # Переиспользуем существующую логику
    return await get_current_metrics(stream_id=stream_id)


@router.get("/health")
async def health_check():
    """
    Проверка состояния системы
    
    Response:
    {
        "status": "healthy",
        "database": "connected",
        "stream_manager": "active",
        "timestamp": "2025-11-08T10:00:00"
    }
    """
    try:
        # Проверяем подключение к БД
        db_status = "disconnected"
        try:
            db = get_database_manager()
            if db.test_connection():
                db_status = "connected"
        except Exception as db_error:
            logger.warning(f"⚠️ БД недоступна: {db_error}")
            db_status = f"error: {str(db_error)}"
        
        # Проверяем STREAM_MANAGER
        stream_status = "inactive"
        active_streams = 0
        if STREAM_MANAGER:
            active_streams = len(STREAM_MANAGER.streams)
            stream_status = "active" if active_streams > 0 else "idle"
        
        # Определяем общий статус
        overall_status = "healthy"
        if db_status != "connected":
            overall_status = "degraded"  # Система работает, но БД недоступна
        
        return {
            "status": overall_status,
            "components": {
                "database": db_status,
                "stream_manager": stream_status,
                "active_streams": active_streams
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка health check: {e}", exc_info=True)
        return JSONResponse(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )


@router.get("/weighing/acts")
async def get_weighing_acts(
    start_date: Optional[str] = Query(None, description="Начало периода (ISO format)"),
    end_date: Optional[str] = Query(None, description="Конец периода (ISO format)"),
    stream_id: Optional[str] = Query(None, description="ID потока"),
    limit: int = Query(50, ge=1, le=1000, description="Количество записей"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации")
):
    """
    Получить список актов взвешивания за период
    
    Query params:
    - start_date: начало периода (ISO format, опционально)
    - end_date: конец периода (ISO format, опционально)
    - stream_id: фильтр по ID потока (опционально)
    - limit: количество записей (по умолчанию 50)
    - offset: смещение для пагинации (по умолчанию 0)
    
    Response:
    {
        "acts": [
            {
                "id": 1,
                "stream_id": "cam101",
                "started_at": "2025-11-08T10:00:00",
                "ended_at": "2025-11-08T10:15:30",
                "duration": 930,
                "left_count": 25,
                "right_count": 23,
                "peak_count": 14,
                "seen_total": 48
            },
            ...
        ],
        "total": 150,
        "limit": 50,
        "offset": 0
    }
    """
    try:
        db = get_database_manager()
        
        # Парсим даты
        start = None
        end = None
        if start_date:
            try:
                start = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        if end_date:
            try:
                end = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        
        # По умолчанию - последние 7 дней
        if not start:
            start = datetime.now() - timedelta(days=7)
        if not end:
            end = datetime.now()
        
        # Получаем акты
        acts = db.get_acts_by_period(
            start_date=start,
            end_date=end,
            stream_id=stream_id
        )
        
        # Применяем пагинацию
        total = len(acts)
        acts_page = acts[offset:offset + limit]
        
        return {
            "acts": acts_page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка получения актов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weighing/stats")
async def get_weighing_stats(
    start_date: Optional[str] = Query(None, description="Начало периода (ISO format)"),
    end_date: Optional[str] = Query(None, description="Конец периода (ISO format)"),
    stream_id: Optional[str] = Query(None, description="ID потока"),
    group_by: str = Query("day", regex="^(day|week|month)$", description="Группировка")
):
    """
    Получить агрегированную статистику по актам взвешивания
    
    Query params:
    - start_date: начало периода (ISO format, опционально, по умолчанию - сегодня)
    - end_date: конец периода (ISO format, опционально, по умолчанию - сегодня)
    - stream_id: фильтр по ID потока (опционально)
    - group_by: группировка (day/week/month, по умолчанию day)
    
    Response:
    {
        "period": {
            "start": "2025-11-08T00:00:00",
            "end": "2025-11-08T23:59:59"
        },
        "stream_id": "cam101",
        "total_acts": 15,
        "total_pigs": 720,
        "left_count": 360,
        "right_count": 360,
        "total_weight": 93600.0,
        "avg_weight": 130.0,
        "avg_duration": 45.5,
        "max_peak": 18
    }
    """
    try:
        db = get_database_manager()
        
        # Парсим даты
        start = None
        end = None
        if start_date:
            try:
                start = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        if end_date:
            try:
                end = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        
        # Получаем статистику
        stats = db.get_stats_summary(
            start_date=start,
            end_date=end,
            stream_id=stream_id
        )
        
        # Убираем детальный список актов из ответа (только агрегаты)
        if "acts" in stats:
            del stats["acts"]
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
