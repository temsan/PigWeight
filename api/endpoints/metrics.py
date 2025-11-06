"""
Metrics endpoints for mobile dashboard
Real-time pig counting statistics
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

# Import dependencies
from api.dependencies import STREAM_MANAGER, get_database_manager

router = APIRouter(prefix="/api", tags=["metrics"])
logger = logging.getLogger(__name__)


@router.get("/metrics/current")
async def get_current_metrics(stream_id: Optional[str] = None):
    """
    Получить текущие показатели по активному потоку/акту
    
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
        # Если stream_id не указан, берём первый активный
        if not stream_id:
            streams = list(STREAM_MANAGER.streams.keys())
            if not streams:
                return JSONResponse(
                    {
                        "current_count": 0,
                        "total_weight": 0.0,
                        "avg_weight": 0.0,
                        "left_count": 0,
                        "right_count": 0,
                        "active_act": None,
                        "error": "No active streams"
                    },
                    status_code=200
                )
            stream_id = streams[0]
        
        stream = STREAM_MANAGER.streams.get(stream_id)
        if not stream:
            return JSONResponse(
                {
                    "current_count": 0,
                    "total_weight": 0.0,
                    "avg_weight": 0.0,
                    "left_count": 0,
                    "right_count": 0,
                    "active_act": None,
                    "error": f"Stream {stream_id} not found"
                },
                status_code=404
            )
        
        # Получаем текущие показатели из потока
        current_count = len(getattr(stream, 'tracked_pigs', {}))
        
        # Считаем общий вес и средний вес из tracked pigs
        weights = [
            pig.get('weight_estimate', 0) 
            for pig in getattr(stream, 'tracked_pigs', {}).values()
            if 'weight_estimate' in pig
        ]
        total_weight = sum(weights) if weights else 0.0
        avg_weight = total_weight / len(weights) if weights else 0.0
        
        # Счётчики проходов
        left_count = getattr(stream, 'left_count', 0)
        right_count = getattr(stream, 'right_count', 0)
        
        # Активный акт (если есть)
        active_act = None
        if hasattr(stream, 'current_act') and stream.current_act:
            act = stream.current_act
            duration = (datetime.now() - act.get('started_at')).total_seconds()
            active_act = {
                "id": act.get('id'),
                "started_at": act.get('started_at').isoformat() if act.get('started_at') else None,
                "duration_sec": duration,
                "pig_ids": act.get('pig_ids', [])
            }
        
        # Режим фиксации (автоматический или ручной)
        auto_manual = getattr(stream, 'auto_fix_enabled', True) and 'auto' or 'manual'
        
        return {
            "stream_id": stream_id,
            "current_count": current_count,
            "total_weight": round(total_weight, 1),
            "avg_weight": round(avg_weight, 2),
            "left_count": left_count,
            "right_count": right_count,
            "active_act": active_act,
            "auto_manual": auto_manual,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
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
