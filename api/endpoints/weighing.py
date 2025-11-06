"""
Weighing acts endpoints for mobile dashboard
Database-backed weighing acts and statistics
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from pig_tracking.database_manager import DatabaseManager

router = APIRouter(prefix="/api/weighing", tags=["weighing"])
logger = logging.getLogger(__name__)

# Глобальный экземпляр DatabaseManager (будет инициализирован в app.py)
db_manager: Optional[DatabaseManager] = None


def init_db_manager(manager: DatabaseManager):
    """Инициализация DatabaseManager из app.py"""
    global db_manager
    db_manager = manager
    logger.info("✅ DatabaseManager инициализирован в weighing endpoints")


@router.get("/acts")
async def get_weighing_acts(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    stream_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """
    Получить список актов взвешивания из БД
    
    Query params:
    - limit: количество записей (по умолчанию 50)
    - offset: смещение для пагинации
    - stream_id: фильтр по ID потока
    - date_from: фильтр по дате начала (ISO format)
    - date_to: фильтр по дате окончания (ISO format)
    
    Response:
    {
        "acts": [
            {
                "id": 1,
                "stream_id": "cam101",
                "started_at": "2025-11-06T12:30:00",
                "ended_at": "2025-11-06T12:45:30",
                "duration_sec": 930,
                "count": 48,
                "total_weight": 1850.5,
                "avg_weight": 38.6,
                "mode": "auto"
            },
            ...
        ],
        "total": 150,
        "limit": 50,
        "offset": 0
    }
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        # Получаем акты из БД
        acts = db_manager.get_weighing_acts(
            limit=limit,
            offset=offset,
            stream_id=stream_id,
            date_from=date_from,
            date_to=date_to
        )
        
        # Получаем общее количество
        total = db_manager.count_weighing_acts(
            stream_id=stream_id,
            date_from=date_from,
            date_to=date_to
        )
        
        return {
            "acts": acts,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting weighing acts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/acts/{act_id}")
async def get_weighing_act(act_id: int):
    """
    Получить детальную информацию об акте взвешивания
    
    Response:
    {
        "id": 1,
        "stream_id": "cam101",
        "started_at": "2025-11-06T12:30:00",
        "ended_at": "2025-11-06T12:45:30",
        "duration_sec": 930,
        "count": 48,
        "total_weight": 1850.5,
        "avg_weight": 38.6,
        "mode": "auto",
        "crossings": [
            {
                "track_id": 1,
                "side": "left",
                "timestamp": "2025-11-06T12:30:05"
            },
            ...
        ]
    }
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        act = db_manager.get_weighing_act_by_id(act_id)
        if not act:
            raise HTTPException(status_code=404, detail=f"Act {act_id} not found")
        
        # Получаем пересечения для этого акта
        crossings = db_manager.get_crossings_by_act(act_id)
        act['crossings'] = crossings
        
        return act
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting act {act_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_weighing_stats(
    stream_id: Optional[str] = None,
    period: str = Query("today", regex="^(today|week|month|all)$")
):
    """
    Получить статистику по актам взвешивания
    
    Query params:
    - stream_id: фильтр по ID потока
    - period: период (today, week, month, all)
    
    Response:
    {
        "period": "today",
        "stream_id": "cam101",
        "total_acts": 12,
        "total_pigs": 576,
        "total_weight": 22200.5,
        "avg_weight": 38.5,
        "avg_duration_sec": 850,
        "acts_by_hour": {
            "08": 2,
            "09": 3,
            ...
        }
    }
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        # Определяем временной диапазон
        now = datetime.now()
        if period == "today":
            date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            date_from = now - timedelta(days=7)
        elif period == "month":
            date_from = now - timedelta(days=30)
        else:  # all
            date_from = None
        
        # Получаем статистику
        stats = db_manager.get_weighing_stats(
            stream_id=stream_id,
            date_from=date_from.isoformat() if date_from else None
        )
        
        return {
            "period": period,
            "stream_id": stream_id,
            **stats
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_act(stream_id: Optional[str] = None):
    """
    Получить последний акт взвешивания
    
    Query params:
    - stream_id: фильтр по ID потока
    
    Response: то же что и /acts/{act_id}
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        act = db_manager.get_latest_weighing_act(stream_id=stream_id)
        if not act:
            raise HTTPException(status_code=404, detail="No acts found")
        
        # Получаем пересечения
        crossings = db_manager.get_crossings_by_act(act['id'])
        act['crossings'] = crossings
        
        return act
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest act: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
