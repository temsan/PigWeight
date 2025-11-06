"""
Weighing acts endpoints
CRUD operations for weighing acts and statistics
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import get_database_manager

router = APIRouter(prefix="/api/weighing", tags=["weighing"])
logger = logging.getLogger(__name__)


@router.get("/acts")
async def get_weighing_acts(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of acts to return")
):
    """
    Получить список актов взвешивания за период
    
    Query params:
    - start_date: начальная дата (по умолчанию: сегодня)
    - end_date: конечная дата (по умолчанию: сегодня)
    - limit: максимальное количество актов
    
    Response:
    {
        "acts": [
            {
                "id": 1,
                "started_at": "2025-11-06T12:30:00",
                "ended_at": "2025-11-06T12:45:30",
                "duration_sec": 930,
                "left_count": 25,
                "right_count": 23,
                "peak_count": 14,
                "total_weight": 1850.5,
                "avg_weight": 132.2
            }
        ],
        "total": 1,
        "start_date": "2025-11-06",
        "end_date": "2025-11-06"
    }
    """
    try:
        db = get_database_manager()
        
        # Парсим даты
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Получаем акты из БД
        acts = db.get_acts_by_period(start_dt, end_dt, limit=limit)
        
        # Форматируем ответ
        formatted_acts = []
        for act in acts:
            formatted_acts.append({
                "id": act.get("id"),
                "started_at": act.get("started_at"),
                "ended_at": act.get("ended_at"),
                "duration_sec": act.get("duration_sec", 0),
                "left_count": act.get("left_count", 0),
                "right_count": act.get("right_count", 0),
                "peak_count": act.get("peak_count", 0),
                "total_weight": round(act.get("total_weight", 0.0), 1),
                "avg_weight": round(act.get("avg_weight", 0.0), 2)
            })
        
        return {
            "acts": formatted_acts,
            "total": len(formatted_acts),
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting weighing acts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_weighing_stats(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Получить статистику по актам взвешивания за период
    
    Response:
    {
        "period": {
            "start": "2025-11-06",
            "end": "2025-11-06"
        },
        "total_acts": 5,
        "total_pigs": 240,
        "total_weight": 9252.5,
        "avg_weight": 38.6,
        "avg_duration_sec": 850,
        "by_day": [
            {
                "date": "2025-11-06",
                "acts": 5,
                "pigs": 240,
                "weight": 9252.5
            }
        ]
    }
    """
    try:
        db = get_database_manager()
        
        # Парсим даты
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Получаем акты
        acts = db.get_acts_by_period(start_dt, end_dt)
        
        if not acts:
            return {
                "period": {
                    "start": start_dt.date().isoformat(),
                    "end": end_dt.date().isoformat()
                },
                "total_acts": 0,
                "total_pigs": 0,
                "total_weight": 0.0,
                "avg_weight": 0.0,
                "avg_duration_sec": 0,
                "by_day": []
            }
        
        # Вычисляем общую статистику
        total_acts = len(acts)
        total_pigs = sum(act.get("left_count", 0) + act.get("right_count", 0) for act in acts)
        total_weight = sum(act.get("total_weight", 0.0) for act in acts)
        avg_weight = total_weight / total_pigs if total_pigs > 0 else 0.0
        avg_duration = sum(act.get("duration_sec", 0) for act in acts) / total_acts if total_acts > 0 else 0
        
        # Группируем по дням
        by_day = {}
        for act in acts:
            date_str = act.get("started_at", "").split("T")[0]
            if date_str not in by_day:
                by_day[date_str] = {
                    "date": date_str,
                    "acts": 0,
                    "pigs": 0,
                    "weight": 0.0
                }
            by_day[date_str]["acts"] += 1
            by_day[date_str]["pigs"] += act.get("left_count", 0) + act.get("right_count", 0)
            by_day[date_str]["weight"] += act.get("total_weight", 0.0)
        
        return {
            "period": {
                "start": start_dt.date().isoformat(),
                "end": end_dt.date().isoformat()
            },
            "total_acts": total_acts,
            "total_pigs": total_pigs,
            "total_weight": round(total_weight, 1),
            "avg_weight": round(avg_weight, 2),
            "avg_duration_sec": int(avg_duration),
            "by_day": sorted(by_day.values(), key=lambda x: x["date"])
        }
        
    except Exception as e:
        logger.error(f"Error getting weighing stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/acts/{act_id}")
async def get_weighing_act(act_id: int):
    """
    Получить детальную информацию об акте взвешивания
    
    Response:
    {
        "id": 1,
        "started_at": "2025-11-06T12:30:00",
        "ended_at": "2025-11-06T12:45:30",
        "duration_sec": 930,
        "left_count": 25,
        "right_count": 23,
        "peak_count": 14,
        "total_weight": 1850.5,
        "avg_weight": 132.2,
        "crossings": [
            {
                "id": 1,
                "timestamp": "2025-11-06T12:30:05",
                "direction": "left",
                "pig_id": 1,
                "x": 0.25,
                "y": 0.45
            }
        ]
    }
    """
    try:
        db = get_database_manager()
        
        # Получаем акт
        act = db.get_act_by_id(act_id)
        if not act:
            raise HTTPException(status_code=404, detail=f"Act {act_id} not found")
        
        # Получаем пересечения
        crossings = db.get_crossings_by_act(act_id)
        
        return {
            "id": act.get("id"),
            "started_at": act.get("started_at"),
            "ended_at": act.get("ended_at"),
            "duration_sec": act.get("duration_sec", 0),
            "left_count": act.get("left_count", 0),
            "right_count": act.get("right_count", 0),
            "peak_count": act.get("peak_count", 0),
            "total_weight": round(act.get("total_weight", 0.0), 1),
            "avg_weight": round(act.get("avg_weight", 0.0), 2),
            "crossings": crossings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting act {act_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
