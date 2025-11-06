"""
API endpoints для получения статистики
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime, timedelta
import logging
import os

from pig_tracking.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stats", tags=["stats"])

# Глобальный экземпляр DatabaseManager
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Получить экземпляр DatabaseManager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_KEY")
        )
    return _db_manager


@router.get("/current")
async def get_current_stats() -> Dict[str, Any]:
    """
    Получить текущую статистику системы
    
    Соответствует спецификации: Требование 9.2
    Заменяет старый /api/metrics/current
    """
    try:
        db = get_db_manager()
        
        # Получить статистику из БД
        stats = db.get_stats_summary()
        
        # Получить последний активный акт
        end = datetime.now()
        start = end - timedelta(hours=1)  # Последний час
        recent_acts = db.get_acts_by_period(start, end)
        
        active_act = None
        if recent_acts:
            # Последний акт считаем активным если он был менее 5 минут назад
            last_act = recent_acts[-1]
            if (datetime.now() - last_act.ended_at).total_seconds() < 300:
                active_act = {
                    "act_id": last_act.act_id,
                    "started_at": last_act.started_at.isoformat(),
                    "duration_sec": last_act.duration_sec,
                    "left_count": last_act.left_count,
                    "right_count": last_act.right_count,
                    "peak_count": last_act.peak_count,
                    "total_weight": last_act.total_weight,
                    "avg_weight": last_act.avg_weight
                }
        
        return {
            "current_count": stats.get("current_count", 0),
            "left_count": stats.get("left_count", 0),
            "right_count": stats.get("right_count", 0),
            "total_weight": stats.get("total_weight", 0),
            "avg_weight": stats.get("avg_weight", 0),
            "active_act": active_act,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting current stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
