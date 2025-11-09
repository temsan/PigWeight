"""
API endpoints для получения статистики
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
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
async def get_current_stats(stream_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    Получить текущую статистику системы
    
    Соответствует спецификации: Требование 9.2
    Заменяет старый /api/metrics/current
    
    Args:
        stream_id: ID потока (опционально)
    """
    try:
        db = get_db_manager()
        
        # Получить статистику из БД
        stats = db.get_stats_summary(stream_id=stream_id)
        
        # Получить последний активный акт
        end = datetime.now()
        start = end - timedelta(hours=1)  # Последний час
        recent_acts = db.get_acts_by_period(start, end, stream_id=stream_id)
        
        active_act = None
        if recent_acts:
            # Последний акт считаем активным если он был менее 5 минут назад
            last_act = recent_acts[-1]
            # get_acts_by_period возвращает словари, не объекты
            ended_at_str = last_act.get("ended_at")
            if ended_at_str:
                try:
                    # Парсим строку ISO формата в datetime
                    if isinstance(ended_at_str, str):
                        # Убираем timezone если есть, для простоты сравнения
                        ended_at_str_clean = ended_at_str.replace('Z', '').split('+')[0].split('.')[0]
                        ended_at = datetime.fromisoformat(ended_at_str_clean)
                    else:
                        ended_at = ended_at_str
                    
                    # Проверяем, был ли акт менее 5 минут назад
                    time_diff = (datetime.now() - ended_at).total_seconds()
                    if time_diff < 300 and time_diff >= 0:
                        started_at_str = last_act.get("started_at")
                        if started_at_str:
                            if isinstance(started_at_str, str):
                                started_at_str_clean = started_at_str.replace('Z', '').split('+')[0].split('.')[0]
                                started_at = datetime.fromisoformat(started_at_str_clean)
                            else:
                                started_at = started_at_str
                            
                            duration_sec = (ended_at - started_at).total_seconds() if ended_at and started_at else 0
                            
                            active_act = {
                                "id": last_act.get("id"),
                                "act_id": last_act.get("id"),
                                "started_at": started_at.isoformat() if started_at else None,
                                "ended_at": ended_at.isoformat() if ended_at else None,
                                "duration_sec": duration_sec,
                                "left_count": last_act.get("left_count", 0),
                                "right_count": last_act.get("right_count", 0),
                                "peak_count": last_act.get("peak_count", 0),
                                "total_weight": last_act.get("total_weight", 0.0),
                                "avg_weight": last_act.get("avg_weight", 0.0)
                            }
                except Exception as parse_error:
                    logger.warning(f"Ошибка парсинга дат в акте: {parse_error}")
        
        # Формируем ответ в формате, ожидаемом мобильным дашбордом
        return {
            "stream_id": stream_id or "default",
            "current_count": stats.get("peak_count", stats.get("current_count", 0)),
            "total_weight": round(stats.get("total_weight", 0.0), 1),
            "avg_weight": round(stats.get("avg_weight", 0.0), 2),
            "left_count": stats.get("left_count", 0),
            "right_count": stats.get("right_count", 0),
            "active_act": active_act,
            "auto_manual": "auto",  # По умолчанию автоматический режим
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting current stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
