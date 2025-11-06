"""
API endpoints для управления актами взвешивания
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import os

from pig_tracking.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/weighing", tags=["weighing"])

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


@router.get("/acts")
async def get_weighing_acts(
    start_date: Optional[str] = Query(None, description="Дата начала (ISO format, по умолчанию сегодня)"),
    end_date: Optional[str] = Query(None, description="Дата окончания (ISO format, по умолчанию сегодня)"),
    stream_id: Optional[str] = Query(None, description="ID потока")
):
    """
    Получить список актов взвешивания за период
    
    Соответствует спецификации: Требование 9.1
    """
    try:
        db = get_db_manager()
        
        # Если даты не указаны, берем сегодня
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        
        acts = db.get_acts_by_period(start, end)
        
        # Фильтрация по stream_id если указан
        if stream_id:
            acts = [a for a in acts if a.get('stream_id') == stream_id]
        
        return {
            "acts": acts,  # Уже словари из БД
            "total": len(acts),
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat()
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат даты: {e}")
    except Exception as e:
        logger.error(f"Error getting weighing acts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_weighing_stats(
    start_date: Optional[str] = Query(None, description="Дата начала (ISO format)"),
    end_date: Optional[str] = Query(None, description="Дата окончания (ISO format)"),
    stream_id: Optional[str] = Query(None, description="ID потока")
):
    """
    Получить агрегированную статистику актов взвешивания
    
    Соответствует спецификации: Требование 9.3
    """
    try:
        db = get_db_manager()
        
        # Если даты не указаны, берем последние 7 дней
        if start_date and end_date:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()
            start = end - timedelta(days=7)
        
        acts = db.get_acts_by_period(start, end)
        
        # Фильтрация по stream_id если указан
        if stream_id:
            acts = [a for a in acts if a.get('stream_id') == stream_id]
        
        # Вычисление статистики
        total_acts = len(acts)
        total_crossings = sum(a.get('left_count', 0) + a.get('right_count', 0) for a in acts)
        total_weight = sum(a.get('total_weight') or 0 for a in acts)
        avg_weight = sum(a.get('avg_weight') or 0 for a in acts) / total_acts if total_acts > 0 else 0
        
        return {
            "total_acts": total_acts,
            "total_crossings": total_crossings,
            "total_weight": round(total_weight, 1),
            "avg_weight": round(avg_weight, 1),
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat()
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат даты: {e}")
    except Exception as e:
        logger.error(f"Error getting weighing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/manual/save")
async def save_manual_weighing(data: Dict[str, Any]):
    """
    Сохранение ручного акта взвешивания
    
    Используется из Live панели для ручного ввода данных
    """
    try:
        db = get_db_manager()
        
        # Валидация данных
        required_fields = ['count', 'total_weight']
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Отсутствует обязательное поле: {field}"
                )
        
        count = int(data['count'])
        total_weight = float(data['total_weight'])
        
        if count <= 0 or total_weight <= 0:
            raise HTTPException(
                status_code=400,
                detail="Количество и вес должны быть больше нуля"
            )
        
        # Создание акта
        from pig_tracking.database_manager import WeighingAct
        
        act = WeighingAct(
            started_at=datetime.now(),
            ended_at=datetime.now(),
            duration_sec=0,
            left_count=count,
            right_count=0,
            peak_count=count,
            total_weight=total_weight,
            avg_weight=round(total_weight / count, 2),
            stream_id=data.get('stream_id', 'manual'),
            video_file=None
        )
        
        # Сохранение в БД
        act_id = db.save_weighing_act(act)
        
        logger.info(f"Manual weighing act saved: {act_id}")
        
        return {
            "status": "success",
            "act_id": act_id,
            "message": "Акт взвешивания сохранен"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving manual weighing: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/acts/{act_id}")
async def get_weighing_act_by_id(act_id: int):
    """
    Получить детальную информацию об акте взвешивания
    
    Args:
        act_id: ID акта
        
    Returns:
        Детали акта с пересечениями
    """
    try:
        db = get_db_manager()
        
        # Получить акт
        result = db.client.table("weighing_acts").select("*").eq("id", act_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Акт {act_id} не найден")
        
        act = result.data[0]
        
        # Получить пересечения для этого акта
        crossings_result = db.client.table("crossings").select("*").eq("act_id", act_id).execute()
        crossings = crossings_result.data if crossings_result.data else []
        
        act['crossings'] = crossings
        
        return act
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting act {act_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
