"""
API endpoints для работы с журналом событий.
Предоставляет доступ к истории пересечений линий, пиковым значениям и группировке по датам.
"""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Импорт системы событий
try:
    from services.event_logger import get_event_logger
    HAVE_EVENT_LOGGER = True
except ImportError:
    HAVE_EVENT_LOGGER = False
    logger.warning("EventLogger не доступен")


@router.get("/events/{stream_id}")
async def get_stream_events(
    stream_id: str,
    event_type: Optional[str] = Query(None, description="Фильтр по типу события"),
    limit: int = Query(100, ge=1, le=1000, description="Максимальное количество событий"),
    since: Optional[float] = Query(None, description="Timestamp начала выборки")
):
    """Получить журнал событий для потока"""
    try:
        if not HAVE_EVENT_LOGGER:
            return JSONResponse({"error": "Система журналирования недоступна"}, status_code=503)
        
        event_logger = get_event_logger()
        events = event_logger.get_events(
            stream_id=stream_id,
            event_type=event_type,
            limit=limit,
            since_timestamp=since
        )
        
        # Преобразуем события в словари
        events_data = [e.to_dict() for e in events]
        
        # Добавляем читаемые метки времени
        for e in events_data:
            e['timestamp_str'] = datetime.fromtimestamp(e['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        stats = event_logger.get_stream_stats(stream_id)
        
        return {
            "stream_id": stream_id,
            "events": events_data,
            "total": len(events_data),
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting events: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/events/{stream_id}/stats")
async def get_stream_stats(stream_id: str):
    """Получить статистику по потоку"""
    try:
        if not HAVE_EVENT_LOGGER:
            return JSONResponse({"error": "Система журналирования недоступна"}, status_code=503)
        
        event_logger = get_event_logger()
        stats = event_logger.get_stream_stats(stream_id)
        
        return {
            "stream_id": stream_id,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/events/{stream_id}/grouped")
async def get_grouped_events(
    stream_id: str,
    date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Дата в формате YYYY-MM-DD")
):
    """Получить события, сгруппированные по дате и актам"""
    try:
        if not HAVE_EVENT_LOGGER:
            return JSONResponse({"error": "Система журналирования недоступна"}, status_code=503)
        
        event_logger = get_event_logger()
        
        # Получаем все события
        all_events = event_logger.get_events(stream_id=stream_id)
        
        # Группируем по дате
        grouped = {}
        for event in all_events:
            event_date = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d')
            
            # Фильтруем по дате если указана
            if date and event_date != date:
                continue
            
            if event_date not in grouped:
                grouped[event_date] = {
                    'date': event_date,
                    'events': [],
                    'crossings': {'left': {'enter': 0, 'exit': 0}, 'right': {'enter': 0, 'exit': 0}},
                    'peak_count': 0
                }
            
            event_dict = event.to_dict()
            event_dict['timestamp_str'] = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
            grouped[event_date]['events'].append(event_dict)
            
            # Считаем пересечения
            if event.event_type == 'line_crossing':
                side = event.metadata.get('side') if event.metadata else None
                direction = event.metadata.get('direction') if event.metadata else None
                if side and direction:
                    grouped[event_date]['crossings'][side][direction] += 1
            
            # Обновляем пик
            if event.event_type == 'peak_count':
                grouped[event_date]['peak_count'] = max(grouped[event_date]['peak_count'], event.pig_count)
        
        # Преобразуем в список и сортируем по дате
        result = sorted(grouped.values(), key=lambda x: x['date'], reverse=True)
        
        return {
            "stream_id": stream_id,
            "filter_date": date,
            "groups": result,
            "total_groups": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error grouping events: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/events/{stream_id}/export")
async def export_events(
    stream_id: str,
    date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Дата для экспорта"),
    format: str = Query("json", regex="^(json|csv)$", description="Формат экспорта")
):
    """Экспорт событий в JSON или CSV"""
    try:
        if not HAVE_EVENT_LOGGER:
            return JSONResponse({"error": "Система журналирования недоступна"}, status_code=503)
        
        event_logger = get_event_logger()
        events = event_logger.get_events(stream_id=stream_id)
        
        # Фильтруем по дате если указана
        if date:
            events = [
                e for e in events 
                if datetime.fromtimestamp(e.timestamp).strftime('%Y-%m-%d') == date
            ]
        
        if format == "csv":
            import io
            import csv
            from fastapi.responses import StreamingResponse
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Заголовки
            writer.writerow(['Timestamp', 'Event Type', 'Pig Count', 'Side', 'Movement', 'Confidence'])
            
            # Данные
            for e in events:
                timestamp_str = datetime.fromtimestamp(e.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([
                    timestamp_str,
                    e.event_type,
                    e.pig_count,
                    e.side or '',
                    e.movement or '',
                    f"{e.confidence:.2f}"
                ])
            
            output.seek(0)
            filename = f"events_{stream_id}_{date or 'all'}.csv"
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        else:  # JSON
            events_data = [e.to_dict() for e in events]
            for e in events_data:
                e['timestamp_str'] = datetime.fromtimestamp(e['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                "stream_id": stream_id,
                "filter_date": date,
                "events": events_data,
                "total": len(events_data)
            }
        
    except Exception as e:
        logger.error(f"Error exporting events: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
