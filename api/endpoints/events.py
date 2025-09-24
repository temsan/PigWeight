"""
API endpoints для работы с событиями системы PigWeight.
Предоставляет доступ к журналу событий, статистике и экспорту данных.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Query, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

from services.event_logger import get_event_logger, EventData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])

@router.get("/streams")
async def get_active_streams():
    """Получает список активных потоков с событиями"""
    try:
        event_logger = get_event_logger()
        
        streams = []
        for stream_id in event_logger.stream_events.keys():
            stats = event_logger.get_stream_stats(stream_id)
            streams.append({
                'stream_id': stream_id,
                'total_events': stats['total_events'],
                'peak_count': stats['peak_count'],
                'event_types': stats['event_types'],
                'latest_event': stats['latest_event']
            })
        
        return {
            'success': True,
            'streams': streams
        }
        
    except Exception as e:
        logger.error(f"Error getting active streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{stream_id}")
async def get_stream_events(
    stream_id: str,
    event_type: Optional[str] = Query(None, description="Тип события: line_crossing, peak_count, activity_spike"),
    limit: Optional[int] = Query(50, description="Максимальное количество событий"),
    since_hours: Optional[int] = Query(None, description="События за последние N часов")
):
    """Получает события для указанного потока"""
    try:
        event_logger = get_event_logger()
        
        # Вычисляем временную метку
        since_timestamp = None
        if since_hours:
            since_timestamp = (datetime.now() - timedelta(hours=since_hours)).timestamp()
        
        events = event_logger.get_events(
            stream_id=stream_id,
            event_type=event_type,
            limit=limit,
            since_timestamp=since_timestamp
        )
        
        # Преобразуем события в словари для JSON
        events_data = []
        for event in events:
            event_dict = event.to_dict()
            # Добавляем человекочитаемое время
            event_dict['datetime'] = datetime.fromtimestamp(event.timestamp).isoformat()
            events_data.append(event_dict)
        
        return {
            'success': True,
            'stream_id': stream_id,
            'events': events_data,
            'total_count': len(events_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting events for stream {stream_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{stream_id}/stats")
async def get_stream_statistics(stream_id: str):
    """Получает статистику по потоку"""
    try:
        event_logger = get_event_logger()
        stats = event_logger.get_stream_stats(stream_id)
        
        # Добавляем дополнительную статистику
        if stats['latest_event']:
            stats['latest_event_datetime'] = datetime.fromtimestamp(stats['latest_event']).isoformat()
        
        return {
            'success': True,
            'stream_id': stream_id,
            'statistics': stats
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics for stream {stream_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{stream_id}/frame/{event_id}")
async def get_event_frame(stream_id: str, event_id: str):
    """Получает кадр, связанный с событием"""
    try:
        event_logger = get_event_logger()
        
        # Находим событие
        events = event_logger.get_events(stream_id)
        target_event = None
        
        for event in events:
            if event.event_id == event_id:
                target_event = event
                break
        
        if not target_event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        if not target_event.frame_path:
            raise HTTPException(status_code=404, detail="Frame not available for this event")
        
        frame_path = Path(target_event.frame_path)
        if not frame_path.exists():
            raise HTTPException(status_code=404, detail="Frame file not found")
        
        return FileResponse(
            path=str(frame_path),
            media_type="image/jpeg",
            filename=f"{event_id}.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting frame for event {event_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{stream_id}/cleanup")
async def cleanup_stream_events(
    stream_id: str,
    max_age_hours: int = Query(24, description="Максимальный возраст событий в часах")
):
    """Очищает старые события для потока"""
    try:
        event_logger = get_event_logger()
        await event_logger.cleanup_old_events(max_age_hours)
        
        return {
            'success': True,
            'message': f'Cleaned up events older than {max_age_hours} hours'
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{stream_id}/export")
async def export_stream_events(
    stream_id: str,
    format: str = Query("json", description="Формат экспорта: json, csv"),
    event_type: Optional[str] = Query(None, description="Тип события для экспорта"),
    since_hours: Optional[int] = Query(None, description="События за последние N часов")
):
    """Экспортирует события потока в различных форматах"""
    try:
        event_logger = get_event_logger()
        
        # Получаем события
        since_timestamp = None
        if since_hours:
            since_timestamp = (datetime.now() - timedelta(hours=since_hours)).timestamp()
        
        events = event_logger.get_events(
            stream_id=stream_id,
            event_type=event_type,
            since_timestamp=since_timestamp
        )
        
        if format.lower() == "json":
            # JSON экспорт
            events_data = []
            for event in events:
                event_dict = event.to_dict()
                event_dict['datetime'] = datetime.fromtimestamp(event.timestamp).isoformat()
                events_data.append(event_dict)
            
            return JSONResponse(
                content={
                    'stream_id': stream_id,
                    'export_format': 'json',
                    'events': events_data,
                    'total_count': len(events_data),
                    'exported_at': datetime.now().isoformat()
                }
            )
        
        elif format.lower() == "csv":
            # CSV экспорт
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Заголовки
            writer.writerow([
                'event_id', 'stream_id', 'event_type', 'timestamp', 'datetime',
                'pig_count', 'confidence', 'frame_path', 'metadata'
            ])
            
            # Данные
            for event in events:
                writer.writerow([
                    event.event_id,
                    event.stream_id,
                    event.event_type,
                    event.timestamp,
                    datetime.fromtimestamp(event.timestamp).isoformat(),
                    event.pig_count,
                    event.confidence,
                    event.frame_path or '',
                    str(event.metadata) if event.metadata else ''
                ])
            
            csv_content = output.getvalue()
            output.close()
            
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={stream_id}_events.csv"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting events for stream {stream_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{stream_id}")
async def delete_stream_events(stream_id: str):
    """Удаляет все события для потока"""
    try:
        event_logger = get_event_logger()
        
        if stream_id in event_logger.stream_events:
            # Удаляем связанные кадры
            events = list(event_logger.stream_events[stream_id])
            for event in events:
                if event.frame_path and Path(event.frame_path).exists():
                    try:
                        Path(event.frame_path).unlink()
                    except Exception as e:
                        logger.error(f"Error deleting frame {event.frame_path}: {e}")
            
            # Очищаем события из памяти
            del event_logger.stream_events[stream_id]
            
            # Очищаем связанные данные
            if stream_id in event_logger.stream_peaks:
                del event_logger.stream_peaks[stream_id]
            if stream_id in event_logger.stream_history:
                del event_logger.stream_history[stream_id]
            
            # Удаляем файл событий
            events_file = event_logger.events_dir / f"{stream_id}_events.jsonl"
            if events_file.exists():
                events_file.unlink()
        
        return {
            'success': True,
            'message': f'All events for stream {stream_id} have been deleted'
        }
        
    except Exception as e:
        logger.error(f"Error deleting events for stream {stream_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))