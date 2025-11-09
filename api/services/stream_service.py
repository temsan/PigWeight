"""
Stream Service
Бизнес-логика управления видео-потоками
"""

import logging
from typing import Dict, Optional, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class StreamService:
    """Сервис для управления видео-потоками"""
    
    def __init__(self, stream_manager):
        """
        Args:
            stream_manager: Глобальный менеджер потоков
        """
        self.stream_manager = stream_manager
    
    def get_active_streams(self) -> List[str]:
        """Получить список активных потоков"""
        if not self.stream_manager:
            return []
        return list(self.stream_manager.streams.keys())
    
    def get_stream(self, stream_id: str) -> Optional[Any]:
        """Получить поток по ID"""
        if not self.stream_manager:
            return None
        return self.stream_manager.streams.get(stream_id)
    
    def stream_exists(self, stream_id: str) -> bool:
        """Проверить существование потока"""
        if not self.stream_manager:
            return False
        return stream_id in self.stream_manager.streams
    
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Получить статус потока"""
        stream = self.get_stream(stream_id)
        if not stream:
            return {
                "exists": False,
                "running": False,
                "error": "Stream not found"
            }
        
        return {
            "exists": True,
            "running": getattr(stream, 'running', False),
            "stream_id": stream_id,
            "last_count": getattr(stream, 'last_count', 0),
            "left_in": getattr(stream, 'left_in', 0),
            "right_in": getattr(stream, 'right_in', 0),
            "total_crossings": getattr(stream, 'total_crossings', 0)
        }
    
    def get_all_streams_status(self) -> List[Dict[str, Any]]:
        """Получить статус всех потоков"""
        streams = self.get_active_streams()
        return [self.get_stream_status(sid) for sid in streams]
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Остановить поток"""
        stream = self.get_stream(stream_id)
        if not stream:
            logger.warning(f"Попытка остановить несуществующий поток: {stream_id}")
            return False
        
        try:
            if hasattr(stream, 'stop'):
                await stream.stop()
            logger.info(f"✅ Поток {stream_id} остановлен")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка остановки потока {stream_id}: {e}")
            return False
    
    def get_stream_metrics(self, stream_id: str) -> Dict[str, Any]:
        """Получить метрики потока для дашборда"""
        stream = self.get_stream(stream_id)
        if not stream:
            return {
                "stream_id": stream_id,
                "error": "Stream not found",
                "current_count": 0,
                "left_count": 0,
                "right_count": 0,
                "total_crossings": 0
            }
        
        return {
            "stream_id": stream_id,
            "current_count": getattr(stream, 'reported_count', 0),
            "left_count": getattr(stream, 'left_in', 0),
            "right_count": getattr(stream, 'right_in', 0),
            "total_crossings": getattr(stream, 'total_crossings', 0),
            "left_flow": getattr(stream, 'left_flow', 0),
            "right_flow": getattr(stream, 'right_flow', 0),
            "timestamp": datetime.now().isoformat()
        }
