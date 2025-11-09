"""
Metrics Service
Бизнес-логика вычисления метрик и статистики
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsService:
    """Сервис для вычисления метрик и статистики"""
    
    def __init__(self, stream_service, act_service, database_manager=None):
        """
        Args:
            stream_service: Сервис управления потоками
            act_service: Сервис управления актами
            database_manager: Менеджер базы данных (опционально)
        """
        self.stream_service = stream_service
        self.act_service = act_service
        self.db = database_manager
    
    def get_current_metrics(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить текущие метрики для дашборда
        
        Комбинирует данные из:
        - Real-time данные из потока (текущий счёт)
        - Исторические данные из БД (статистика)
        - Активный акт (если есть)
        
        Args:
            stream_id: ID потока (опционально)
        
        Returns:
            Словарь с метриками
        """
        # Если stream_id не указан, берём первый активный
        if not stream_id:
            streams = self.stream_service.get_active_streams()
            if streams:
                stream_id = streams[0]
            else:
                stream_id = "default"
        
        # Получаем поток
        stream = self.stream_service.get_stream(stream_id)
        
        # Получаем статистику из БД
        db_stats = {}
        if self.db:
            try:
                db_stats = self.db.get_stats_summary(stream_id=stream_id)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка получения статистики из БД: {e}")
        
        # Получаем активный акт
        active_act = None
        auto_manual = "auto"
        
        if stream:
            active_act = self.act_service.get_active_act(stream)
            auto_manual = "auto" if getattr(stream, 'auto_fix_enabled', True) else "manual"
        
        # Формируем ответ
        return {
            "stream_id": stream_id,
            "current_count": db_stats.get("total_pigs", 0) if db_stats else 0,
            "total_weight": round(db_stats.get("total_weight", 0.0), 1) if db_stats else 0.0,
            "avg_weight": round(db_stats.get("avg_weight", 0.0), 2) if db_stats else 0.0,
            "left_count": db_stats.get("left_count", 0) if db_stats else 0,
            "right_count": db_stats.get("right_count", 0) if db_stats else 0,
            "active_act": active_act,
            "auto_manual": auto_manual,
            "timestamp": datetime.now().isoformat(),
            "database_available": bool(self.db and db_stats)
        }
    
    def calculate_act_metrics(self, act_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вычислить метрики для акта
        
        Args:
            act_data: Данные акта
        
        Returns:
            Словарь с вычисленными метриками
        """
        left_count = act_data.get('left_count', 0)
        right_count = act_data.get('right_count', 0)
        total_pigs = act_data.get('total_pigs', left_count + right_count)
        total_weight = act_data.get('total_weight', 0.0)
        
        # Вычисляем средний вес
        avg_weight = 0.0
        if total_pigs > 0 and total_weight > 0:
            avg_weight = total_weight / total_pigs
        
        # Вычисляем баланс потоков
        flow_balance = left_count - right_count
        flow_balance_pct = 0.0
        if total_pigs > 0:
            flow_balance_pct = (flow_balance / total_pigs) * 100
        
        return {
            "total_pigs": total_pigs,
            "left_count": left_count,
            "right_count": right_count,
            "total_weight": round(total_weight, 1),
            "avg_weight": round(avg_weight, 2),
            "flow_balance": flow_balance,
            "flow_balance_pct": round(flow_balance_pct, 1),
            "duration_sec": act_data.get('duration_sec', 0),
            "peak_count": act_data.get('peak_count', 0)
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Получить состояние системы
        
        Returns:
            Словарь с состоянием компонентов
        """
        # Проверяем БД
        db_status = "disconnected"
        if self.db:
            try:
                if self.db.test_connection():
                    db_status = "connected"
                else:
                    db_status = "error"
            except Exception as e:
                db_status = f"error: {str(e)}"
        
        # Проверяем потоки
        active_streams = self.stream_service.get_active_streams()
        stream_status = "active" if active_streams else "idle"
        
        # Определяем общий статус
        overall_status = "healthy"
        if db_status != "connected":
            overall_status = "degraded"
        if not active_streams and db_status != "connected":
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "components": {
                "database": db_status,
                "stream_manager": stream_status,
                "active_streams": len(active_streams)
            },
            "streams": active_streams,
            "timestamp": datetime.now().isoformat()
        }
    
    def aggregate_daily_stats(
        self,
        acts: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Агрегировать статистику по актам за день
        
        Args:
            acts: Список актов
        
        Returns:
            Агрегированная статистика
        """
        if not acts:
            return {
                "total_acts": 0,
                "total_pigs": 0,
                "total_weight": 0.0,
                "avg_weight": 0.0,
                "avg_duration": 0.0,
                "max_peak": 0
            }
        
        total_acts = len(acts)
        total_pigs = sum(act.get('left_count', 0) + act.get('right_count', 0) for act in acts)
        total_weight = sum(act.get('total_weight', 0.0) for act in acts)
        total_duration = sum(act.get('duration_sec', 0) for act in acts)
        max_peak = max((act.get('peak_count', 0) for act in acts), default=0)
        
        avg_weight = 0.0
        if total_pigs > 0 and total_weight > 0:
            avg_weight = total_weight / total_pigs
        
        avg_duration = 0.0
        if total_acts > 0:
            avg_duration = total_duration / total_acts
        
        return {
            "total_acts": total_acts,
            "total_pigs": total_pigs,
            "total_weight": round(total_weight, 1),
            "avg_weight": round(avg_weight, 2),
            "avg_duration": round(avg_duration, 1),
            "max_peak": max_peak,
            "left_count": sum(act.get('left_count', 0) for act in acts),
            "right_count": sum(act.get('right_count', 0) for act in acts)
        }
