"""
Модуль для работы с базой данных Supabase.
Сохранение и получение данных о пересечениях и актах взвешивания.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict

try:
    from supabase import create_client, Client
    HAVE_SUPABASE = True
except ImportError:
    HAVE_SUPABASE = False
    Client = None

from pig_tracking.act_detector import WeighingAct
from pig_tracking.crossing_counter import CrossingEvent

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Менеджер для работы с базой данных Supabase.
    
    Таблицы:
    - weighing_acts: акты взвешивания
    - crossings: пересечения линий
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ):
        if not HAVE_SUPABASE:
            raise ImportError(
                "supabase-py не установлен. Установите: pip install supabase"
            )
        
        # Получаем параметры из переменных окружения
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL и SUPABASE_KEY должны быть установлены в .env "
                "или переданы в конструктор"
            )
        
        # Создаем клиент
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"✅ Подключение к Supabase: {self.supabase_url}")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Supabase: {e}")
            raise
    
    def save_crossing(
        self,
        crossing: CrossingEvent,
        act_id: Optional[int] = None,
        stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Сохраняет событие пересечения линии в базу данных.
        
        Args:
            crossing: событие пересечения
            act_id: ID акта взвешивания (опционально)
            stream_id: ID видеопотока (опционально)
            
        Returns:
            Сохраненная запись
        """
        try:
            data = {
                "act_id": act_id,
                "stream_id": stream_id,
                "track_id": crossing.track_id,
                "side": crossing.side,
                "mode": crossing.mode,
                "x": float(crossing.x),
                "y": float(crossing.y),
                "timestamp": datetime.fromtimestamp(crossing.timestamp).isoformat()
            }
            
            result = self.client.table("crossings").insert(data).execute()
            logger.debug(f"Сохранено пересечение: track_id={crossing.track_id}, side={crossing.side}")
            return result.data[0] if result.data else {}
            
        except Exception as e:
            logger.error(f"Ошибка сохранения пересечения: {e}")
            raise
    
    def save_weighing_act(
        self,
        act: WeighingAct,
        stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Сохраняет акт взвешивания в базу данных.
        
        Args:
            act: акт взвешивания
            stream_id: ID видеопотока (опционально)
            
        Returns:
            Сохраненная запись с ID
        """
        try:
            data = {
                "stream_id": stream_id,
                "started_at": datetime.fromtimestamp(act.started_at).isoformat(),
                "ended_at": datetime.fromtimestamp(act.ended_at).isoformat() if act.ended_at else None,
                "left_count": act.left_count,
                "right_count": act.right_count,
                "peak_count": act.peak_count,
                "seen_total": len(act.seen_labels),
                "duration": act.duration if act.ended_at else None
            }
            
            result = self.client.table("weighing_acts").insert(data).execute()
            saved_act = result.data[0] if result.data else {}
            
            logger.info(
                f"✅ Сохранен акт #{act.act_id}: "
                f"left={act.left_count}, right={act.right_count}, "
                f"peak={act.peak_count}, duration={act.duration:.1f}s"
            )
            
            return saved_act
            
        except Exception as e:
            logger.error(f"Ошибка сохранения акта: {e}")
            raise
    
    def get_acts_by_period(
        self,
        start_date: datetime,
        end_date: datetime,
        stream_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Получает акты взвешивания за указанный период.
        
        Args:
            start_date: начало периода
            end_date: конец периода
            stream_id: фильтр по ID потока (опционально)
            
        Returns:
            Список актов
        """
        try:
            query = self.client.table("weighing_acts").select("*")
            
            # Фильтр по дате
            query = query.gte("started_at", start_date.isoformat())
            query = query.lte("started_at", end_date.isoformat())
            
            # Фильтр по stream_id
            if stream_id:
                query = query.eq("stream_id", stream_id)
            
            # Сортировка по дате
            query = query.order("started_at", desc=False)
            
            result = query.execute()
            acts = result.data if result.data else []
            
            logger.info(f"Получено {len(acts)} актов за период {start_date.date()} - {end_date.date()}")
            return acts
            
        except Exception as e:
            logger.error(f"Ошибка получения актов: {e}")
            raise
    
    def get_crossings_by_act(self, act_id: int) -> List[Dict[str, Any]]:
        """
        Получает все пересечения для указанного акта.
        
        Args:
            act_id: ID акта
            
        Returns:
            Список пересечений
        """
        try:
            result = self.client.table("crossings")\
                .select("*")\
                .eq("act_id", act_id)\
                .order("timestamp", desc=False)\
                .execute()
            
            crossings = result.data if result.data else []
            logger.debug(f"Получено {len(crossings)} пересечений для акта #{act_id}")
            return crossings
            
        except Exception as e:
            logger.error(f"Ошибка получения пересечений: {e}")
            raise
    
    def get_stats_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Получает сводную статистику за период.
        
        Args:
            start_date: начало периода (по умолчанию - сегодня)
            end_date: конец периода (по умолчанию - сегодня)
            
        Returns:
            Словарь со статистикой
        """
        try:
            # По умолчанию - сегодня
            if not start_date:
                start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if not end_date:
                end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Получаем акты
            acts = self.get_acts_by_period(start_date, end_date)
            
            # Вычисляем статистику
            total_acts = len(acts)
            total_left = sum(act.get("left_count", 0) for act in acts)
            total_right = sum(act.get("right_count", 0) for act in acts)
            total_seen = sum(act.get("seen_total", 0) for act in acts)
            avg_duration = sum(act.get("duration", 0) for act in acts) / total_acts if total_acts > 0 else 0
            max_peak = max((act.get("peak_count", 0) for act in acts), default=0)
            
            summary = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "total_acts": total_acts,
                "total_left_in": total_left,
                "total_right_in": total_right,
                "total_seen": total_seen,
                "avg_duration": round(avg_duration, 1),
                "max_peak": max_peak,
                "acts": acts
            }
            
            logger.info(
                f"Статистика за период: {total_acts} актов, "
                f"left={total_left}, right={total_right}, seen={total_seen}"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Проверяет подключение к базе данных.
        
        Returns:
            True если подключение успешно
        """
        try:
            # Пробуем выполнить простой запрос
            result = self.client.table("weighing_acts").select("id").limit(1).execute()
            logger.info("✅ Подключение к базе данных успешно")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к базе данных: {e}")
            return False
    
    def get_weighing_acts(
        self,
        limit: int = 50,
        offset: int = 0,
        stream_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получает список актов с пагинацией и фильтрами"""
        try:
            query = self.client.table("weighing_acts").select("*")
            
            if stream_id:
                query = query.eq("stream_id", stream_id)
            if date_from:
                query = query.gte("started_at", date_from)
            if date_to:
                query = query.lte("started_at", date_to)
            
            query = query.order("started_at", desc=True).limit(limit).offset(offset)
            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting weighing acts: {e}")
            return []
    
    def count_weighing_acts(
        self,
        stream_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> int:
        """Подсчитывает общее количество актов"""
        try:
            query = self.client.table("weighing_acts").select("id", count="exact")
            
            if stream_id:
                query = query.eq("stream_id", stream_id)
            if date_from:
                query = query.gte("started_at", date_from)
            if date_to:
                query = query.lte("started_at", date_to)
            
            result = query.execute()
            return result.count if hasattr(result, 'count') else 0
        except Exception as e:
            logger.error(f"Error counting acts: {e}")
            return 0
    
    def get_weighing_act_by_id(self, act_id: int) -> Optional[Dict[str, Any]]:
        """Получает акт по ID"""
        try:
            result = self.client.table("weighing_acts")\
                .select("*")\
                .eq("id", act_id)\
                .single()\
                .execute()
            return result.data if result.data else None
        except Exception as e:
            logger.error(f"Error getting act {act_id}: {e}")
            return None
    
    def get_latest_weighing_act(self, stream_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Получает последний акт"""
        try:
            query = self.client.table("weighing_acts").select("*")
            if stream_id:
                query = query.eq("stream_id", stream_id)
            
            result = query.order("started_at", desc=True).limit(1).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting latest act: {e}")
            return None
    
    def get_weighing_stats(
        self,
        stream_id: Optional[str] = None,
        date_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """Получает статистику по актам"""
        try:
            query = self.client.table("weighing_acts").select("*")
            
            if stream_id:
                query = query.eq("stream_id", stream_id)
            if date_from:
                query = query.gte("started_at", date_from)
            
            result = query.execute()
            acts = result.data if result.data else []
            
            total_acts = len(acts)
            total_pigs = sum(act.get("left_count", 0) + act.get("right_count", 0) for act in acts)
            total_weight = sum(act.get("total_weight", 0) for act in acts)
            avg_weight = total_weight / total_pigs if total_pigs > 0 else 0
            avg_duration = sum(act.get("duration", 0) for act in acts) / total_acts if total_acts > 0 else 0
            
            # Группировка по часам
            acts_by_hour = {}
            for act in acts:
                if act.get("started_at"):
                    hour = datetime.fromisoformat(act["started_at"]).strftime("%H")
                    acts_by_hour[hour] = acts_by_hour.get(hour, 0) + 1
            
            return {
                "total_acts": total_acts,
                "total_pigs": total_pigs,
                "total_weight": round(total_weight, 1),
                "avg_weight": round(avg_weight, 2),
                "avg_duration_sec": round(avg_duration, 1),
                "acts_by_hour": acts_by_hour
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_acts": 0,
                "total_pigs": 0,
                "total_weight": 0.0,
                "avg_weight": 0.0,
                "avg_duration_sec": 0.0,
                "acts_by_hour": {}
            }
    
    def create_tables_if_not_exist(self):
        """
        Создает таблицы, если они не существуют.
        
        Примечание: В Supabase таблицы обычно создаются через миграции SQL.
        Этот метод только для справки - SQL код для создания таблиц.
        """
        sql_weighing_acts = """
        CREATE TABLE IF NOT EXISTS weighing_acts (
            id SERIAL PRIMARY KEY,
            stream_id VARCHAR(255),
            started_at TIMESTAMP NOT NULL,
            ended_at TIMESTAMP,
            left_count INTEGER DEFAULT 0,
            right_count INTEGER DEFAULT 0,
            peak_count INTEGER DEFAULT 0,
            seen_total INTEGER DEFAULT 0,
            duration FLOAT,
            total_weight FLOAT DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_weighing_acts_started_at 
            ON weighing_acts(started_at);
        CREATE INDEX IF NOT EXISTS idx_weighing_acts_stream_id 
            ON weighing_acts(stream_id);
        """
        
        sql_crossings = """
        CREATE TABLE IF NOT EXISTS crossings (
            id SERIAL PRIMARY KEY,
            act_id INTEGER REFERENCES weighing_acts(id) ON DELETE CASCADE,
            stream_id VARCHAR(255),
            track_id INTEGER NOT NULL,
            side VARCHAR(10) NOT NULL,
            mode VARCHAR(10) NOT NULL,
            x FLOAT NOT NULL,
            y FLOAT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_crossings_act_id 
            ON crossings(act_id);
        CREATE INDEX IF NOT EXISTS idx_crossings_timestamp 
            ON crossings(timestamp);
        CREATE INDEX IF NOT EXISTS idx_crossings_track_id 
            ON crossings(track_id);
        """
        
        logger.info("SQL для создания таблиц:")
        logger.info(sql_weighing_acts)
        logger.info(sql_crossings)
        logger.info("Выполните эти SQL команды в Supabase SQL Editor")
        
        return {
            "weighing_acts": sql_weighing_acts,
            "crossings": sql_crossings
        }
