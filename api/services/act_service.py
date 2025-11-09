"""
Act Service
Бизнес-логика управления актами взвешивания
"""

import logging
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ActService:
    """Сервис для управления актами взвешивания"""
    
    def __init__(self, database_manager=None):
        """
        Args:
            database_manager: Менеджер базы данных (опционально)
        """
        self.db = database_manager
    
    def get_active_act(self, stream) -> Optional[Dict[str, Any]]:
        """Получить активный акт из потока"""
        if not stream:
            return None
        
        if not hasattr(stream, 'current_act') or not stream.current_act:
            return None
        
        act = stream.current_act
        duration = (datetime.now() - act.get('started_at')).total_seconds()
        
        return {
            "id": act.get('id'),
            "started_at": act.get('started_at').isoformat() if act.get('started_at') else None,
            "duration_sec": duration,
            "pig_ids": act.get('pig_ids', []),
            "left_count": act.get('left_count', 0),
            "right_count": act.get('right_count', 0),
            "peak_count": act.get('peak_count', 0)
        }
    
    def finalize_act(self, stream, manual: bool = False) -> Optional[Dict[str, Any]]:
        """
        Завершить текущий акт
        
        Args:
            stream: Видео-поток
            manual: Ручное завершение (True) или автоматическое (False)
        
        Returns:
            Данные завершённого акта или None
        """
        if not stream or not hasattr(stream, 'current_act') or not stream.current_act:
            logger.warning("Нет активного акта для завершения")
            return None
        
        act = stream.current_act
        act['ended_at'] = datetime.now()
        duration = (act['ended_at'] - act['started_at']).total_seconds()
        
        # Формируем данные акта
        finalized_act = {
            "id": act.get('id'),
            "stream_id": stream.stream_id,
            "started_at": act['started_at'].isoformat(),
            "ended_at": act['ended_at'].isoformat(),
            "duration_sec": duration,
            "left_count": act.get('left_count', 0),
            "right_count": act.get('right_count', 0),
            "peak_count": act.get('peak_count', 0),
            "total_pigs": len(act.get('pig_ids', [])),
            "total_weight": act.get('total_weight', 0.0),
            "avg_weight": act.get('avg_weight', 0.0),
            "manual": manual
        }
        
        # Сохраняем в БД
        if self.db:
            try:
                from pig_tracking.database_manager import WeighingAct
                db_act = WeighingAct(
                    started_at=act['started_at'],
                    ended_at=act['ended_at'],
                    duration_sec=duration,
                    left_count=finalized_act['left_count'],
                    right_count=finalized_act['right_count'],
                    peak_count=finalized_act['peak_count'],
                    total_weight=finalized_act['total_weight'],
                    avg_weight=finalized_act['avg_weight'],
                    stream_id=stream.stream_id,
                    video_file=None
                )
                act_id = self.db.save_weighing_act(db_act)
                finalized_act['db_id'] = act_id
                logger.info(f"✅ Акт сохранён в БД: {act_id}")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения акта в БД: {e}")
        
        # Очищаем текущий акт
        stream.current_act = None
        
        # Вызываем финализацию в файлы (если есть)
        if hasattr(stream, '_finalize_act_to_files'):
            try:
                stream._finalize_act_to_files()
            except Exception as e:
                logger.error(f"❌ Ошибка финализации акта в файлы: {e}")
        
        logger.info(f"📌 Акт завершён {'вручную' if manual else 'автоматически'}: {finalized_act}")
        
        return finalized_act
    
    def get_acts_by_period(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        stream_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Получить акты за период из БД
        
        Args:
            start_date: Начало периода (опционально)
            end_date: Конец периода (опционально)
            stream_id: ID потока (опционально)
        
        Returns:
            Список актов
        """
        if not self.db:
            logger.warning("База данных недоступна")
            return []
        
        # По умолчанию - последние 7 дней
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        try:
            acts = self.db.get_acts_by_period(
                start_date=start_date,
                end_date=end_date,
                stream_id=stream_id
            )
            return acts
        except Exception as e:
            logger.error(f"❌ Ошибка получения актов из БД: {e}")
            return []
    
    def get_latest_act(self, stream_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Получить последний завершённый акт"""
        acts = self.get_acts_by_period(
            start_date=datetime.now().replace(hour=0, minute=0, second=0),
            end_date=datetime.now(),
            stream_id=stream_id
        )
        
        if not acts:
            return None
        
        return acts[-1]  # Последний акт
    
    def get_act_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получить агрегированную статистику по актам
        
        Returns:
            Словарь с метриками
        """
        if not self.db:
            return {
                "total_acts": 0,
                "total_pigs": 0,
                "total_weight": 0.0,
                "avg_weight": 0.0,
                "error": "Database unavailable"
            }
        
        try:
            stats = self.db.get_stats_summary(
                start_date=start_date,
                end_date=end_date,
                stream_id=stream_id
            )
            return stats
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {
                "total_acts": 0,
                "total_pigs": 0,
                "total_weight": 0.0,
                "avg_weight": 0.0,
                "error": str(e)
            }
