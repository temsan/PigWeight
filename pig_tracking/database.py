"""
Модуль для работы с базой данных Supabase
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from supabase import create_client, Client

logger = logging.getLogger(__name__)

@dataclass
class CrossingEvent:
    """Событие пересечения линии"""
    pig_id: int
    direction: str  # "left" or "right"
    timestamp: datetime
    line_x: float
    line_y: float
    weight_estimate: Optional[float] = None
    act_id: Optional[int] = None
    stream_id: Optional[str] = None

@dataclass
class WeighingAct:
    """Акт взвешивания"""
    started_at: datetime
    ended_at: datetime
    duration_sec: float
    left_count: int
    right_count: int
    peak_count: int
    total_weight: Optional[float] = None
    avg_weight: Optional[float] = None
    stream_id: Optional[str] = None
    video_file: Optional[str] = None
    id: Optional[int] = None
    crossings: List[CrossingEvent] = None

    def __post_init__(self):
        if self.crossings is None:
            self.crossings = []

class DatabaseManager:
    """Менеджер для работы с базой данных Supabase"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Инициализация подключения к Supabase
        
        Args:
            supabase_url: URL Supabase (по умолчанию из .env)
            supabase_key: API ключ Supabase (по умолчанию из .env)
        """
        self.url = supabase_url or os.getenv('SUPABASE_URL', 'http://localhost:8000')
        # Используем service_role ключ для полного доступа к БД
        self.key = supabase_key or os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')
        
        if not self.key:
            raise ValueError("SUPABASE_KEY не найден в переменных окружения")
        
        try:
            # Создаем клиент Supabase
            # Библиотека supabase-py автоматически добавляет правильные заголовки
            self.client: Client = create_client(self.url, self.key)
            logger.info(f"Подключение к Supabase: {self.url} (key: {self.key[:20]}...)")
            
            # Тестируем подключение
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Ошибка подключения к Supabase: {e}")
            raise
    
    def _test_connection(self):
        """Тестирует подключение к базе данных"""
        try:
            # Простой запрос для проверки подключения
            result = self.client.table('weighing_acts').select("count", count="exact").execute()
            logger.info("✅ Подключение к Supabase успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования подключения: {e}")
            raise
    
    def save_crossing(self, crossing: CrossingEvent) -> int:
        """
        Сохраняет событие пересечения линии
        
        Args:
            crossing: Событие пересечения
            
        Returns:
            ID созданной записи
        """
        try:
            data = {
                'act_id': crossing.act_id,
                'pig_id': crossing.pig_id,
                'direction': crossing.direction,
                'crossed_at': crossing.timestamp.isoformat(),
                'line_x': crossing.line_x,
                'line_y': crossing.line_y,
                'weight_estimate': crossing.weight_estimate,
                'stream_id': crossing.stream_id
            }
            
            result = self.client.table('crossings').insert(data).execute()
            
            if result.data:
                crossing_id = result.data[0]['id']
                logger.debug(f"Сохранен проход: pig_id={crossing.pig_id}, direction={crossing.direction}, id={crossing_id}")
                return crossing_id
            else:
                raise Exception("Не удалось получить ID созданной записи")
                
        except Exception as e:
            logger.error(f"Ошибка сохранения прохода: {e}")
            raise
    
    def save_weighing_act(self, act: WeighingAct) -> int:
        """
        Сохраняет акт взвешивания
        
        Args:
            act: Акт взвешивания
            
        Returns:
            ID созданной записи
        """
        try:
            data = {
                'started_at': act.started_at.isoformat(),
                'ended_at': act.ended_at.isoformat(),
                'duration_sec': act.duration_sec,
                'left_count': act.left_count,
                'right_count': act.right_count,
                'peak_count': act.peak_count,
                'total_weight': act.total_weight,
                'avg_weight': act.avg_weight,
                'stream_id': act.stream_id,
                'video_file': act.video_file
            }
            
            result = self.client.table('weighing_acts').insert(data).execute()
            
            if result.data:
                act_id = result.data[0]['id']
                act.id = act_id
                
                # Сохраняем связанные проходы
                if act.crossings:
                    for crossing in act.crossings:
                        crossing.act_id = act_id
                        self.save_crossing(crossing)
                
                logger.info(f"Сохранен акт взвешивания: {act.started_at} - {act.ended_at}, left={act.left_count}, right={act.right_count}, id={act_id}")
                return act_id
            else:
                raise Exception("Не удалось получить ID созданной записи")
                
        except Exception as e:
            logger.error(f"Ошибка сохранения акта взвешивания: {e}")
            raise
    
    def get_acts_by_period(self, start: datetime, end: datetime, stream_id: str = None) -> List[WeighingAct]:
        """
        Получает акты взвешивания за период
        
        Args:
            start: Начало периода
            end: Конец периода
            stream_id: ID потока (опционально)
            
        Returns:
            Список актов взвешивания
        """
        try:
            query = self.client.table('weighing_acts')\
                .select('*')\
                .gte('started_at', start.isoformat())\
                .lte('started_at', end.isoformat())\
                .order('started_at')
            
            if stream_id:
                query = query.eq('stream_id', stream_id)
            
            result = query.execute()
            
            acts = []
            for row in result.data:
                act = WeighingAct(
                    id=row['id'],
                    started_at=datetime.fromisoformat(row['started_at'].replace('Z', '+00:00')),
                    ended_at=datetime.fromisoformat(row['ended_at'].replace('Z', '+00:00')),
                    duration_sec=row['duration_sec'],
                    left_count=row['left_count'],
                    right_count=row['right_count'],
                    peak_count=row['peak_count'],
                    total_weight=row['total_weight'],
                    avg_weight=row['avg_weight'],
                    stream_id=row['stream_id'],
                    video_file=row['video_file']
                )
                acts.append(act)
            
            logger.info(f"Получено {len(acts)} актов за период {start} - {end}")
            return acts
            
        except Exception as e:
            logger.error(f"Ошибка получения актов: {e}")
            raise
    
    def get_crossings_by_act(self, act_id: int) -> List[CrossingEvent]:
        """
        Получает проходы для конкретного акта
        
        Args:
            act_id: ID акта взвешивания
            
        Returns:
            Список проходов
        """
        try:
            result = self.client.table('crossings')\
                .select('*')\
                .eq('act_id', act_id)\
                .order('crossed_at')\
                .execute()
            
            crossings = []
            for row in result.data:
                crossing = CrossingEvent(
                    pig_id=row['pig_id'],
                    direction=row['direction'],
                    timestamp=datetime.fromisoformat(row['crossed_at'].replace('Z', '+00:00')),
                    line_x=row['line_x'],
                    line_y=row['line_y'],
                    weight_estimate=row['weight_estimate'],
                    act_id=row['act_id'],
                    stream_id=row['stream_id']
                )
                crossings.append(crossing)
            
            return crossings
            
        except Exception as e:
            logger.error(f"Ошибка получения проходов для акта {act_id}: {e}")
            raise
    
    def get_pig_passages(self, act_id: int = None) -> List[Dict[str, Any]]:
        """
        Получает агрегированные данные о проходах свиней
        Группирует пересечения линий по pig_id в одну запись
        
        Args:
            act_id: ID акта (опционально, если None - все акты)
            
        Returns:
            Список проходов свиней с агрегированными данными
        """
        try:
            # Получаем все пересечения
            query = self.client.table('crossings').select('*')
            
            if act_id:
                query = query.eq('act_id', act_id)
            
            result = query.order('crossed_at').execute()
            
            # Группируем по pig_id и act_id
            passages = {}
            for row in result.data:
                key = (row['act_id'], row['pig_id'])
                
                if key not in passages:
                    passages[key] = {
                        'act_id': row['act_id'],
                        'pig_id': row['pig_id'],
                        'stream_id': row['stream_id'],
                        'first_crossing': row['crossed_at'],
                        'last_crossing': row['crossed_at'],
                        'crossings': [],
                        'weights': []
                    }
                
                passages[key]['last_crossing'] = row['crossed_at']
                passages[key]['crossings'].append({
                    'direction': row['direction'],
                    'timestamp': row['crossed_at'],
                    'line_x': row['line_x'],
                    'line_y': row['line_y']
                })
                
                if row['weight_estimate']:
                    passages[key]['weights'].append(row['weight_estimate'])
            
            # Формируем итоговый список
            result_list = []
            for passage in passages.values():
                # Вычисляем путь
                path = ' -> '.join([c['direction'] for c in passage['crossings']])
                
                # Средний вес
                avg_weight = sum(passage['weights']) / len(passage['weights']) if passage['weights'] else None
                
                result_list.append({
                    'act_id': passage['act_id'],
                    'pig_id': passage['pig_id'],
                    'stream_id': passage['stream_id'],
                    'entered_at': passage['first_crossing'],
                    'exited_at': passage['last_crossing'],
                    'crossings_count': len(passage['crossings']),
                    'path': path,
                    'avg_weight': round(avg_weight, 1) if avg_weight else None
                })
            
            logger.info(f"Получено {len(result_list)} проходов свиней")
            return result_list
            
        except Exception as e:
            logger.error(f"Ошибка получения проходов: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получает общую статистику
        
        Returns:
            Словарь со статистикой
        """
        try:
            # Общее количество актов
            acts_result = self.client.table('weighing_acts').select("count", count="exact").execute()
            total_acts = acts_result.count
            
            # Общее количество проходов
            crossings_result = self.client.table('crossings').select("count", count="exact").execute()
            total_crossings = crossings_result.count
            
            # Последний акт
            last_act_result = self.client.table('weighing_acts')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            last_act = None
            if last_act_result.data:
                row = last_act_result.data[0]
                last_act = {
                    'started_at': row['started_at'],
                    'ended_at': row['ended_at'],
                    'left_count': row['left_count'],
                    'right_count': row['right_count'],
                    'peak_count': row['peak_count']
                }
            
            stats = {
                'total_acts': total_acts,
                'total_crossings': total_crossings,
                'last_act': last_act,
                'database_url': self.url
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            raise
    
    def clear_all_data(self):
        """
        Очищает все данные (для тестирования)
        ВНИМАНИЕ: Удаляет все записи!
        """
        try:
            # Сначала удаляем проходы (из-за foreign key)
            self.client.table('crossings').delete().neq('id', 0).execute()
            
            # Затем удаляем акты
            self.client.table('weighing_acts').delete().neq('id', 0).execute()
            
            logger.warning("🗑️ Все данные удалены из базы")
            
        except Exception as e:
            logger.error(f"Ошибка очистки данных: {e}")
            raise