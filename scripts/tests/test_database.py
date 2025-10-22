#!/usr/bin/env python3
"""
Тестовый скрипт для проверки DatabaseManager
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

from pig_tracking.database import DatabaseManager, WeighingAct, CrossingEvent

def test_database():
    """Тестирует подключение и основные операции с базой данных"""
    
    print("🧪 Тестирование DatabaseManager...")
    
    try:
        # Инициализация
        db = DatabaseManager()
        print("✅ Подключение к базе данных успешно")
        
        # Получение статистики
        stats = db.get_stats()
        print(f"📊 Статистика: {stats['total_acts']} актов, {stats['total_crossings']} проходов")
        
        # Создание тестового акта
        now = datetime.now()
        test_act = WeighingAct(
            started_at=now - timedelta(minutes=5),
            ended_at=now,
            duration_sec=300.0,
            left_count=15,
            right_count=12,
            peak_count=8,
            total_weight=1200.5,
            avg_weight=44.5,
            stream_id="test_stream",
            video_file="test_video.mp4"
        )
        
        # Добавляем тестовые проходы
        test_act.crossings = [
            CrossingEvent(
                pig_id=1,
                direction="left",
                timestamp=now - timedelta(minutes=4),
                line_x=0.25,
                line_y=0.5,
                weight_estimate=45.0,
                stream_id="test_stream"
            ),
            CrossingEvent(
                pig_id=2,
                direction="right",
                timestamp=now - timedelta(minutes=3),
                line_x=0.75,
                line_y=0.6,
                weight_estimate=42.0,
                stream_id="test_stream"
            )
        ]
        
        # Сохранение акта
        act_id = db.save_weighing_act(test_act)
        print(f"✅ Тестовый акт сохранен с ID: {act_id}")
        
        # Получение актов за период
        acts = db.get_acts_by_period(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1)
        )
        print(f"✅ Получено {len(acts)} актов за период")
        
        if acts:
            act = acts[-1]  # Последний акт
            print(f"   Последний акт: {act.started_at} - {act.ended_at}")
            print(f"   Проходы: слева={act.left_count}, справа={act.right_count}, пик={act.peak_count}")
            
            # Получение проходов для акта
            crossings = db.get_crossings_by_act(act.id)
            print(f"   Связанных проходов: {len(crossings)}")
        
        # Обновленная статистика
        stats = db.get_stats()
        print(f"📊 Обновленная статистика: {stats['total_acts']} актов, {stats['total_crossings']} проходов")
        
        print("🎉 Все тесты прошли успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Загружаем переменные окружения из .env
    from dotenv import load_dotenv
    load_dotenv()
    
    success = test_database()
    sys.exit(0 if success else 1)