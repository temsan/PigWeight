#!/usr/bin/env python3
"""
Скрипт для тестирования интеграции всех компонентов системы.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

from pig_tracking.video_processor import IntegratedVideoProcessor
from pig_tracking.database import DatabaseManager, WeighingAct, CrossingEvent
from pig_tracking.weight_estimator import get_weight_estimator


async def test_video_processing():
    """Тест обработки видео"""
    print("\n" + "="*60)
    print("🧪 ТЕСТ 1: Обработка видео")
    print("="*60)
    
    try:
        processor = IntegratedVideoProcessor(
            stream_id="test_integration",
            conf_threshold=0.30,
            img_size=960
        )
        
        print("✅ IntegratedVideoProcessor создан")
        
        # Проверяем компоненты
        print(f"   • CrossingCounter: {processor.crossing_counter is not None}")
        print(f"   • ActDetector: {processor.act_detector is not None}")
        print(f"   • WeightEstimator: {processor.weight_estimator is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def test_weight_estimator():
    """Тест оценщика веса"""
    print("\n" + "="*60)
    print("🧪 ТЕСТ 2: Оценка веса")
    print("="*60)
    
    try:
        estimator = get_weight_estimator()
        print("✅ WeightEstimator создан")
        
        # Тестовые оценки
        weights = []
        for i in range(5):
            weight = estimator.estimate_weight(pig_id=i)
            weights.append(weight)
            print(f"   • Свинья {i}: {weight} кг")
        
        # Статистика
        stats = estimator.get_stats()
        print(f"\n📊 Статистика:")
        print(f"   • Средний вес: {stats['avg_weight_kg']} кг")
        print(f"   • Стандартное отклонение: {stats['weight_std_kg']} кг")
        print(f"   • Отслеживается свиней: {stats['tracked_pigs']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def test_database_connection():
    """Тест подключения к БД"""
    print("\n" + "="*60)
    print("🧪 ТЕСТ 3: Подключение к БД")
    print("="*60)
    
    try:
        db = DatabaseManager()
        print("✅ DatabaseManager создан")
        
        # Получаем статистику
        stats = db.get_stats()
        print(f"\n📊 Статистика БД:")
        print(f"   • Всего актов: {stats['total_acts']}")
        print(f"   • Всего проходов: {stats['total_crossings']}")
        print(f"   • URL: {stats['database_url']}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ БД недоступна: {e}")
        print("   Это нормально, если Docker не запущен")
        return None  # None означает "не критично"


def test_data_conversion():
    """Тест конвертации данных"""
    print("\n" + "="*60)
    print("🧪 ТЕСТ 4: Конвертация данных")
    print("="*60)
    
    try:
        # Создаем тестовый акт (формат из video_processor)
        test_act = {
            'act_id': 1,
            'started_at': 1730000000.0,
            'ended_at': 1730000045.0,
            'duration': 45.0,
            'left_count': 15,
            'right_count': 14,
            'peak_count': 8,
            'total_weight': 880.5,
            'avg_weight': 110.1,
            'crossings': [
                {
                    'track_id': 42,
                    'side': 'left',
                    'mode': 'enter',
                    'x': 0.25,
                    'y': 0.53,
                    'timestamp': 1730000010.0,
                    'weight_estimate': 112.3
                }
            ]
        }
        
        print("✅ Тестовый акт создан")
        
        # Конвертируем в формат БД
        started_at = datetime.fromtimestamp(test_act['started_at'])
        ended_at = datetime.fromtimestamp(test_act['ended_at'])
        
        db_act = WeighingAct(
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=test_act['duration'],
            left_count=test_act['left_count'],
            right_count=test_act['right_count'],
            peak_count=test_act['peak_count'],
            total_weight=test_act.get('total_weight'),
            avg_weight=test_act.get('avg_weight'),
            stream_id='test',
            video_file='test.mp4'
        )
        
        print("✅ Акт сконвертирован в формат БД")
        print(f"   • started_at: {db_act.started_at}")
        print(f"   • ended_at: {db_act.ended_at}")
        print(f"   • duration: {db_act.duration_sec}s")
        print(f"   • total_weight: {db_act.total_weight} кг")
        
        # Конвертируем пересечение
        crossing = test_act['crossings'][0]
        crossing_time = datetime.fromtimestamp(crossing['timestamp'])
        
        db_crossing = CrossingEvent(
            pig_id=crossing['track_id'],
            direction=crossing['side'],
            timestamp=crossing_time,
            line_x=crossing['x'],
            line_y=crossing['y'],
            weight_estimate=crossing.get('weight_estimate'),
            stream_id='test'
        )
        
        print("✅ Пересечение сконвертировано в формат БД")
        print(f"   • pig_id: {db_crossing.pig_id}")
        print(f"   • direction: {db_crossing.direction}")
        print(f"   • timestamp: {db_crossing.timestamp}")
        print(f"   • weight: {db_crossing.weight_estimate} кг")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Главная функция тестирования"""
    print("\n" + "="*60)
    print("🐷 PigWeight - Тест интеграции компонентов")
    print("="*60)
    
    results = []
    
    # Запускаем тесты
    results.append(("Обработка видео", await test_video_processing()))
    results.append(("Оценка веса", test_weight_estimator()))
    results.append(("Подключение к БД", test_database_connection()))
    results.append(("Конвертация данных", test_data_conversion()))
    
    # Итоги
    print("\n" + "="*60)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            print(f"✅ {name}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {name}: FAILED")
            failed += 1
        else:
            print(f"⚠️ {name}: SKIPPED")
            skipped += 1
    
    print("\n" + "="*60)
    print(f"Всего тестов: {len(results)}")
    print(f"Успешно: {passed}")
    print(f"Провалено: {failed}")
    print(f"Пропущено: {skipped}")
    print("="*60)
    
    if failed > 0:
        print("\n❌ Некоторые тесты провалены!")
        return 1
    elif passed == len(results):
        print("\n✅ Все тесты пройдены успешно!")
        return 0
    else:
        print("\n⚠️ Некоторые тесты пропущены")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
