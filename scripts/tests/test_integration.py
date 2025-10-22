#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции всех модулей
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_video_processing():
    """Тестирует обработку видео"""
    print("🧪 Тест обработки видео")
    print("=" * 60)
    
    try:
        from pig_tracking.video_processor import IntegratedVideoProcessor
        from pig_tracking.database import DatabaseManager
        
        # Проверяем наличие тестового видео
        test_videos = list(Path('uploads').glob('*.mp4'))
        if not test_videos:
            print("❌ Тестовые видео не найдены в папке uploads/")
            print("   Поместите видеофайл в папку uploads/ для тестирования")
            return False
        
        test_video = test_videos[0]
        print(f"📹 Тестовое видео: {test_video.name}")
        
        # Создаем процессор
        print("\n⏳ Инициализация процессора...")
        processor = IntegratedVideoProcessor(
            stream_id="test",
            conf_threshold=0.30,
            img_size=960
        )
        
        await processor.initialize()
        print("✅ Процессор инициализирован")
        
        # Обрабатываем первые 100 кадров для теста
        print("\n⏳ Обработка первых 100 кадров...")
        summary = await processor.process_video_file(
            str(test_video),
            max_frames=100  # Ограничиваем для быстрого теста
        )
        
        print("\n✅ Обработка завершена!")
        print("\n📊 Результаты:")
        print(f"   • Обработано кадров: {summary['frames_processed']}")
        print(f"   • Обнаружено свиней: {summary.get('total_detections', 0)}")
        print(f"   • Пересечений линий: {summary['crossing_stats']['total_crossings']}")
        print(f"   • Актов взвешивания: {summary['act_stats']['completed_acts_count']}")
        
        # Тестируем сохранение в базу
        if summary['act_stats']['completed_acts_count'] > 0:
            print("\n⏳ Тестирование сохранения в базу...")
            
            db = DatabaseManager()
            
            from pig_tracking.database import WeighingAct, CrossingEvent
            
            for act in summary['act_stats']['completed_acts']:
                db_act = WeighingAct(
                    started_at=act['started_at'],
                    ended_at=act['ended_at'],
                    duration_sec=act['duration_sec'],
                    left_count=act['left_count'],
                    right_count=act['right_count'],
                    peak_count=act['peak_count'],
                    stream_id="test",
                    video_file=test_video.name
                )
                
                act_id = db.save_weighing_act(db_act)
                print(f"   ✅ Акт {act_id} сохранен в базу")
            
            print("✅ Сохранение в базу успешно")
        
        print("\n🎉 Все тесты пройдены успешно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        logger.error("Ошибка тестирования", exc_info=True)
        return False

async def test_database():
    """Тестирует подключение к базе данных"""
    print("\n🧪 Тест подключения к базе данных")
    print("=" * 60)
    
    try:
        from pig_tracking.database import DatabaseManager
        
        db = DatabaseManager()
        print("✅ Подключение к базе данных успешно")
        
        stats = db.get_stats()
        print(f"📊 Статистика: {stats['total_acts']} актов, {stats['total_crossings']} проходов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения к базе: {e}")
        return False

async def main():
    """Главная функция"""
    print("🐷 Тестирование системы отслеживания свиней")
    print("=" * 60)
    
    # Загружаем переменные окружения
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv не установлен")
    
    # Проверяем переменные окружения
    if not os.getenv('SUPABASE_KEY'):
        print("❌ SUPABASE_KEY не найден в .env")
        print("   Скопируйте .env.example в .env")
        return 1
    
    # Тест базы данных
    db_ok = await test_database()
    if not db_ok:
        print("\n⚠️ Убедитесь что Supabase запущен: docker-compose up -d")
        return 1
    
    # Тест обработки видео
    video_ok = await test_video_processing()
    
    if db_ok and video_ok:
        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        print("\nСистема готова к работе!")
        print("Запустите: python console_app.py")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))