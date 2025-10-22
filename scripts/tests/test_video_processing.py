#!/usr/bin/env python3
"""
Тестирование обработки видео без базы данных
"""

import sys
from pathlib import Path
from pig_tracking.video_processor import VideoProcessor
from pig_tracking.tracker import PigTracker
from core.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_video_processing():
    """Тестирует обработку видео"""
    
    print("🐷 Тестирование обработки видео")
    print("=" * 60)
    
    # Проверяем наличие видео
    video_path = Path('uploads/test_video.mp4')
    if not video_path.exists():
        print(f"❌ Видео не найдено: {video_path}")
        return False
    
    print(f"✅ Видео найдено: {video_path}")
    print(f"📊 Размер: {video_path.stat().st_size / 1024:.1f} KB")
    
    # Создаем процессор
    print("\n🔧 Инициализация процессора...")
    try:
        processor = VideoProcessor(
            model_path=str(settings.MODEL_PATH),
            conf_threshold=settings.CONF_THRESHOLD,
            device=settings.DEVICE
        )
        print("✅ Процессор инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False
    
    # Создаем трекер
    tracker = PigTracker(
        iou_threshold=settings.IOU_THRESHOLD,
        max_age=settings.MAX_AGE
    )
    print("✅ Трекер создан")
    
    # Обрабатываем видео
    print(f"\n🎬 Обработка видео...")
    print(f"Модель: {settings.MODEL_PATH}")
    print(f"Confidence: {settings.CONF_THRESHOLD}")
    print(f"Device: {settings.DEVICE}")
    
    try:
        results = processor.process_video(
            video_path=str(video_path),
            tracker=tracker,
            save_output=True,
            output_dir='outputs'
        )
        
        print("\n✅ Обработка завершена!")
        print("=" * 60)
        print(f"📊 Результаты:")
        print(f"  • Всего кадров: {results.get('total_frames', 0)}")
        print(f"  • Обработано: {results.get('processed_frames', 0)}")
        print(f"  • Обнаружено объектов: {results.get('total_detections', 0)}")
        print(f"  • Уникальных треков: {results.get('unique_tracks', 0)}")
        print(f"  • Проходов через левую линию: {results.get('left_crosses', 0)}")
        print(f"  • Проходов через правую линию: {results.get('right_crosses', 0)}")
        
        if 'output_video' in results:
            print(f"\n🎥 Видео с результатами: {results['output_video']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка обработки: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_video_processing()
    sys.exit(0 if success else 1)
