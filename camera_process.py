#!/usr/bin/env python3
"""
Обработка видео с камеры в реальном времени
"""

import asyncio
import sys
import cv2
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pig_tracking.video_processor import IntegratedVideoProcessor
from core.config import get_config

async def main():
    config = get_config()
    
    # Выбор источника
    print("\n📹 Выберите источник видео:")
    print("1. Веб-камера (0)")
    print("2. Внешняя камера (1)")
    print("3. IP камера (введите URL)")
    print("4. Видеофайл")
    
    choice = input("\nВыбор (1-4): ").strip()
    
    if choice == "1":
        source = 0
        source_name = "webcam_0"
    elif choice == "2":
        source = 1
        source_name = "webcam_1"
    elif choice == "3":
        source = input("Введите URL камеры (rtsp://...): ").strip()
        source_name = "ip_camera"
    elif choice == "4":
        source = input("Путь к видео: ").strip()
        source_name = Path(source).stem
    else:
        print("❌ Неверный выбор")
        return
    
    print(f"\n🎬 Открываем источник: {source}")
    
    # Открываем видео
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть источник: {source}")
        return
    
    # Получаем параметры
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Источник открыт: {width}x{height} @ {fps:.1f} FPS")
    print("=" * 60)
    
    # Создаем процессор
    processor = IntegratedVideoProcessor(
        stream_id=source_name,
        conf_threshold=config.CONF_THRESHOLD,
        img_size=config.IMG_SIZE
    )
    
    print("⏳ Инициализация процессора...")
    await processor.initialize()
    
    print("✅ Процессор готов!")
    print("\n🎯 Обработка в реальном времени...")
    print("   Нажмите 'q' для выхода, 's' для статистики")
    print("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Не удалось прочитать кадр")
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # Обрабатываем кадр
            result = await processor.process_frame(frame, timestamp)
            
            # Рисуем результаты на кадре
            display_frame = frame.copy()
            
            # Рисуем линии
            h, w = frame.shape[:2]
            line_left_x = int(0.25 * w)
            line_right_x = int(0.75 * w)
            cv2.line(display_frame, (line_left_x, 0), (line_left_x, h), (0, 255, 0), 2)
            cv2.line(display_frame, (line_right_x, 0), (line_right_x, h), (0, 255, 0), 2)
            
            # Рисуем bbox'ы
            for obj in result['tracked_objects']:
                bbox = obj['bbox']
                track_id = obj['id']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Информация на экране
            stats = result['crossing_stats']
            act_stats = result['act_stats']
            
            info_text = [
                f"FPS: {1/result['processing_time']:.1f}",
                f"Detected: {result['current_count']}",
                f"Crossings: {stats['total_crossings']}",
                f"Acts: {act_stats['completed_acts_count']}",
                f"Left: {stats['left_crossings']} | Right: {stats['right_crossings']}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(display_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Показываем кадр
            cv2.imshow('Pig Tracking', display_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️ Остановка...")
                break
            elif key == ord('s'):
                # Показываем статистику
                elapsed = time.time() - start_time
                print(f"\n📊 Статистика:")
                print(f"   Кадров: {frame_count}")
                print(f"   Время: {elapsed:.1f}s")
                print(f"   FPS: {frame_count/elapsed:.1f}")
                print(f"   Проходов: {stats['total_crossings']}")
                print(f"   Актов: {act_stats['completed_acts_count']}")
            
            # Логируем каждые 100 кадров
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"   Кадр {frame_count}, FPS: {frame_count/elapsed:.1f}, "
                      f"Проходов: {stats['total_crossings']}, "
                      f"Актов: {act_stats['completed_acts_count']}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Финальная статистика
        elapsed = time.time() - start_time
        stats = processor.get_stats()
        
        print("\n" + "=" * 60)
        print("📊 Финальная статистика:")
        print(f"   Кадров обработано: {frame_count}")
        print(f"   Время работы: {elapsed:.1f}s")
        print(f"   Средний FPS: {frame_count/elapsed:.1f}")
        print(f"   Проходов: {stats['crossing_stats']['total_crossings']}")
        print(f"   Актов: {stats['act_stats']['completed_acts_count']}")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
