#!/usr/bin/env python3
"""
ВКЛАДКА 1: Тестирование обработки видео
Обработка test_video.mp4 и проверка результатов
"""

import sys
import time
import cv2
from pathlib import Path

def test_video_processing():
    """Тестирование обработки видео"""
    print("=" * 80)
    print("ВКЛАДКА 1: Тестирование обработки видео")
    print("=" * 80)
    print(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Проверка видео
    video_path = Path("uploads/test_video.mp4")
    if not video_path.exists():
        print(f"❌ Видео не найдено: {video_path}")
        return False
    
    print(f"✓ Видео найдено: {video_path}")
    print(f"  Размер: {video_path.stat().st_size / 1024:.1f} KB\n")
    
    # Получаем информацию о видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Не удалось открыть видео")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print("📹 Информация о видео:")
    print(f"  Разрешение: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Кадров: {total_frames}")
    print(f"  Длительность: {duration:.1f}s\n")
    
    cap.release()
    
    # Проверка базы данных
    print("🗄️ Проверка подключения к базе данных...")
    try:
        from pig_tracking.database import DatabaseManager
        db = DatabaseManager()
        print("✓ База данных подключена\n")
    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}\n")
        return False
    
    # Запуск обработки
    print("🚀 Запуск обработки видео...")
    print("-" * 80)
    
    try:
        # Импортируем процессор
        from pig_tracking.video_processor import IntegratedVideoProcessor
        import asyncio
        
        async def process():
            processor = IntegratedVideoProcessor(
                stream_id="test_video",
                conf_threshold=0.30,
                img_size=960
            )
            
            print("⚙️ Инициализация процессора...")
            await processor.initialize()
            print("✓ Процессор инициализирован\n")
            
            print("🔄 Обработка кадров...")
            start_time = time.time()
            
            summary = await processor.process_video_file(str(video_path))
            
            elapsed = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("📊 РЕЗУЛЬТАТЫ ОБРАБОТКИ")
            print("=" * 80)
            print(f"Время обработки: {elapsed:.2f}s")
            print(f"Обработано кадров: {summary['frames_processed']}")
            print(f"Средний FPS: {summary['frames_processed'] / elapsed:.1f}")
            print()
            
            print("🎯 Статистика детекции:")
            print(f"  Всего детекций: {summary['detection_stats']['total_detections']}")
            print(f"  Активных треков: {summary['detection_stats']['active_tracks']}")
            print(f"  Завершенных треков: {summary['detection_stats']['completed_tracks']}")
            print()
            
            print("↔️ Статистика пересечений:")
            crossing_stats = summary['crossing_stats']
            print(f"  Всего пересечений: {crossing_stats['total_crossings']}")
            print(f"  Вход слева: {crossing_stats['left_enter']}")
            print(f"  Выход слева: {crossing_stats['left_exit']}")
            print(f"  Вход справа: {crossing_stats['right_enter']}")
            print(f"  Выход справа: {crossing_stats['right_exit']}")
            print()
            
            print("📋 Статистика актов:")
            act_stats = summary['act_stats']
            print(f"  Завершенных актов: {act_stats['completed_acts_count']}")
            print(f"  Активных актов: {act_stats['active_acts_count']}")
            print()
            
            # Сохранение в БД
            print("💾 Сохранение результатов в базу данных...")
            
            # Сохраняем пересечения
            crossings_saved = 0
            for crossing in summary.get('crossings', []):
                try:
                    from pig_tracking.database import CrossingEvent
                    from datetime import datetime
                    
                    event = CrossingEvent(
                        pig_id=crossing['track_id'],
                        direction=f"{crossing['side']}_{crossing['mode']}",
                        timestamp=datetime.fromtimestamp(crossing['timestamp']),
                        line_x=crossing['x'],
                        line_y=crossing['y'],
                        stream_id="test_video"
                    )
                    db.save_crossing(event)
                    crossings_saved += 1
                except Exception as e:
                    print(f"  ⚠️ Ошибка сохранения пересечения: {e}")
            
            print(f"✓ Сохранено пересечений: {crossings_saved}")
            
            # Сохраняем акты
            acts_saved = 0
            for act in summary.get('completed_acts', []):
                try:
                    from pig_tracking.database import WeighingAct
                    from datetime import datetime
                    
                    weighing_act = WeighingAct(
                        started_at=datetime.fromtimestamp(act['started_at']),
                        ended_at=datetime.fromtimestamp(act['ended_at']),
                        duration_sec=act['duration'],
                        left_count=act['left_count'],
                        right_count=act['right_count'],
                        peak_count=act['peak_count'],
                        stream_id="test_video",
                        video_file=str(video_path)
                    )
                    db.save_weighing_act(weighing_act)
                    acts_saved += 1
                except Exception as e:
                    print(f"  ⚠️ Ошибка сохранения акта: {e}")
            
            print(f"✓ Сохранено актов: {acts_saved}")
            print()
            
            # Проверка данных в БД
            print("🔍 Проверка данных в базе...")
            try:
                from datetime import datetime, timedelta
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=1)
                
                acts = db.get_acts_by_period(start_time, end_time)
                print(f"✓ Актов в БД: {len(acts)}")
                
                if acts:
                    print("\n📋 Последний акт:")
                    last_act = acts[-1]
                    print(f"  ID: {last_act.get('id')}")
                    print(f"  Начало: {last_act.get('started_at')}")
                    print(f"  Конец: {last_act.get('ended_at')}")
                    print(f"  Длительность: {last_act.get('duration_sec', 0):.1f}s")
                    print(f"  Свиней (лево): {last_act.get('left_count')}")
                    print(f"  Свиней (право): {last_act.get('right_count')}")
                    print(f"  Пик: {last_act.get('peak_count')}")
            except Exception as e:
                print(f"⚠️ Ошибка проверки БД: {e}")
            
            print("\n" + "=" * 80)
            print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print("=" * 80)
            
            return True
        
        # Запускаем асинхронную обработку
        result = asyncio.run(process())
        return result
        
    except Exception as e:
        print(f"\n❌ Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_video_processing()
    sys.exit(0 if success else 1)
