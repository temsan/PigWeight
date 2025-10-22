#!/usr/bin/env python3
"""
Тестирование MVP системы отслеживания свиней
Полный цикл: обработка видео -> запись в БД -> проверка API
"""

import sys
import time
from pathlib import Path
import cv2

def test_system_ready():
    """Проверка готовности системы"""
    print("\n" + "="*80)
    print("ШАГ 1: Проверка готовности системы")
    print("="*80)
    
    # Прямая проверка компонентов
    try:
        from pig_tracking.database import DatabaseManager
        import cv2
        from pathlib import Path
        
        # Проверка БД
        db = DatabaseManager()
        print("✓ База данных подключена")
        
        # Проверка видео
        video_path = Path("uploads/test_video.mp4")
        if not video_path.exists():
            print("✗ Тестовое видео не найдено")
            return False
        print("✓ Тестовое видео найдено")
        
        # Проверка OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("✗ Не удается открыть видео")
            return False
        cap.release()
        print("✓ OpenCV работает")
        
        print("\n✓ Система готова к тестированию")
        return True
        
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        return False

def test_video_processing():
    """Тест обработки видео"""
    print("\n" + "="*80)
    print("🎬 ШАГ 2: Обработка тестового видео")
    print("="*80)
    
    video_path = Path("uploads/test_video.mp4")
    if not video_path.exists():
        print(f"❌ Видео не найдено: {video_path}")
        return False
    
    print(f"📹 Видео: {video_path}")
    print(f"📊 Размер: {video_path.stat().st_size / 1024:.1f} KB")
    
    try:
        # Импортируем модули
        from pig_tracking.video_processor import VideoProcessor
        from pig_tracking.database import DatabaseManager
        
        # Инициализация
        print("\n⚙️ Инициализация компонентов...")
        db = DatabaseManager()
        processor = VideoProcessor(db)
        
        # Открываем видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("❌ Не удалось открыть видео")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   FPS: {fps}")
        print(f"   Кадров: {total_frames}")
        print(f"   Разрешение: {width}x{height}")
        
        # Обработка видео
        print(f"\n🔄 Обработка {total_frames} кадров...")
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обрабатываем каждый кадр
            timestamp = frame_count / fps
            processor.process_frame(frame, timestamp)
            
            frame_count += 1
            
            # Прогресс каждые 30 кадров
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(f"   Кадр {frame_count}/{total_frames} | "
                      f"FPS: {fps_actual:.1f} | "
                      f"Время: {elapsed:.1f}s")
        
        cap.release()
        
        # Финализация
        processor.finalize()
        
        # Статистика
        elapsed_total = time.time() - start_time
        fps_avg = frame_count / elapsed_total if elapsed_total > 0 else 0
        
        print(f"\n📊 Результаты обработки:")
        print(f"   Обработано кадров: {frame_count}")
        print(f"   Время обработки: {elapsed_total:.2f}s")
        print(f"   Средний FPS: {fps_avg:.1f}")
        
        stats = processor.get_statistics()
        print(f"\n📈 Статистика детекции:")
        print(f"   Всего детекций: {stats.get('total_detections', 0)}")
        print(f"   Активных треков: {stats.get('active_tracks', 0)}")
        print(f"   Завершенных треков: {stats.get('completed_tracks', 0)}")
        
        print("\n✅ Обработка видео завершена успешно")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при обработке видео: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database():
    """Проверка данных в базе"""
    print("\n" + "="*80)
    print("💾 ШАГ 3: Проверка данных в базе")
    print("="*80)
    
    try:
        from pig_tracking.database import DatabaseManager
        
        db = DatabaseManager()
        
        # Получаем данные
        acts = db.get_weighing_acts()
        crossings = db.get_line_crossings(limit=100)
        
        print(f"\n📊 Данные в базе:")
        print(f"   Актов взвешивания: {len(acts)}")
        print(f"   Пересечений линий: {len(crossings)}")
        
        # Показываем последние акты
        if acts:
            print(f"\n📋 Последние акты взвешивания:")
            for i, act in enumerate(acts[-3:], 1):
                print(f"\n   Акт #{i}:")
                print(f"      ID: {act.get('id')}")
                print(f"      Свиней: {act.get('pig_count')}")
                print(f"      Начало: {act.get('start_time')}")
                print(f"      Конец: {act.get('end_time')}")
                print(f"      Длительность: {act.get('duration_sec', 0):.1f}s")
        
        # Показываем последние пересечения
        if crossings:
            print(f"\n🚶 Последние пересечения линий:")
            for i, crossing in enumerate(crossings[-5:], 1):
                print(f"   #{i}: Track {crossing.get('track_id')} | "
                      f"Линия: {crossing.get('line_name')} | "
                      f"Направление: {crossing.get('direction')} | "
                      f"Время: {crossing.get('timestamp')}")
        
        print("\n✅ Проверка базы данных завершена")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при проверке БД: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api():
    """Проверка API эндпоинтов"""
    print("\n" + "="*80)
    print("🌐 ШАГ 4: Проверка API")
    print("="*80)
    
    try:
        import requests
        
        base_url = "http://localhost:8080"
        
        # Проверяем доступность API
        print(f"\n🔍 Проверка доступности API: {base_url}")
        
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API доступен: {response.json()}")
            else:
                print(f"⚠️ API вернул статус: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠️ API не запущен")
            print("💡 Запустите API: python -m uvicorn api.app:app --port 8080")
            return False
        
        # Тестируем эндпоинты
        endpoints = [
            ("/api/weighing-acts", "Акты взвешивания"),
            ("/api/line-crossings", "Пересечения линий"),
            ("/api/statistics", "Статистика"),
        ]
        
        print(f"\n🧪 Тестирование эндпоинтов:")
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ {name}: {len(data) if isinstance(data, list) else 'OK'}")
                else:
                    print(f"   ❌ {name}: статус {response.status_code}")
            except Exception as e:
                print(f"   ❌ {name}: {e}")
        
        print("\n✅ Проверка API завершена")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при проверке API: {e}")
        return False

def generate_report(results):
    """Генерация отчета о тестировании"""
    print("\n" + "="*80)
    print("📝 ОТЧЕТ О ТЕСТИРОВАНИИ MVP")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print(f"\n📊 Результаты: {passed}/{total} тестов пройдено")
    print(f"\n{'Тест':<30} {'Результат':<10}")
    print("-" * 40)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status:<10}")
    
    print("\n" + "="*80)
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ MVP готов к использованию")
    else:
        print("⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")
        print("🔧 Требуется доработка")
    
    print("="*80)
    
    return passed == total

def main():
    """Главная функция тестирования"""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ MVP - СИСТЕМА ОТСЛЕЖИВАНИЯ СВИНЕЙ")
    print("="*80)
    print(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Запускаем тесты последовательно
    results["Готовность системы"] = test_system_ready()
    
    if results["Готовность системы"]:
        results["Обработка видео"] = test_video_processing()
        results["База данных"] = test_database()
        results["API"] = test_api()
    else:
        print("\n❌ Система не готова, дальнейшие тесты пропущены")
        results["Обработка видео"] = False
        results["База данных"] = False
        results["API"] = False
    
    # Генерируем отчет
    success = generate_report(results)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
