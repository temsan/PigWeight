#!/usr/bin/env python3
"""
Тестовый скрипт для быстрой проверки видеопотока на бэкенде.
Помогает выявить проблемы с черным экраном и отображением масок.
"""
import os
import sys
import cv2
import time
import asyncio
import numpy as np
from pathlib import Path

# Добавляем корневую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import VideoStream, FileStream, DemoStream
from core.config import load_config
from services.model_adapter import ModelAdapter


def test_frame_extraction(video_path: str, max_frames: int = 10):
    """Тестируем извлечение кадров из видео"""
    print(f"🎥 Тестируем извлечение кадров из: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Файл не найден: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удается открыть видео: {video_path}")
        return False
    
    frame_count = 0
    black_frames = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Проверяем на черный кадр
        frame_mean = np.mean(frame)
        if frame_mean < 10:  # Очень темный кадр
            black_frames += 1
            print(f"⚠️  Кадр {frame_count}: очень темный (mean={frame_mean:.1f})")
        else:
            print(f"✅ Кадр {frame_count}: OK (mean={frame_mean:.1f}, shape={frame.shape})")
    
    cap.release()
    
    print(f"📊 Результат: {frame_count} кадров, {black_frames} черных")
    return black_frames == 0


def test_model_inference(video_path: str):
    """Тестируем инференс модели"""
    print(f"🧠 Тестируем инференс модели...")
    
    try:
        config = load_config()
        model_path = config.get('model_path', 'models/pig_yolo11-seg.v4')
        
        # Проверяем доступность моделей (.pt или .onnx)
        pt_path = f"{model_path}.pt"
        onnx_path = f"{model_path}.onnx"
        if not (os.path.exists(pt_path) or os.path.exists(onnx_path)):
            print(f"❌ Модели не найдены: {pt_path} или {onnx_path}")
            return False
            
        print(f"📂 Загружаем модель: {model_path}")
        model = ModelAdapter(model_path)
        
        # Берем один кадр для теста
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Не удается извлечь кадр для теста")
            return False
            
        print(f"🔍 Запускаем инференс на кадре {frame.shape}")
        start_time = time.time()
        results = model.infer([frame])
        inference_time = time.time() - start_time
        
        if results and len(results) > 0:
            result = results[0]
            detections = result.get('detections', 0)
            confidence = result.get('confidence', 0.0)
            masks = result.get('masks', [])
            
            print(f"✅ Инференс завершен за {inference_time:.3f}с")
            print(f"📈 Детекций: {detections}, уверенность: {confidence:.3f}")
            print(f"🎭 Масок: {len(masks) if masks else 0}")
            
            if masks:
                print("✅ Маски найдены - отображение должно работать")
            else:
                print("⚠️  Масок не найдено - проверьте модель и пороги")
                
            return True
        else:
            print("❌ Инференс не вернул результатов")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка инференса: {e}")
        return False


async def test_stream_class(video_path: str):
    """Тестируем класс VideoStream"""
    print(f"🔄 Тестируем класс VideoStream...")
    
    try:
        # Создаем поток
        stream = FileStream("test_stream", video_path)
        
        # Запускаем
        await stream.start()
        
        # Ждем немного
        await asyncio.sleep(2)
        
        # Проверяем состояние
        if stream.running:
            print("✅ Поток запущен")
            
            # Пробуем получить JPEG
            jpeg_data = await stream.get_jpeg()
            if jpeg_data:
                print(f"✅ JPEG получен, размер: {len(jpeg_data)} байт")
                
                # Декодируем для проверки
                nparr = np.frombuffer(jpeg_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    mean_val = np.mean(img)
                    print(f"✅ JPEG валиден, mean={mean_val:.1f}, shape={img.shape}")
                    
                    if mean_val < 10:
                        print("⚠️  JPEG очень темный - возможно черный экран!")
                    
                    return True
                else:
                    print("❌ JPEG поврежден")
                    return False
            else:
                print("❌ JPEG не получен")
                return False
        else:
            print("❌ Поток не запустился")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования потока: {e}")
        return False
    finally:
        try:
            await stream.stop()
        except:
            pass


def test_video_files():
    """Ищем и тестируем доступные видеофайлы"""
    print("🔍 Ищем видеофайлы для тестирования...")
    
    # Проверяем стандартные папки
    test_paths = [
        "temp/",
        "uploads/",
        "records/",
        "../test_videos/"
    ]
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    found_videos = []
    
    for test_path in test_paths:
        if os.path.exists(test_path):
            for root, dirs, files in os.walk(test_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        full_path = os.path.join(root, file)
                        found_videos.append(full_path)
    
    if not found_videos:
        print("❌ Видеофайлы не найдены")
        print("💡 Поместите тестовый видеофайл в папку temp/ или uploads/")
        return None
        
    print(f"📁 Найдено видеофайлов: {len(found_videos)}")
    for video in found_videos:
        print(f"   - {video}")
    
    # Берем первый файл для тестирования
    return found_videos[0]


def main():
    """Главная функция тестирования"""
    print("🚀 Запуск тестирования видеопотока")
    print("=" * 50)
    
    # Ищем видеофайл
    video_path = test_video_files()
    if not video_path:
        return
    
    print(f"\n🎯 Тестируем файл: {video_path}")
    print("-" * 50)
    
    # Тест 1: Извлечение кадров
    print("\n1️⃣ Тест извлечения кадров")
    frame_test_ok = test_frame_extraction(video_path)
    
    # Тест 2: Инференс модели  
    print("\n2️⃣ Тест инференса модели")
    model_test_ok = test_model_inference(video_path)
    
    # Тест 3: Класс VideoStream
    print("\n3️⃣ Тест класса VideoStream")
    try:
        stream_test_ok = asyncio.run(test_stream_class(video_path))
    except Exception as e:
        print(f"❌ Ошибка async теста: {e}")
        stream_test_ok = False
    
    # Итоговый отчет
    print("\n" + "=" * 50)
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    print(f"🎥 Извлечение кадров: {'✅ OK' if frame_test_ok else '❌ FAIL'}")
    print(f"🧠 Инференс модели: {'✅ OK' if model_test_ok else '❌ FAIL'}")
    print(f"🔄 VideoStream класс: {'✅ OK' if stream_test_ok else '❌ FAIL'}")
    
    if not frame_test_ok:
        print("\n💡 Проблема с извлечением кадров - проверьте видеофайл")
    elif not model_test_ok:
        print("\n💡 Проблема с моделью - проверьте путь к модели в .env")
    elif not stream_test_ok:
        print("\n💡 Проблема с VideoStream - проверьте логику потока")
    else:
        print("\n🎉 Все тесты прошли успешно!")
        print("💡 Проблема может быть в frontend или WebRTC")


if __name__ == "__main__":
    main()
