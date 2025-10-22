#!/usr/bin/env python3
"""
Простой тест инференса для проверки масок
"""

import sys
import os
sys.path.append('.')

import numpy as np
import cv2

# Прямой импорт для избежания циклических зависимостей
import os
os.environ['DEVICE'] = 'cuda:0'
os.environ['USE_HALF'] = 'true'
os.environ['MODEL_PATH'] = 'models/pig_yolo11-seg.v4.pt'

from services.model_adapter import ModelAdapter

def test_inference():
    print("=== Тест инференса ===")
    model_path = "models/pig_yolo11-seg.v4.pt"
    device = "cuda:0"
    use_half = True
    
    print(f"MODEL_PATH: {model_path}")
    print(f"DEVICE: {device}")
    print(f"USE_HALF: {use_half}")
    
    # Проверяем что модель существует
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    print(f"✅ Модель найдена: {model_path}")
    
    try:
        # Создаем адаптер
        adapter = ModelAdapter(model_path, device)
        print("✅ ModelAdapter создан")
        
        # Создаем тестовое изображение
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print(f"✅ Тестовое изображение создано: {test_img.shape}")
        
        # Запускаем инференс
        results = adapter.infer([test_img])
        print(f"✅ Инференс выполнен, получено результатов: {len(results)}")
        
        if results:
            result = results[0]
            print(f"📊 Результат:")
            print(f"  - Детекций: {result.get('detections', 0)}")
            print(f"  - Уверенность: {result.get('confidence', 0.0)}")
            print(f"  - Маски: {len(result.get('masks', []))}")
            print(f"  - BBoxes: {len(result.get('bboxes', []))}")
            print(f"  - Центроиды: {len(result.get('centroids', []))}")
            
            if result.get('masks'):
                print(f"🎭 Первая маска: {type(result['masks'][0])}, размер: {len(result['masks'][0])}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
