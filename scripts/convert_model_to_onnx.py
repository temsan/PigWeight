#!/usr/bin/env python3
"""
Скрипт для конвертации YOLO модели в ONNX формат.
Решает проблемы с dtype конфликтами и улучшает производительность.
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

def convert_yolo_to_onnx(model_path: str, output_path: str = None, imgsz: int = 960):
    """Конвертирует YOLO модель в ONNX формат"""
    
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return False
        
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    print(f"🔄 Конвертируем модель: {model_path}")
    print(f"📁 Выходной файл: {output_path}")
    print(f"📐 Размер изображения: {imgsz}")
    
    try:
        from ultralytics import YOLO
        
        # Загружаем модель
        print("📂 Загружаем YOLO модель...")
        model = YOLO(model_path)
        
        # Экспортируем в ONNX
        print("⚙️ Экспортируем в ONNX...")
        success = model.export(
            format='onnx',
            imgsz=imgsz,
            opset=11,  # Совместимость с большинством систем
            simplify=True,  # Упрощаем граф для лучшей производительности
            dynamic=False,  # Фиксированный размер для оптимизации
            half=False,  # Принудительно float32 для избежания проблем
        )
        
        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✅ Конвертация успешна!")
            print(f"📊 Размер ONNX файла: {file_size:.1f} MB")
            
            # Проверяем работоспособность
            print("🧪 Тестируем ONNX модель...")
            test_onnx_model(output_path, imgsz)
            
            return True
        else:
            print("❌ Конвертация не удалась")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        return False


def test_onnx_model(onnx_path: str, imgsz: int):
    """Тестируем работоспособность ONNX модели"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print(f"🔍 Загружаем ONNX модель: {onnx_path}")
        
        # Создаем сессию
        session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']  # Используем CPU для стабильности
        )
        
        # Получаем информацию о входах/выходах
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        print(f"📥 Вход: {input_info.name}, форма: {input_info.shape}, тип: {input_info.type}")
        print(f"📤 Выходов: {len(output_info)}")
        
        # Создаем тестовое изображение
        if len(input_info.shape) == 4:  # NCHW формат
            test_input = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        else:
            print(f"⚠️ Неожиданная форма входа: {input_info.shape}")
            return False
            
        print(f"🧪 Тестовый вход: {test_input.shape}")
        
        # Запускаем инференс
        outputs = session.run(None, {input_info.name: test_input})
        
        print(f"✅ ONNX модель работает корректно!")
        print(f"📊 Выходы: {[out.shape for out in outputs]}")
        
        return True
        
    except ImportError:
        print("⚠️ ONNX Runtime не установлен, пропускаем тест")
        print("   Установите: pip install onnxruntime")
        return True
    except Exception as e:
        print(f"❌ Ошибка тестирования ONNX: {e}")
        return False


def main():
    """Главная функция"""
    print("🚀 Конвертация YOLO модели в ONNX")
    print("=" * 50)
    
    # Список моделей для конвертации
    models_to_convert = [
        "models/pig_yolo11-seg.v4.pt",
        "models/best.pt",  # На случай если захотим исправить и её
    ]
    
    successful_conversions = 0
    
    for model_path in models_to_convert:
        if os.path.exists(model_path):
            print(f"\n🎯 Конвертируем: {model_path}")
            print("-" * 40)
            
            if convert_yolo_to_onnx(model_path):
                successful_conversions += 1
                print(f"✅ {model_path} -> ONNX успешно")
            else:
                print(f"❌ {model_path} -> ONNX не удалось")
        else:
            print(f"⚠️ Пропускаем отсутствующую модель: {model_path}")
    
    print("\n" + "=" * 50)
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    print(f"✅ Успешно конвертировано: {successful_conversions}")
    print(f"📁 ONNX модели сохранены в папке models/")
    
    if successful_conversions > 0:
        print("\n💡 Следующие шаги:")
        print("1. Обновите конфигурацию для использования ONNX модели")
        print("2. Перезапустите сервер")
        print("3. Протестируйте работу интерфейса")
        
        print("\n🔧 Пример обновления конфигурации:")
        print('# В core/config.py измените MODEL_PATH на:')
        print('"MODEL_PATH": "models/pig_yolo11-seg.v4.onnx",')


if __name__ == "__main__":
    main()
