"""
Актуализированный скрипт для конвертации модели в ONNX формат.

Скрипт автоматически использует конфигурацию из `core/config.py`
для определения пути к модели и размера изображения.

Использование:
python scripts/convert_to_onnx.py
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path для импорта `core`
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

try:
    from ultralytics import YOLO
    from core.config import CONFIG
    print("✅ Зависимости (Ultralytics, PyTorch) и конфигурация загружены.")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Пожалуйста, убедитесь, что все зависимости установлены: pip install -r requirements.txt")
    sys.exit(1)

def convert_model_from_config_to_onnx():
    """
    Конвертирует модель, указанную в `core/config.py`, в формат ONNX.
    """
    try:
        # 1. Получаем параметры из конфигурации
        base_model_path = CONFIG.get("MODEL_PATH")
        if not base_model_path:
            print("❌ Ошибка: `MODEL_PATH` не найден в `core/config.py`.")
            return

        input_pt_path = Path(f"{base_model_path}.pt")
        output_onnx_path = Path(f"{base_model_path}.onnx")
        img_size = CONFIG.get("IMG_SIZE", 960)
        opset_version = 12  # Рекомендуемая версия для совместимости

        print("\n" + "="*60)
        print("🦄 Конвертер PyTorch -> ONNX")
        print("="*60)
        print(f"▶️  Начинаем конвертацию...")
        print(f"  • Входная модель (.pt): {input_pt_path}")
        print(f"  • Выходная модель (.onnx): {output_onnx_path}")
        print(f"  • Размер изображения: {img_size}x{img_size}")
        print(f"  • Opset: {opset_version}")
        print("="*60 + "\n")

        if not input_pt_path.exists():
            print(f"❌ Ошибка: Входная модель не найдена по пути '{input_pt_path}'")
            return

        # 2. Загружаем модель YOLO
        print("🔄 Загрузка PyTorch модели...")
        model = YOLO(input_pt_path)
        print("✅ Модель успешно загружена.")

        # 3. Экспортируем модель в формат ONNX
        print("\n🔄 Экспорт в ONNX...")
        model.export(
            format='onnx',
            imgsz=img_size,
            opset=opset_version,
            simplify=True,      # Включаем упрощение для лучшей производительности
            verbose=False       # Оставляем вывод чистым
        )

        if not output_onnx_path.exists():
             # Ultralytics может добавлять суффикс, проверим это
             output_onnx_path = next(Path(base_model_path).parent.glob(f"{Path(base_model_path).name}*.onnx"), None)
             if not output_onnx_path or not output_onnx_path.exists():
                raise FileNotFoundError("Не удалось найти созданный ONNX файл.")


        print("\n" + "="*60)
        print("🎉 Конвертация успешно завершена!")
        print("="*60)
        print(f"📄 Ваша ONNX модель готова: {output_onnx_path}")
        print("\n💡 Теперь система автоматически будет использовать ее при работе на CPU.")

    except Exception as e:
        print(f"\n❌ Произошла ошибка во время конвертации: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    convert_model_from_config_to_onnx()