#!/usr/bin/env python3
"""
Конвертация модели YOLO в ONNX формат для оптимизации CPU inference.

Использование:
python scripts/convert_to_onnx.py --model_path models/pig_yolo11-seg.pt --output_path models/pig_yolo11-seg.onnx --img_size 960

Преимущества ONNX:
- Оптимизированное выполнение на CPU
- Кросс-платформенная поддержка
- Графовая оптимизация
- Поддержка различных execution providers
"""

import argparse
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from ultralytics import YOLO
    print("✅ PyTorch и Ultralytics импортированы успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установите зависимости: pip install torch torchvision ultralytics")
    sys.exit(1)


def convert_yolo_to_onnx(model_path: str, output_path: str, img_size: int = 960,
                        opset_version: int = 11, simplify: bool = True):
    """
    Конвертация YOLO модели в ONNX формат.

    Args:
        model_path: Путь к PyTorch модели (.pt)
        output_path: Путь для сохранения ONNX модели
        img_size: Размер входного изображения
        opset_version: Версия ONNX opset
        simplify: Упростить модель с onnx-simplifier
    """

    print(f"🚀 Начинаем конвертацию модели...")
    print(f"📁 Модель: {model_path}")
    print(f"📤 Выход: {output_path}")
    print(f"📏 Размер: {img_size}x{img_size}")
    print(f"🔧 Opset: {opset_version}")

    # Проверяем существование входной модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return False

    try:
        # Загружаем модель
        print("📥 Загружаем PyTorch модель...")
        model = YOLO(model_path)

        # Создаем фиктивный вход для ONNX
        dummy_input = torch.randn(1, 3, img_size, img_size)

        # Конвертируем в ONNX
        print("🔄 Конвертируем в ONNX...")
        model.model.eval()

        # Экспортируем в ONNX
        success = model.export(
            format='onnx',
            imgsz=img_size,
            opset=opset_version,
            simplify=simplify,
            dynamic=False,  # Фиксированный размер батча
            verbose=True
        )

    except Exception as e:
        print(f"❌ Ошибка при конвертации: {e}")
        import traceback
        traceback.print_exc()
        return False

        if success:
            print("✅ Конвертация завершена успешно!")
            print(f"📄 ONNX модель сохранена: {output_path}")

            # Проверяем размер файла
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(".1f")
            # Валидируем ONNX модель
            validate_onnx_model(output_path)

            return True
        else:
            print("❌ Ошибка конвертации")
            return False

    except Exception as e:
        print(f"❌ Ошибка при конвертации: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_onnx_model(onnx_path: str):
    """Валидация ONNX модели"""
    try:
        import onnxruntime as ort
        print("🔍 Валидируем ONNX модель...")

        # Создаем inference session
        session = ort.InferenceSession(onnx_path)

        # Получаем информацию о входах/выходах
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print("📊 Информация о модели:")
        print(f"   Входы: {len(inputs)}")
        for i, inp in enumerate(inputs):
            print(f"     {i}: {inp.name} {inp.shape} {inp.type}")

        print(f"   Выходы: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"     {i}: {out.name} {out.shape} {out.type}")

        # Тестовый inference
        print("🧪 Тестируем inference...")
        import numpy as np
        dummy_input = np.random.randn(1, 3, 960, 960).astype(np.float32)

        # Запускаем inference
        result = session.run(None, {inputs[0].name: dummy_input})

        print("✅ Inference прошел успешно!")
        print(f"   Выходов: {len(result)}")
        for i, res in enumerate(result):
            print(f"   Выход {i}: {res.shape}")

    except ImportError:
        print("⚠️  onnxruntime не установлен, пропускаем валидацию")
        print("   Установите: pip install onnxruntime")
    except Exception as e:
        print(f"⚠️  Ошибка валидации: {e}")


def optimize_onnx_model(onnx_path: str):
    """Оптимизация ONNX модели"""
    try:
        import onnx
        from onnx import optimizer
        print("🔧 Оптимизируем ONNX модель...")

        # Загружаем модель
        model = onnx.load(onnx_path)

        # Применяем оптимизации
        optimized_model = optimizer.optimize(model, [
            'eliminate_deadend',
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_unused_initializer',
            'fuse_add_bias_into_conv',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv'
        ])

        # Сохраняем оптимизированную модель
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        onnx.save(optimized_model, optimized_path)

        print(f"✅ Оптимизированная модель сохранена: {optimized_path}")

        # Сравниваем размеры
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
        print(".1f")
        return optimized_path

    except ImportError:
        print("⚠️  onnx не установлен, пропускаем оптимизацию")
        print("   Установите: pip install onnx onnxoptimizer")
        return onnx_path
    except Exception as e:
        print(f"⚠️  Ошибка оптимизации: {e}")
        return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX format")
    parser.add_argument("--model_path", required=True, help="Path to PyTorch model (.pt)")
    parser.add_argument("--output_path", help="Path to save ONNX model")
    parser.add_argument("--img_size", type=int, default=960, help="Input image size")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")

    args = parser.parse_args()

    # Автоматически генерируем путь вывода если не указан
    if not args.output_path:
        model_name = Path(args.model_path).stem
        args.output_path = f"models/{model_name}.onnx"

    # Создаем директорию если нужно
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("=" * 60)
    print("🦄 YOLO to ONNX Converter")
    print("=" * 60)

    # Конвертируем модель
    success = convert_yolo_to_onnx(
        args.model_path,
        args.output_path,
        args.img_size,
        args.opset,
        args.simplify
    )

    if success and args.optimize:
        args.output_path = optimize_onnx_model(args.output_path)

    if success:
        print("\n" + "=" * 60)
        print("✅ Конвертация завершена успешно!")
        print("=" * 60)
        print(f"📄 ONNX модель: {args.output_path}")
        print("\n💡 Использование в коде:"        print("   from services.model_adapter import ModelAdapter"        print(f"   adapter = ModelAdapter('{args.output_path}')")
        print("   results = adapter.infer([image])")
    else:
        print("\n❌ Конвертация завершилась с ошибкой")
        sys.exit(1)


if __name__ == "__main__":
    main()
