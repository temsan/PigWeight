#!/usr/bin/env python3
import os
import sys

print("=== Детальная диагностика CUDA ===")

# Проверяем переменные окружения
print("\n1. Переменные окружения:")
device_env = os.getenv("DEVICE")
print(f"DEVICE={device_env}")

# Проверяем torch
print("\n2. PyTorch проверка:")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version in PyTorch: {torch.version.cuda}")

    # Проверяем CUDA доступность
    print("\n3. CUDA доступность:")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Тестируем логику из api/app.py
    print("\n4. Тестируем логику из api/app.py:")
    _cuda_ok = bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())
    print(f"_cuda_ok: {_cuda_ok}")

    DEVICE = device_env or ("cuda:0" if _cuda_ok else "cpu")
    print(f"DEVICE after first check: {DEVICE}")

    # Вторая проверка
    if DEVICE and DEVICE.startswith('cuda') and not (torch.cuda.is_available() if hasattr(torch, 'cuda') else False):
        print("⚠️ CUDA недоступна во второй проверке!")
        DEVICE = 'cpu'
    else:
        print("✅ CUDA доступна во второй проверке")

    print(f"Final DEVICE: {DEVICE}")

except Exception as e:
    print(f"❌ Ошибка при импорте torch: {e}")
    sys.exit(1)

# Проверяем CUDA_PATH
print("\n5. CUDA Toolkit:")
cuda_path = os.getenv("CUDA_PATH")
if cuda_path:
    print(f"CUDA_PATH: {cuda_path}")
else:
    print("CUDA_PATH не установлена")

print("\n6. Рекомендации:")
if torch.cuda.is_available():
    print("✅ CUDA доступна - можно использовать GPU")
    print("   Рекомендуется: DEVICE=cuda:0, USE_HALF=true")
else:
    print("❌ CUDA недоступна - используйте CPU")
    print("   Рекомендуется: DEVICE=cpu, USE_HALF=false")
