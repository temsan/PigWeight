#!/usr/bin/env python3
import torch
import sys

print("=== CUDA Проверка ===")
print(f"PyTorch version: {torch.__version__}")

try:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA не доступна")
        print("Возможные причины:")
        print("1. Нет GPU в системе")
        print("2. CUDA драйверы не установлены")
        print("3. Несовместимая версия CUDA")
except Exception as e:
    print(f"❌ Ошибка при проверке CUDA: {e}")

print("\n=== Переменные окружения ===")
import os
for k, v in os.environ.items():
    if any(x in k.upper() for x in ['CUDA', 'GPU']):
        print(f"{k}={v}")
