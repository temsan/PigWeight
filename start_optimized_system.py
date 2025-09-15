#!/usr/bin/env python3
"""
Оптимизированный запуск PigWeight системы
Применяет все рекомендации по производительности
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Запуск оптимизированной PigWeight системы...")
    
    # Устанавливаем оптимизированные переменные окружения
    optimized_env = {
        # Основные настройки
        "USE_OPTIMIZED_PREPROCESSING": "true",
        "PREPROCESSING_METHOD": "adaptive",
        "ANTI_LETTERBOX": "false",
        
        # Производительность
        "FPS": "12",  # Уменьшенный FPS для плавности
        "BATCH_SIZE": "8",  # Увеличенный batch size
        "MAX_WAIT_MS": "50",
        "BROADCAST_MIN_INTERVAL": "0.1",  # Уменьшенная частота обновлений
        
        # Стабильность (CPU вместо CUDA для избежания проблем)
        "DEVICE": "cpu",
        "USE_HALF": "false",
        
        # Кэширование
        "FRAME_BROKER_CACHE": "16",
        "RESULTS_TTL_SECONDS": "30",
        
        # Отключаем hot reload для production
        "RELOAD": "false",
        "DEBUG": "false",
    }
    
    # Обновляем переменные окружения
    current_env = os.environ.copy()
    current_env.update(optimized_env)
    
    print("✅ Применены оптимизированные настройки:")
    for key, value in optimized_env.items():
        print(f"   {key}={value}")
    
    print("\n🎯 Ожидаемые улучшения:")
    print("   • Плавное воспроизведение видео (без тикания)")
    print("   • Стабильная работа на CPU")
    print("   • Оптимизированная предобработка")
    print("   • Уменьшенная нагрузка на UI")
    
    print("\n🌐 Система будет доступна на: http://localhost:8000")
    print("📊 Для мониторинга производительности смотрите логи\n")
    
    try:
        # Запускаем основную систему
        subprocess.run([sys.executable, "main.py"], env=current_env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Система остановлена пользователем")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка запуска: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()