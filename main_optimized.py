#!/usr/bin/env python3
"""
PigWeight - Оптимизированная точка входа с автоматическим выбором рантайма
Поддержка профилей производительности: ULTRA_PERFORMANCE, BALANCED, POWER_SAVING, CPU_ONLY
"""

import os
import sys
import urllib.request
import subprocess
import logging
import argparse
from pathlib import Path

def print_banner():
    """Отображает баннер системы"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🐷 PigWeight v3.0                        ║
║              Система видеоаналитики свиней                    ║
║                                                              ║
║  🚀 Автоматический выбор оптимального рантайма               ║
║  ⚡ Поддержка CUDA, ONNX Runtime, DirectML                  ║
║  🎯 Профили производительности                               ║
╚══════════════════════════════════════════════════════════════╝
    """)

def main():
    """Главная функция с автоматическим выбором рантайма"""
    
    # Аргументы командной строки
    parser = argparse.ArgumentParser(
        description='PigWeight - Система видеоаналитики с автоматическим выбором рантайма',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main_optimized.py                    # Автоматический выбор (рекомендуемый)
  python main_optimized.py --runtime pytorch  # Принудительно PyTorch (профиль авто)
  python main_optimized.py --runtime onnx-gpu # Принудительно ONNX GPU (профиль авто)
  python main_optimized.py --install          # Установка зависимостей

Профиль производительности всегда выбирается автоматически на основе вашего железа.
        """
    )
    
    parser.add_argument(
        '--runtime', 
        choices=['auto', 'pytorch', 'onnx-gpu', 'onnx-cpu', 'cpu'], 
        default='auto',
        help='Выбор рантайма (по умолчанию: auto - автоматический выбор)'
    )
    
    parser.add_argument(
        '--install', 
        action='store_true',
        help='Установить все зависимости и выйти'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Режим отладки с подробными логами'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='Порт для запуска сервера (по умолчанию: 8000)'
    )
    
    args = parser.parse_args()
    
    # Отображаем баннер
    if not args.install:
        print_banner()
    
    # Установка зависимостей
    if args.install:
        install_requirements()
        return
    
    # Устанавливаем переменные окружения для debug режима
    if args.debug:
        os.environ['DEBUG'] = 'true'
        os.environ['RELOAD'] = 'true'
    
    # Устанавливаем порт
    if args.port != 8000:
        os.environ['PORT'] = str(args.port)
    
    try:
        # Импортируем функции конфигурации
        from core.config import detect_optimal_runtime, apply_runtime_optimizations, apply_performance_profile
        
        print("🔍 Анализ системы и выбор оптимального рантайма...")
        
        # Выбираем оптимальный рантайм и профиль
        if args.runtime == 'auto':
            # Автоматический выбор (рекомендуемый)
            runtime_info = detect_optimal_runtime()
            apply_runtime_optimizations(runtime_info)
            
            print(f"✅ Выбран рантайм: {runtime_info['runtime']}")
            print(f"🎯 Устройство: {runtime_info['device']}")
            print(f"⚡ Профиль: {runtime_info['profile']} (автоматически)")
            print("📋 Причины выбора:")
            for reason in runtime_info['reasons']:
                print(f"   • {reason}")
                
        else:
            # Ручной выбор рантайма (профиль всё равно автоматический)
            print(f"🔧 Ручной выбор рантайма: {args.runtime}")
            
            # Определяем оптимальный профиль даже при ручном рантайме
            runtime_info = detect_optimal_runtime()
            
            if args.runtime == 'pytorch':
                runtime_info['runtime'] = 'pytorch'
                os.environ['DEVICE'] = 'auto'
                os.environ['PREFER_ONNX'] = 'false'
            elif args.runtime == 'onnx-gpu':
                runtime_info['runtime'] = 'onnx-gpu'
                os.environ['DEVICE'] = 'cuda:0'
                os.environ['PREFER_ONNX'] = 'true'
                os.environ['ONNX_PROVIDER'] = 'CUDAExecutionProvider'
            elif args.runtime == 'onnx-cpu':
                runtime_info['runtime'] = 'onnx-cpu'
                runtime_info['profile'] = 'CPU_ONLY'  # Принудительно CPU профиль
                os.environ['DEVICE'] = 'cpu'
                os.environ['PREFER_ONNX'] = 'true'
                os.environ['ONNX_PROVIDER'] = 'CPUExecutionProvider'
            elif args.runtime == 'cpu':
                runtime_info['runtime'] = 'cpu'
                runtime_info['profile'] = 'CPU_ONLY'
                os.environ['DEVICE'] = 'cpu'
                os.environ['USE_HALF'] = 'false'
            
            apply_runtime_optimizations(runtime_info)
            print(f"✅ Профиль: {runtime_info['profile']} (автоматически)")
        
        print("\n" + "="*60)
        
        # Импортируем и запускаем основное приложение
        from main import main as run_main
        run_main()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Попробуйте установить зависимости: python main_optimized.py --install")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def install_requirements():
    """Улучшенная установка зависимостей с проверками"""
    print("📦 Установка зависимостей PigWeight...")
    
    try:
        requirements_path = Path(__file__).parent / 'requirements.txt'
        if not requirements_path.exists():
            print("❌ Файл requirements.txt не найден!")
            return False
            
        print("📋 Установка основных зависимостей...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ])
        
        print("🔍 Проверка установленных библиотек...")
        
        # Проверяем PyTorch
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ PyTorch с CUDA: {gpu_name}")
            else:
                print("✅ PyTorch (только CPU)")
        except ImportError:
            print("⚠️ PyTorch не установлен")
        
        # Проверяем ONNX Runtime
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                print("✅ ONNX Runtime с GPU поддержкой")
            else:
                print("✅ ONNX Runtime (только CPU)")
            print(f"   Доступные провайдеры: {', '.join(providers)}")
        except ImportError:
            print("⚠️ ONNX Runtime не установлен")
            print("💡 Для установки GPU версии: pip install onnxruntime-gpu")
        
        # Проверяем другие зависимости
        deps = ['ultralytics', 'opencv-python', 'fastapi', 'uvicorn', 'psutil']
        for dep in deps:
            try:
                __import__(dep.replace('-', '_'))
                print(f"✅ {dep}")
            except ImportError:
                print(f"❌ {dep}")
        
        print("\n🎉 Установка завершена!")
        print("🚀 Запуск: python main_optimized.py")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки: {e}")
        print("💡 Попробуйте:")
        print("   pip install --upgrade pip")
        print("   pip install fastapi uvicorn torch ultralytics opencv-python")
        return False

if __name__ == '__main__':
    main()