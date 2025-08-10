#!/usr/bin/env python3
"""
Стартовый скрипт для PigWeight GPU-Accelerated Video System
Проверяет зависимости и запускает систему
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Печать приветственного баннера"""
    print("\n" + "="*70)
    print("🚀 PigWeight GPU-Accelerated Video Processing System")
    print("="*70)
    print("📊 Features: CUDA • Frame Indexing • Batch Inference • WebRTC")
    print("⚡ Maximum Performance Video API")
    print("="*70 + "\n")

def check_python_version():
    """Проверка версии Python"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("⚠️  Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_directories():
    """Создание необходимых директорий"""
    print("\n📁 Setting up directories...")
    
    required_dirs = [
        'uploads',
        'frame_indices',
        'static',
        'logs'
    ]
    
    for directory in required_dirs:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   📂 Created: {directory}")
        else:
            print(f"   ✅ Exists: {directory}")
    
    return True

def check_dependencies():
    """Проверка основных зависимостей"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('fastapi', 'FastAPI'),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - MISSING")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements_gpu.txt")
        return False
    
    return True

def check_gpu():
    """Проверка GPU доступности"""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"   ✅ CUDA Available")
            print(f"   🔥 GPU: {device_name}")
            print(f"   💾 Memory: {device_memory:.1f}GB")
            print(f"   📊 Devices: {device_count}")
            
            return True
        else:
            print(f"   ⚠️  CUDA not available - will use CPU")
            return False
    except Exception as e:
        print(f"   ❌ GPU check failed: {e}")
        return False

def test_basic_functionality():
    """Тестирование базовой функциональности"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Импорт нашего модуля
        from gpu_video_processor import GPUVideoProcessor
        
        # Инициализация процессора
        processor = GPUVideoProcessor(
            use_cuda=True,
            max_cache_size=10,
            batch_size=2,
            index_keyframes_only=True
        )
        
        print(f"   ✅ GPU Processor initialized")
        print(f"   💻 Device: {processor.device}")
        print(f"   🔥 CUDA: {processor.use_cuda}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def start_server(port=8000):
    """Запуск GPU сервера"""
    print(f"\n🚀 Starting GPU server on port {port}...")
    
    try:
        # Проверяем, что порт свободен
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print(f"   ⚠️  Port {port} is already in use")
            return False
        
        # Запускаем сервер
        print("   📡 Starting FastAPI server...")
        print(f"   🌐 Server will be available at: http://localhost:{port}")
        print("   📊 API Documentation: http://localhost:8000/docs")
        print("   🎮 WebRTC Client: http://localhost:8000/webrtc")
        print("\n   Press Ctrl+C to stop the server\n")
        
        # Импорт и запуск
        from gpu_endpoints import create_gpu_app
        import uvicorn
        
        app = create_gpu_app()
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=1,  # GPU requires single worker
            access_log=True,
            log_level="info"
        )
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
        return True
    except Exception as e:
        print(f"   ❌ Server start failed: {e}")
        return False

def print_usage_info():
    """Печать информации по использованию"""
    print("\n💡 USAGE INFORMATION:")
    print("-" * 50)
    print("🔧 Configuration:")
    print("   • Edit gpu_video_processor.py to customize GPU settings")
    print("   • Modify gpu_endpoints.py to add new API endpoints")
    print("   • Check requirements_gpu.txt for dependencies")
    print("\n📊 API Endpoints:")
    print("   • POST /api/gpu/video/upload - Upload video with GPU indexing")
    print("   • GET /api/gpu/video/{id}/frame - Ultra-fast frame access")
    print("   • POST /api/gpu/video/{id}/batch_inference - Batch processing")
    print("   • GET /api/gpu/video/{id}/stream - GPU-accelerated streaming")
    print("   • GET /api/gpu/stats - Performance statistics")
    print("\n🎮 WebRTC:")
    print("   • POST /api/webrtc/start - Start WebRTC stream")
    print("   • GET /webrtc - WebRTC client interface")
    print("\n🧪 Testing:")
    print("   • python simple_gpu_test.py - Basic functionality test")
    print("   • python test_gpu_performance.py - Comprehensive benchmarks")

def main():
    """Главная функция"""
    print_banner()
    
    # Системные проверки
    checks = [
        ("Python Version", check_python_version),
        ("Directories", check_directories),
        ("Dependencies", check_dependencies),
        ("GPU Support", check_gpu),
        ("Functionality", test_basic_functionality),
    ]
    
    print("🔍 SYSTEM CHECKS:")
    print("-" * 30)
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            result = check_func()
            if not result and check_name in ["Python Version", "Dependencies"]:
                all_passed = False
        except Exception as e:
            print(f"   ❌ {check_name} failed: {e}")
            if check_name in ["Python Version", "Dependencies"]:
                all_passed = False
    
    print("\n" + "="*50)
    
    if not all_passed:
        print("❌ SYSTEM NOT READY")
        print("💡 Please fix the issues above before starting the server")
        return 1
    
    print("✅ SYSTEM READY")
    print("🎯 All checks passed! Ready to start GPU server.")
    
    # Показываем информацию по использованию
    print_usage_info()
    
    # Предлагаем запустить сервер
    try:
        response = input("\n🚀 Start GPU server now? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            start_server()
        else:
            print("\n💡 To start manually: python gpu_endpoints.py")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
