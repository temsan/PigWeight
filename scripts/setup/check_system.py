#!/usr/bin/env python3
"""
Скрипт проверки готовности системы отслеживания свиней
"""

import os
import sys
import subprocess
from pathlib import Path

def check_item(name, check_func):
    """Проверяет один элемент системы"""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}")
            return True
        else:
            print(f"❌ {name}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

def check_docker():
    """Проверяет Docker"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_docker_compose():
    """Проверяет Docker Compose"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_supabase_running():
    """Проверяет запущен ли Supabase"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        return 'postgres' in result.stdout and result.returncode == 0
    except Exception:
        return False

def check_env_file():
    """Проверяет файл .env"""
    return Path('.env').exists()

def check_supabase_key():
    """Проверяет SUPABASE_KEY в .env"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return bool(os.getenv('SUPABASE_KEY'))
    except ImportError:
        return False

def check_model_file():
    """Проверяет файл модели"""
    model_path = os.getenv('MODEL_PATH', 'models/pig_yolo11-seg.v4.pt')
    return Path(model_path).exists()

def check_uploads_dir():
    """Проверяет папку uploads"""
    return Path('uploads').exists()

def check_python_packages():
    """Проверяет установленные пакеты"""
    try:
        import supabase
        import cv2
        import ultralytics
        return True
    except ImportError:
        return False

def check_database_connection():
    """Проверяет подключение к базе данных"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from pig_tracking.database import DatabaseManager
        db = DatabaseManager()
        stats = db.get_stats()
        return True
    except Exception:
        return False

def main():
    """Главная функция"""
    print("🔍 Проверка готовности системы отслеживания свиней")
    print("=" * 60)
    
    checks = [
        ("Docker установлен", check_docker),
        ("Docker Compose установлен", check_docker_compose),
        ("Файл .env существует", check_env_file),
        ("SUPABASE_KEY настроен", check_supabase_key),
        ("Supabase запущен", check_supabase_running),
        ("Папка uploads существует", check_uploads_dir),
        ("Модель YOLO существует", check_model_file),
        ("Python пакеты установлены", check_python_packages),
        ("Подключение к базе данных", check_database_connection),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_item(name, check_func)
        results.append((name, result))
    
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Результат: {passed}/{total} проверок пройдено")
    
    if passed == total:
        print("🎉 Система готова к работе!")
        print("\nЗапустите: python console_app.py")
        return 0
    else:
        print("\n⚠️ Система не готова. Исправьте ошибки:")
        
        for name, result in results:
            if not result:
                print(f"  • {name}")
        
        print("\n💡 Рекомендации:")
        
        if not results[0][1]:  # Docker
            print("  • Установите Docker: https://docs.docker.com/get-docker/")
        
        if not results[1][1]:  # Docker Compose
            print("  • Установите Docker Compose: https://docs.docker.com/compose/install/")
        
        if not results[2][1]:  # .env
            print("  • Скопируйте файл: cp .env.example .env")
        
        if not results[4][1]:  # Supabase
            print("  • Запустите Supabase: docker-compose up -d")
        
        if not results[5][1]:  # uploads
            print("  • Создайте папку: mkdir uploads")
        
        if not results[6][1]:  # Model
            print("  • Проверьте путь к модели в .env")
        
        if not results[7][1]:  # Packages
            print("  • Установите пакеты: pip install -r requirements-pig-tracking.txt")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())