#!/usr/bin/env python3
"""
Скрипт быстрого запуска системы отслеживания свиней
"""

import os
import sys
import subprocess
from pathlib import Path

def check_supabase():
    """Проверяет запущен ли Supabase"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if 'supabase' in result.stdout or 'postgres' in result.stdout:
            print("✅ Supabase запущен")
            return True
        else:
            print("❌ Supabase не запущен")
            return False
    except Exception:
        print("❌ Не удалось проверить статус Docker")
        return False

def start_supabase():
    """Запускает Supabase"""
    print("🚀 Запускаем Supabase...")
    try:
        result = subprocess.run(['docker-compose', 'up', '-d'], check=True)
        print("✅ Supabase запущен")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ошибка запуска Supabase")
        return False
    except FileNotFoundError:
        print("❌ Docker Compose не найден")
        return False

def install_dependencies():
    """Устанавливает зависимости"""
    print("📦 Проверяем зависимости...")
    try:
        import supabase
        print("✅ supabase установлен")
    except ImportError:
        print("📦 Устанавливаем зависимости...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-pig-tracking.txt'])

def check_env():
    """Проверяет файл .env"""
    if not Path('.env').exists():
        if Path('.env.example').exists():
            print("📄 Копируем .env.example в .env...")
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Файл .env создан")
        else:
            print("❌ Файл .env.example не найден")
            return False
    else:
        print("✅ Файл .env существует")
    return True

def main():
    """Главная функция"""
    print("🐷 Система отслеживания свиней")
    print("=" * 50)
    
    # Проверяем .env
    if not check_env():
        return 1
    
    # Устанавливаем зависимости
    install_dependencies()
    
    # Проверяем Supabase
    if not check_supabase():
        if not start_supabase():
            return 1
        
        # Ждем запуска
        print("⏳ Ждем запуска Supabase (10 секунд)...")
        import time
        time.sleep(10)
    
    # Запускаем основное приложение
    print("🚀 Запускаем консольное приложение...")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, 'console_app.py'] + sys.argv[1:])
    except KeyboardInterrupt:
        print("\n⏹️ Завершение работы...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())