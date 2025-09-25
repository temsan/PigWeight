#!/usr/bin/env python3
"""
Скрипт для тестирования системы с локальными видеофайлами
"""

import os
import sys
import requests
import time
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_upload_video(video_path: str, server_url: str = "http://localhost:8000"):
    """Тестирование загрузки видеофайла"""
    print(f"🎬 Тестирование загрузки: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Файл не найден: {video_path}")
        return False
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
            response = requests.post(f"{server_url}/api/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Файл успешно загружен:")
            print(f"   - Имя: {result.get('filename')}")
            print(f"   - Размер: {result.get('size_mb')} MB")
            print(f"   - FPS: {result.get('fps')}")
            print(f"   - Длительность: {result.get('duration')} сек")
            return result.get('file_path')
        else:
            print(f"❌ Ошибка загрузки: {response.status_code}")
            print(f"   Ответ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Исключение при загрузке: {e}")
        return False

def test_system_status(server_url: str = "http://localhost:8000"):
    """Тестирование статуса системы"""
    print("🔍 Проверка статуса системы...")
    
    try:
        response = requests.get(f"{server_url}/api/system/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print("✅ Система работает:")
            print(f"   - Статус: {status.get('status')}")
            print(f"   - Модель: {status.get('model_path')}")
            print(f"   - Устройство: {status.get('device')}")
            return True
        else:
            print(f"❌ Ошибка получения статуса: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Исключение при проверке статуса: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование системы PigWeight с локальными файлами")
    print("=" * 60)
    
    # Проверяем статус системы
    if not test_system_status():
        print("❌ Система недоступна. Убедитесь, что сервер запущен.")
        return
    
    print()
    
    # Ищем видеофайлы для тестирования
    uploads_dir = Path("uploads")
    temp_dir = Path("temp")
    
    test_files = []
    
    # Ищем файлы в uploads
    if uploads_dir.exists():
        for file_path in uploads_dir.glob("*.mp4"):
            if file_path.stat().st_size < 100 * 1024 * 1024:  # Меньше 100MB
                test_files.append(str(file_path))
                if len(test_files) >= 3:  # Максимум 3 файла
                    break
    
    # Ищем файлы в temp
    if temp_dir.exists() and len(test_files) < 3:
        for file_path in temp_dir.glob("*.mp4"):
            if file_path.stat().st_size < 100 * 1024 * 1024:  # Меньше 100MB
                test_files.append(str(file_path))
                if len(test_files) >= 3:  # Максимум 3 файла
                    break
    
    if not test_files:
        print("❌ Не найдены подходящие видеофайлы для тестирования")
        print("   Ищите файлы .mp4 размером менее 100MB в папках uploads/ и temp/")
        return
    
    print(f"📁 Найдено {len(test_files)} файлов для тестирования:")
    for file_path in test_files:
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"   - {os.path.basename(file_path)} ({size_mb:.1f} MB)")
    
    print()
    
    # Тестируем загрузку файлов
    uploaded_files = []
    for file_path in test_files:
        result = test_upload_video(file_path)
        if result:
            uploaded_files.append(result)
        print()
        time.sleep(1)  # Небольшая пауза между загрузками
    
    if uploaded_files:
        print(f"✅ Успешно загружено {len(uploaded_files)} файлов")
        print("🎯 Система готова к работе с локальными файлами!")
    else:
        print("❌ Не удалось загрузить ни одного файла")

if __name__ == "__main__":
    main()
