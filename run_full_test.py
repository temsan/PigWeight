#!/usr/bin/env python3
"""
Полное тестирование системы отслеживания свиней
"""

import time
import sys
from pathlib import Path

def wait_for_supabase():
    """Ждем запуска Supabase"""
    print("⏳ Ожидание запуска Supabase...")
    max_attempts = 15
    
    for attempt in range(max_attempts):
        try:
            import requests
            response = requests.get("http://localhost:8000/rest/v1/", timeout=2)
            if response.status_code in [200, 401, 404]:
                print("✅ Supabase доступен!")
                return True
        except:
            pass
        
        print(f"   Попытка {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    return False

def main():
    print("\n" + "="*80)
    print("🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ")
    print("="*80 + "\n")
    
    # Ждем Supabase
    if not wait_for_supabase():
        print("❌ Supabase не запустился")
        return False
    
    # 1. Проверка системы
    print("\n1️⃣ Проверка готовности системы...")
    import subprocess
    result = subprocess.run([sys.executable, "check_system.py"], capture_output=True, text=True)
    print(result.stdout)
    
    # 2. Тест интеграции
    print("\n2️⃣ Тест интеграции...")
    result = subprocess.run([sys.executable, "test_integration.py"], capture_output=True, text=True)
    print(result.stdout)
    
    # 3. Проверка видео
    print("\n3️⃣ Проверка видео...")
    video_path = Path("uploads/test_video.mp4")
    if video_path.exists():
        print(f"✅ Видео: {video_path} ({video_path.stat().st_size / 1024:.1f} KB)")
    else:
        print("❌ Видео не найдено")
    
    print("\n" + "="*80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*80)
    
    return True

if __name__ == '__main__':
    main()
