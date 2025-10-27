#!/usr/bin/env python3
"""Проверка прогресса обработки видео"""
import os
from pathlib import Path
from datetime import datetime

print("\n📊 Статус обработки видео")
print("=" * 60)

# Проверяем результаты
results_dir = Path("results")
if results_dir.exists():
    json_files = list(results_dir.glob("*.json"))
    if json_files:
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        print(f"✅ Найден результат: {latest.name}")
        print(f"   Время создания: {mtime.strftime('%H:%M:%S')}")
        
        # Читаем результат
        import json
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📈 Результаты:")
        print(f"   Кадров обработано: {data.get('frames_processed', 0)}")
        print(f"   Актов найдено: {data.get('act_stats', {}).get('completed_acts_count', 0)}")
        print(f"   Проходов: {data.get('crossing_stats', {}).get('total_crossings', 0)}")
    else:
        print("⏳ Результаты еще не готовы...")
else:
    print("⏳ Папка results не создана, обработка еще не началась...")

# Проверяем процесс
print(f"\n🔍 Проверка процесса:")
print(f"   Ищем python процессы с console_app.py...")

import psutil
found = False
for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
    try:
        if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
            cmdline = ' '.join(proc.info['cmdline'])
            if 'console_app.py' in cmdline:
                runtime = datetime.now().timestamp() - proc.info['create_time']
                print(f"   ✅ Найден процесс PID {proc.info['pid']}")
                print(f"      Время работы: {runtime/60:.1f} минут")
                found = True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not found:
    print("   ⚠️ Процесс не найден (возможно завершился)")

print("\n" + "=" * 60)
