#!/usr/bin/env python3
"""Показать текущие результаты"""

from pig_tracking.database import DatabaseManager
from pathlib import Path
import json

print("="*80)
print("📊 Текущие результаты обработки")
print("="*80)
print()

# 1. Проверка БД
print("1️⃣ База данных:")
try:
    db = DatabaseManager()
    stats = db.get_stats()
    print(f"   ✅ Подключено к {stats['database_url']}")
    print(f"   📋 Актов взвешивания: {stats['total_acts']}")
    print(f"   🔄 Пересечений линий: {stats['total_crossings']}")
    
    if stats['last_act']:
        print(f"\n   Последний акт:")
        print(f"   • Начало: {stats['last_act']['started_at']}")
        print(f"   • Конец: {stats['last_act']['ended_at']}")
        print(f"   • Вход слева: {stats['last_act']['left_count']}")
        print(f"   • Выход справа: {stats['last_act']['right_count']}")
        print(f"   • Пик: {stats['last_act']['peak_count']}")
except Exception as e:
    print(f"   ❌ БД недоступна: {e}")

print()

# 2. Проверка JSON результатов
print("2️⃣ JSON результаты:")
results_dir = Path('results')
if results_dir.exists():
    files = sorted(results_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        latest = files[0]
        print(f"   📄 Последний файл: {latest.name}")
        print(f"   📅 Дата: {latest.stat().st_mtime}")
        
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"\n   Статистика:")
            print(f"   • Кадров: {data.get('frames_processed', 0)}")
            print(f"   • Актов: {data.get('act_stats', {}).get('completed_acts_count', 0)}")
            print(f"   • Пересечений: {data.get('crossing_stats', {}).get('total_crossings', 0)}")
    else:
        print("   ⏳ Результаты еще не созданы")
else:
    print("   ⏳ Папка results не существует")

print()

# 3. Статус обработки
print("3️⃣ Текущая обработка:")
import subprocess
result = subprocess.run(['powershell', '-Command', 
    'Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.WorkingSet -gt 500MB} | Select-Object Id, @{Name="Memory(GB)";Expression={[math]::Round($_.WS / 1GB, 2)}}'],
    capture_output=True, text=True)

if result.stdout.strip():
    print("   🟢 Обработка идет")
    print(result.stdout)
else:
    print("   ⏹️ Нет активной обработки")

print()
print("="*80)
