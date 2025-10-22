#!/usr/bin/env python3
"""
Тестирование прямого подключения к Supabase
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('SUPABASE_URL', 'http://localhost:8000')
anon_key = os.getenv('SUPABASE_KEY')
service_key = os.getenv('SUPABASE_SERVICE_KEY')

print("🧪 Тестирование прямого подключения к Supabase\n")

# Тест 1: Проверка доступности
print("1. Проверка доступности Kong...")
try:
    response = requests.get(f"{url}/")
    print(f"   ✅ Kong доступен: {response.status_code}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Тест 2: Запрос с anon ключом
print("\n2. Запрос с anon ключом...")
headers = {
    "apikey": anon_key,
    "Authorization": f"Bearer {anon_key}"
}
try:
    response = requests.get(f"{url}/rest/v1/weighing_acts", headers=headers)
    print(f"   Статус: {response.status_code}")
    print(f"   Ответ: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Тест 3: Запрос с service_role ключом
print("\n3. Запрос с service_role ключом...")
headers = {
    "apikey": service_key,
    "Authorization": f"Bearer {service_key}"
}
try:
    response = requests.get(f"{url}/rest/v1/weighing_acts", headers=headers)
    print(f"   Статус: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ Успешно! Записей: {len(response.json())}")
    else:
        print(f"   Ответ: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Тест 4: Проверка таблиц
print("\n4. Список таблиц...")
headers = {
    "apikey": service_key,
    "Authorization": f"Bearer {service_key}"
}
try:
    response = requests.get(f"{url}/rest/v1/", headers=headers)
    print(f"   Статус: {response.status_code}")
    print(f"   Ответ: {response.text[:500]}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
