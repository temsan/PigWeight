#!/usr/bin/env python3
"""
Исправление подключения к Supabase
Проблема: используется anon ключ вместо service_role
"""

import os
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

print("🔧 Исправление подключения к Supabase\n")

# Текущие ключи
anon_key = os.getenv('SUPABASE_KEY')
service_key = os.getenv('SUPABASE_SERVICE_KEY')

print(f"📋 Текущие ключи:")
print(f"   SUPABASE_KEY (anon): {anon_key[:50]}...")
print(f"   SUPABASE_SERVICE_KEY: {service_key[:50]}...")

# Проверяем подключение с service_role ключом
print(f"\n🧪 Тестирование подключения с service_role ключом...")

try:
    from supabase import create_client
    
    url = os.getenv('SUPABASE_URL', 'http://localhost:8000')
    
    # Пробуем с service_role ключом
    client = create_client(url, service_key)
    
    # Тестовый запрос
    result = client.table('weighing_acts').select("*").limit(1).execute()
    
    print(f"✅ Подключение успешно с service_role ключом!")
    print(f"   Записей в weighing_acts: {len(result.data)}")
    
    print(f"\n💡 Решение:")
    print(f"   Используйте SUPABASE_SERVICE_KEY вместо SUPABASE_KEY")
    print(f"   Или обновите код для использования service_role ключа")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print(f"\n🔍 Проверьте:")
    print(f"   1. Docker контейнеры запущены: docker ps")
    print(f"   2. Supabase доступен: curl http://localhost:8000")
    print(f"   3. JWT секрет в docker-compose.yml совпадает с ключами")
