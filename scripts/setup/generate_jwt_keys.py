#!/usr/bin/env python3
"""
Генерация JWT ключей для Supabase
"""

import jwt
import time

# JWT секрет из docker-compose.yml
JWT_SECRET = "your-super-secret-jwt-token-with-at-least-32-characters-long"

print("🔑 Генерация JWT ключей для Supabase\n")
print(f"JWT Secret: {JWT_SECRET}\n")

# Генерируем anon ключ
anon_payload = {
    "iss": "supabase-demo",
    "role": "anon",
    "exp": 1983812996  # Далекое будущее
}

anon_key = jwt.encode(anon_payload, JWT_SECRET, algorithm="HS256")
print(f"SUPABASE_KEY (anon):")
print(f"{anon_key}\n")

# Генерируем service_role ключ
service_payload = {
    "iss": "supabase-demo",
    "role": "service_role",
    "exp": 1983812996
}

service_key = jwt.encode(service_payload, JWT_SECRET, algorithm="HS256")
print(f"SUPABASE_SERVICE_KEY:")
print(f"{service_key}\n")

# Сохраняем в файл
with open('.env.supabase', 'w') as f:
    f.write(f"# Supabase JWT ключи (сгенерированы {time.strftime('%Y-%m-%d %H:%M:%S')})\n")
    f.write(f"SUPABASE_URL=http://localhost:8000\n")
    f.write(f"SUPABASE_KEY={anon_key}\n")
    f.write(f"SUPABASE_SERVICE_KEY={service_key}\n")

print(f"✅ Ключи сохранены в .env.supabase")
print(f"\n💡 Скопируйте эти ключи в ваш .env файл")
