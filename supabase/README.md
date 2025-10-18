# Локальный Supabase для системы отслеживания свиней

## Быстрый старт

1. **Скопировать настройки:**
   ```bash
   cp .env.example .env
   ```

2. **Запустить Supabase:**
   ```bash
   docker-compose up -d
   ```

3. **Проверить что все работает:**
   - База данных: http://localhost:5432
   - API: http://localhost:8000
   - Supabase Studio: http://localhost:8000 (если настроен)

## Доступы

- **Database URL:** `postgresql://postgres:your-super-secret-and-long-postgres-password@localhost:5432/postgres`
- **API URL:** `http://localhost:8000`
- **Anon Key:** `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0`
- **Service Key:** `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU`

## Таблицы

После запуска автоматически создаются таблицы:
- `weighing_acts` - акты взвешивания
- `crossings` - отдельные проходы свиней
- `excel_schemas` - схемы Excel шаблонов

## Команды

```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Перезапуск с пересозданием
docker-compose down -v
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Подключение к базе
psql postgresql://postgres:your-super-secret-and-long-postgres-password@localhost:5432/postgres
```

## Тестирование подключения

```python
from supabase import create_client, Client

url = "http://localhost:8000"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"

supabase: Client = create_client(url, key)

# Тест подключения
result = supabase.table('weighing_acts').select("*").execute()
print("Подключение успешно!", result)
```