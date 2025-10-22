# 🚀 Быстрая справка - PigWeight

## ⚡ Основные команды

### Проверка системы
```bash
python scripts/setup/check_system.py
```

### Запуск Supabase
```bash
docker-compose up -d
```

### Запуск приложения
```bash
# Консольное приложение
python console_app.py

# API сервер
python -m uvicorn api.app:app --port 8080 --reload
```

### Тестирование
```bash
# Тестирование MVP
python scripts/tests/test_mvp.py

# Тестирование API
python scripts/tests/test_api_endpoints.py

# Тестирование БД
python scripts/tests/test_database.py
```

---

## 📁 Где что находится

| Что искать | Где находится |
|------------|---------------|
| Документация | `docs/` |
| Руководства | `docs/guides/` |
| Отчеты | `docs/reports/` |
| Скрипты настройки | `scripts/setup/` |
| Тесты | `scripts/tests/` |
| Утилиты | `scripts/utils/` |
| API | `api/` |
| Основной код | `pig_tracking/` |
| Конфигурация | `core/config.py` |
| База данных | `supabase/` |
| Видео | `uploads/` |
| Логи | `logs/` |

---

## 🔧 Настройка

### Переменные окружения (.env)
```bash
# Supabase
SUPABASE_URL=http://localhost:8000
SUPABASE_SERVICE_KEY=your-key-here

# Модель
MODEL_PATH=yolo11n-seg.pt
CONF_THRESHOLD=0.30

# Устройство
DEVICE=cpu
```

### Docker
```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Логи
docker-compose logs -f

# Перезапуск
docker-compose restart
```

---

## 🐛 Решение проблем

### База данных не подключается
```bash
# 1. Проверить Docker
docker ps

# 2. Перезапустить Kong
docker restart pigweight-kong-1

# 3. Проверить ключи
python scripts/setup/fix_supabase_connection.py
```

### Модель не найдена
```bash
# Проверить путь в .env
MODEL_PATH=yolo11n-seg.pt

# Модель скачается автоматически при первом запуске
```

### API не запускается
```bash
# Проверить порт
netstat -ano | findstr :8080

# Использовать другой порт
python -m uvicorn api.app:app --port 8081
```

---

## 📊 Мониторинг

### Проверка статуса
```bash
# Система
python scripts/setup/check_system.py

# Docker
docker ps

# База данных
python -c "from pig_tracking.database import DatabaseManager; db = DatabaseManager(); print(db.get_stats())"
```

### Логи
```bash
# Приложение
tail -f logs/console.log

# Docker
docker-compose logs -f

# Конкретный сервис
docker logs pigweight-kong-1
```

---

## 🔗 Полезные ссылки

- **Swagger UI:** http://localhost:8080/docs
- **Supabase Studio:** http://localhost:8000
- **API Health:** http://localhost:8080/health

---

## 📚 Документация

- [README.md](README.md) - главная документация
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - структура проекта
- [docs/guides/QUICKSTART.md](docs/guides/QUICKSTART.md) - быстрый старт
- [docs/guides/TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md) - решение проблем
- [PARALLEL_TASKS.md](PARALLEL_TASKS.md) - задания для работы

---

**Обновлено:** 2025-10-18
