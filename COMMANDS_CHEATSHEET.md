# ⚡ Шпаргалка команд - Система отслеживания свиней

## 🚀 Быстрый старт

### Запуск API сервера
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

### Запуск тестов API
```bash
python test_api_full.py
```

### Обработка видео
```bash
python console_app.py
```

---

## 🌐 Веб-интерфейсы

| Интерфейс | URL | Описание |
|-----------|-----|----------|
| Swagger UI | http://localhost:8080/docs | API документация |
| Monitor | http://localhost:8080/monitor.html | Мониторинг в реальном времени |
| Dashboard | http://localhost:8080/dashboard | Дашборд |
| Monitoring | http://localhost:8080/monitoring | Системный мониторинг |

---

## 🔍 Проверка API

### Health Check
```bash
curl http://localhost:8080/health
```

### Получить акты взвешивания
```bash
curl http://localhost:8080/api/journal/acts
```

### Получить статистику
```bash
curl http://localhost:8080/api/weighing/stats
```

### Получить записи
```bash
curl http://localhost:8080/api/records
```

### Получить пересечения линий
```bash
curl http://localhost:8080/api/line-crossings?limit=10
```

---

## 🗄️ База данных

### Проверка подключения
```bash
python -c "from pig_tracking.database import DatabaseManager; db = DatabaseManager(); print('✅ БД подключена')"
```

### Получить статистику из БД
```bash
python -c "from pig_tracking.database import DatabaseManager; db = DatabaseManager(); print(db.get_stats())"
```

### Проверка таблиц (через Supabase Studio)
```
http://localhost:8000
```

---

## 🐳 Docker

### Проверка статуса
```bash
docker ps
```

### Логи Supabase
```bash
docker-compose logs -f
```

### Перезапуск Supabase
```bash
docker-compose restart
```

### Остановка всех контейнеров
```bash
docker-compose down
```

### Запуск Supabase
```bash
docker-compose up -d
```

---

## 📊 Тестирование

### Полное тестирование API
```bash
python test_api_full.py
```

### Базовое тестирование
```bash
python scripts/tests/test_api_endpoints.py
```

### Проверка системы
```bash
python check_system.py
```

### Интеграционные тесты
```bash
python test_integration.py
```

---

## 📝 Логи

### Просмотр логов приложения
```bash
cat logs/app.log
```

### Просмотр логов консоли
```bash
cat logs/console.log
```

### Просмотр последних 50 строк
```bash
tail -n 50 logs/app.log
```

### Мониторинг логов в реальном времени
```bash
tail -f logs/app.log
```

---

## 🎥 Обработка видео

### Обработка конкретного видео
```bash
python console_app.py --video uploads/test_video.mp4
```

### Интерактивный выбор видео
```bash
python console_app.py
```

### Создание тестового видео
```bash
python create_test_video.py
```

---

## 🔧 Отладка

### Проверка переменных окружения
```bash
cat .env | grep -E "SUPABASE|MODEL|DEVICE"
```

### Проверка установленных пакетов
```bash
pip list | grep -E "ultralytics|supabase|fastapi"
```

### Проверка версии Python
```bash
python --version
```

### Проверка доступности порта
```bash
netstat -an | findstr :8080
```

---

## 📦 Установка зависимостей

### Установка всех зависимостей
```bash
pip install -r requirements.txt
```

### Установка только API зависимостей
```bash
pip install fastapi uvicorn requests
```

### Установка только ML зависимостей
```bash
pip install ultralytics opencv-python numpy
```

### Установка Supabase клиента
```bash
pip install supabase
```

---

## 🔄 Обновление

### Обновление зависимостей
```bash
pip install --upgrade -r requirements.txt
```

### Обновление конкретного пакета
```bash
pip install --upgrade ultralytics
```

---

## 🧹 Очистка

### Очистка кэша Python
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### Очистка логов
```bash
rm -f logs/*.log
```

### Очистка временных файлов
```bash
rm -rf temp/*
rm -rf results/*
```

---

## 📊 Статистика

### Размер репозитория
```bash
du -sh .
```

### Количество строк кода
```bash
find . -name "*.py" -not -path "./.venv/*" | xargs wc -l
```

### Список больших файлов
```bash
find . -type f -size +10M -not -path "./.venv/*"
```

---

## 🎯 Быстрые проверки

### Проверка всего за 30 секунд
```bash
# 1. Проверка Docker
docker ps

# 2. Проверка API
curl http://localhost:8080/health

# 3. Проверка БД
python -c "from pig_tracking.database import DatabaseManager; db = DatabaseManager(); print('✅ OK')"

# 4. Запуск тестов
python test_api_full.py
```

---

## 🆘 Troubleshooting

### API не отвечает
```bash
# Проверить процесс
ps aux | grep uvicorn

# Убить процесс
pkill -f uvicorn

# Перезапустить
python -m uvicorn api.app:app --port 8080 --reload
```

### БД недоступна
```bash
# Проверить Docker
docker ps

# Перезапустить Supabase
docker-compose restart

# Проверить логи
docker-compose logs supabase
```

### Модель не найдена
```bash
# Проверить наличие модели
ls -lh models/

# Проверить путь в .env
cat .env | grep MODEL_PATH

# Скачать модель (если нужно)
# wget URL -O models/pig_yolo11-seg.v4.pt
```

---

## 📱 Мобильные команды (через SSH)

### Проверка статуса
```bash
ssh user@server "cd /path/to/project && docker ps && curl localhost:8080/health"
```

### Просмотр логов
```bash
ssh user@server "tail -f /path/to/project/logs/app.log"
```

### Перезапуск API
```bash
ssh user@server "cd /path/to/project && pkill -f uvicorn && python -m uvicorn api.app:app --port 8080 &"
```

---

## 🎨 Полезные алиасы (добавить в .bashrc или .zshrc)

```bash
# Алиасы для проекта
alias pig-api='python -m uvicorn api.app:app --port 8080 --reload'
alias pig-test='python test_api_full.py'
alias pig-console='python console_app.py'
alias pig-logs='tail -f logs/app.log'
alias pig-docker='docker-compose up -d'
alias pig-health='curl http://localhost:8080/health'
```

---

## 📚 Документация

### Открыть документацию
```bash
# Swagger UI
open http://localhost:8080/docs

# README
cat README.md

# Спецификация
cat .kiro/specs/pig-tracking-system/requirements-simple.md
```

---

**Создано:** 2025-10-18  
**Обновлено:** В реальном времени  
**Версия:** 1.0
