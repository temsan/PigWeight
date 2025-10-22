# Структура проекта - Система отслеживания свиней

## 📁 Основные директории

### `/api` - API сервер
FastAPI приложение для REST API
- `app.py` - главный файл приложения
- `routes/` - эндпоинты API
- `middleware/` - middleware (CORS, security)

### `/pig_tracking` - Основной модуль
Ядро системы отслеживания
- `database.py` - работа с Supabase
- `tracker.py` - трекинг объектов
- `line_crossing.py` - детекция пересечений
- `weighing_detector.py` - определение актов взвешивания
- `video_processor.py` - обработка видео

### `/core` - Конфигурация
- `config.py` - настройки системы

### `/supabase` - База данных
- `config/` - конфигурация Kong
- `migrations/` - миграции БД

### `/docs` - Документация
- `guides/` - руководства и гайды
- `reports/` - отчеты о тестировании
- `archive/` - архив старых документов

### `/scripts` - Скрипты
- `setup/` - скрипты настройки и проверки
- `tests/` - тестовые скрипты
- `utils/` - утилиты

### `/static` - Статические файлы
Веб-интерфейс, CSS, JS

### `/uploads` - Загруженные видео
Видеофайлы для обработки

### `/logs` - Логи
Логи работы системы

### `/models` - Модели YOLO
Обученные модели для детекции

---

## 🚀 Основные файлы

### Запуск приложений:
- `main.py` - главный файл (legacy)
- `console_app.py` - консольное приложение
- `start_pig_tracking.py` - запуск системы

### Конфигурация:
- `.env` - переменные окружения
- `.env.example` - пример конфигурации
- `docker-compose.yml` - Docker конфигурация
- `requirements.txt` - зависимости Python

### Документация:
- `README.md` - основная документация
- `README_PIG_TRACKING.md` - документация системы трекинга
- `CHANGELOG.md` - история изменений
- `TODO.md` - список задач

### Рабочие документы:
- `CONTEXT_FOR_PARALLEL_WORK.md` - контекст для параллельной работы
- `PARALLEL_TASKS.md` - задания для параллельной работы

---

## 🔧 Скрипты в `/scripts`

### Setup (настройка):
- `check_system.py` - проверка готовности системы
- `create_test_video.py` - создание тестового видео
- `generate_jwt_keys.py` - генерация JWT ключей
- `fix_supabase_connection.py` - исправление подключения к БД

### Tests (тесты):
- `test_mvp.py` - тестирование MVP
- `test_api_endpoints.py` - тестирование API
- `test_database.py` - тестирование БД
- `test_integration.py` - интеграционные тесты
- `test_inference.py` - тестирование инференса
- `validate_setup.py` - валидация настройки

### Utils (утилиты):
- `cleanup_venv.bat` - очистка виртуального окружения
- `start_server_venv.bat` - запуск сервера в venv
- `start_server.bat` - запуск сервера

---

## 📊 Документация в `/docs`

### Guides (руководства):
- `ARCHITECTURE_REDESIGN.md` - архитектура системы
- `DEPLOYMENT_GUIDE.md` - руководство по деплою
- `QUICKSTART.md` - быстрый старт
- `SETUP_GUIDE.md` - руководство по настройке
- `TROUBLESHOOTING.md` - решение проблем

### Reports (отчеты):
- `MVP_TEST_REPORT.md` - отчет о тестировании MVP
- `deep_perf_analysis.json` - анализ производительности
- `measurements_analysis.json` - анализ измерений
- `pipeline_analysis.json` - анализ pipeline

### Archive (архив):
- Старые документы и заметки

---

## 🎯 Быстрый старт

### 1. Проверка системы:
```bash
python scripts/setup/check_system.py
```

### 2. Запуск Supabase:
```bash
docker-compose up -d
```

### 3. Запуск консольного приложения:
```bash
python console_app.py
```

### 4. Запуск API:
```bash
python -m uvicorn api.app:app --port 8080 --reload
```

---

## 📝 Спецификация проекта

Находится в `.kiro/specs/pig-tracking-system/`:
- `requirements.md` - требования
- `design.md` - дизайн
- `tasks.md` - задачи

---

**Обновлено:** 2025-10-18  
**Версия:** 1.0
