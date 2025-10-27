# ⚡ Быстрые команды - PigWeight

Шпаргалка по основным командам системы.

---

## 🧪 Тестирование

```bash
# Тест интеграции всех компонентов
python test_integration.py

# Проверка синтаксиса
python -m py_compile console_app.py
python -m py_compile pig_tracking/weight_estimator.py
```

---

## 🎬 Обработка видео

```bash
# Интерактивный выбор видео
python console_app.py

# Конкретный файл
python console_app.py --video uploads/test.mp4

# С отладкой
python console_app.py --video uploads/test.mp4 --debug
```

---

## 🐳 Docker / Supabase

```bash
# Запустить
docker-compose up -d

# Проверить статус
docker-compose ps

# Остановить
docker-compose down

# Логи
docker-compose logs -f supabase-db

# Открыть Supabase Studio
start http://localhost:8000
```

---

## 🔍 Проверка результатов

```bash
# Список результатов
dir results\*.json

# Последний результат
dir results\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Просмотр JSON
python -m json.tool results\latest.json

# События
dir records\events\*.jsonl

# Последние события
Get-Content records\events\latest.jsonl -Tail 10
```

---

## 🌐 API сервер

```bash
# Запуск
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload

# Открыть веб-интерфейс
start http://localhost:8080

# Открыть API документацию
start http://localhost:8080/docs

# Мониторинг
start http://localhost:8080/monitor
```

---

## 📊 Статистика

```bash
# Количество обработанных видео
(Get-ChildItem results\*.json).Count

# Количество событий
(Get-ChildItem records\events\*.jsonl).Count

# Размер результатов
(Get-ChildItem results -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB

# Последняя обработка
Get-ChildItem results\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object Name, LastWriteTime
```

---

## 🔧 Настройка

```bash
# Редактировать конфигурацию
notepad .env

# Проверить переменные окружения
python -c "from core.config import get_config; c = get_config(); print(f'Model: {c.MODEL_PATH}'); print(f'Device: {c.DEVICE}'); print(f'Conf: {c.CONF_THRESHOLD}')"

# Проверить CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 🧹 Очистка

```bash
# Очистить результаты
Remove-Item results\*.json

# Очистить события
Remove-Item records\events\*.jsonl

# Очистить логи
Remove-Item logs\*.log

# Очистить кеш Python
Remove-Item -Recurse -Force __pycache__, *\__pycache__
```

---

## 📚 Документация

```bash
# Открыть основные файлы
notepad README.md
notepad QUICKSTART.md
notepad STATUS_BACKGROUND_PROCESSING.md
notepad FIXES_REPORT.md
notepad SUMMARY.md

# Открыть спецификации
notepad .kiro\specs\pig-tracking-system\tasks.md
notepad .kiro\specs\pig-tracking-system\requirements.md
```

---

## 🐛 Отладка

```bash
# Просмотр логов
Get-Content logs\app.log -Tail 50
Get-Content logs\perf.log -Tail 50

# Мониторинг в реальном времени
Get-Content logs\app.log -Wait

# Проверка процессов
Get-Process python

# Использование памяти
Get-Process python | Select-Object Name, @{Name="Memory(MB)";Expression={[math]::Round($_.WS / 1MB, 2)}}
```

---

## 🎯 Тестовый режим

```bash
# Сверка с Excel
python console_app.py --mode test --video uploads/test.mp4 --excel-reference docs/manual.xlsx

# С указанием папки результатов
python console_app.py --mode test --video uploads/test.mp4 --excel-reference docs/manual.xlsx --output test_results/

# Просмотр метрик
python -m json.tool test_results\metrics_*.json
```

---

## 📦 Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# Обновление зависимостей
pip install --upgrade -r requirements.txt

# Проверка установленных пакетов
pip list | findstr "torch ultralytics opencv supabase"

# Установка CUDA версии PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Быстрый старт (копипаста)

```bash
# 1. Тест без БД
python test_integration.py

# 2. Обработка видео
python console_app.py

# 3. Запуск Docker (опционально)
docker-compose up -d

# 4. Обработка с БД
python console_app.py --video uploads/test.mp4

# 5. Проверка результатов
dir results\*.json
dir records\events\*.jsonl
```

---

## 💡 Полезные алиасы (PowerShell)

Добавьте в `$PROFILE`:

```powershell
# PigWeight алиасы
function pig-test { python test_integration.py }
function pig-run { python console_app.py }
function pig-docker { docker-compose up -d }
function pig-logs { Get-Content logs\app.log -Tail 50 }
function pig-results { dir results\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 5 }
function pig-events { dir records\events\*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 5 }
```

Использование:
```bash
pig-test      # Запустить тесты
pig-run       # Запустить обработку
pig-docker    # Запустить Docker
pig-logs      # Показать логи
pig-results   # Показать результаты
pig-events    # Показать события
```

---

**Версия:** 1.1.0  
**Дата:** 27.10.2025
