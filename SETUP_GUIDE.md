# 🚀 Полное руководство по установке и запуску

## Шаг 1: Подготовка окружения

### 1.1 Установка зависимостей

```bash
# Установка Python пакетов
pip install -r requirements-pig-tracking.txt
```

### 1.2 Настройка переменных окружения

```bash
# Копирование файла настроек
cp .env.example .env

# Файл .env уже содержит правильные настройки для локального Supabase
# Редактировать не нужно, если используете локальный Supabase
```

## Шаг 2: Запуск Supabase

### 2.1 Запуск через Docker Compose

```bash
# Запуск всех сервисов Supabase
docker-compose up -d

# Проверка что все запустилось
docker-compose ps

# Должны быть запущены: db, kong, auth, rest, realtime, storage
```

### 2.2 Ожидание готовности

```bash
# Подождите 10-15 секунд пока все сервисы запустятся
# Проверьте логи если что-то не работает:
docker-compose logs -f
```

## Шаг 3: Проверка системы

### 3.1 Автоматическая проверка

```bash
# Запуск скрипта проверки
python check_system.py

# Должно быть: ✅ 9/9 проверок пройдено
```

### 3.2 Ручная проверка компонентов

```bash
# Тест подключения к базе данных
python test_database.py

# Тест интеграции всех модулей
python test_integration.py
```

## Шаг 4: Подготовка видео

### 4.1 Размещение видеофайлов

```bash
# Создайте папку uploads если её нет
mkdir uploads

# Поместите видеофайлы в папку uploads/
# Поддерживаемые форматы: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm
```

### 4.2 Проверка модели YOLO

```bash
# Убедитесь что модель существует
ls -lh models/pig_yolo11-seg.v4.pt

# Если модели нет, укажите правильный путь в .env:
# MODEL_PATH=models/your_model.pt
```

## Шаг 5: Запуск приложения

### Вариант A: Автоматический запуск (рекомендуется)

```bash
# Скрипт автоматически проверит и запустит все необходимое
python start_pig_tracking.py
```

### Вариант B: Ручной запуск

```bash
# Интерактивный выбор видео из списка
python console_app.py

# Обработка конкретного файла
python console_app.py --video uploads/my_video.mp4

# С отладочным выводом
python console_app.py --video uploads/my_video.mp4 --debug
```

## Шаг 6: Просмотр результатов

### 6.1 Через Supabase Studio

```bash
# Откройте в браузере
http://localhost:8000

# Перейдите в Table Editor
# Просмотрите таблицы:
# - weighing_acts (акты взвешивания)
# - crossings (отдельные проходы)
```

### 6.2 Через Python

```python
from pig_tracking.database import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()

# Получить все акты за последний день
acts = db.get_acts_by_period(
    start=datetime.now() - timedelta(days=1),
    end=datetime.now()
)

for act in acts:
    print(f"Акт: {act.started_at} - {act.ended_at}")
    print(f"  Слева: {act.left_count}, Справа: {act.right_count}")
    print(f"  Пик: {act.peak_count}")
```

### 6.3 Через SQL

```bash
# Подключение к базе данных
psql postgresql://postgres:your-super-secret-and-long-postgres-password@localhost:5432/postgres

# Запросы
SELECT * FROM weighing_acts ORDER BY started_at DESC LIMIT 10;
SELECT * FROM crossings WHERE act_id = 1;
```

## Решение проблем

### Проблема: Supabase не запускается

```bash
# Остановить все контейнеры
docker-compose down

# Удалить volumes (ВНИМАНИЕ: удалит все данные)
docker-compose down -v

# Запустить заново
docker-compose up -d

# Проверить логи
docker-compose logs -f db
```

### Проблема: Ошибка подключения к базе

```bash
# Проверить что Supabase запущен
docker-compose ps

# Проверить переменные окружения
cat .env | grep SUPABASE

# Проверить что порт 8000 не занят
netstat -an | grep 8000  # Linux/Mac
netstat -an | findstr 8000  # Windows
```

### Проблема: Модель не найдена

```bash
# Проверить путь к модели
ls -lh models/

# Обновить путь в .env
nano .env
# MODEL_PATH=models/your_actual_model.pt
```

### Проблема: Видео не обрабатывается

```bash
# Проверить формат видео
file uploads/your_video.mp4

# Попробовать с отладкой
python console_app.py --video uploads/your_video.mp4 --debug

# Проверить логи
tail -f logs/console.log
```

### Проблема: Низкая производительность

```bash
# Проверить использование GPU
python -c "import torch; print(torch.cuda.is_available())"

# Если GPU недоступен, система будет использовать CPU
# Это нормально, но обработка будет медленнее

# Можно уменьшить размер изображения в .env:
# IMG_SIZE=640  # вместо 960
```

## Настройка параметров

### Основные параметры в .env

```env
# Порог уверенности детекции (0.0-1.0)
# Меньше = больше детекций, но больше ложных срабатываний
CONF_THRESHOLD=0.30

# Размер изображения для модели
# Больше = точнее, но медленнее
IMG_SIZE=960

# Позиции линий детекции (0.0-1.0 от ширины кадра)
LINE_LEFT_X=0.25
LINE_RIGHT_X=0.75

# Минимум свиней для начала акта взвешивания
MIN_PIGS_FOR_ACT=3

# Максимальный интервал без активности для завершения акта (секунды)
MAX_INTERVAL_SEC=30.0

# Cooldown между проходами одной свиньи (секунды)
CROSS_COOLDOWN_SEC=1.0
```

## Следующие шаги

После успешного запуска MVP:

1. **Этап 2: Excel экспорт и сверка**
   - Анализ формата Excel
   - Экспорт результатов
   - Сверка с ручными записями

2. **Этап 3: Веб-интерфейс**
   - REST API
   - Мобильная страница
   - Кнопки экспорта/сверки

## Полезные команды

```bash
# Остановить Supabase
docker-compose down

# Перезапустить Supabase
docker-compose restart

# Просмотр логов
docker-compose logs -f

# Очистка всех данных (ВНИМАНИЕ!)
docker-compose down -v

# Проверка системы
python check_system.py

# Тест базы данных
python test_database.py

# Тест интеграции
python test_integration.py

# Запуск приложения
python console_app.py
```

## Поддержка

- Документация: `README_PIG_TRACKING.md`
- Быстрый старт: `QUICKSTART.md`
- Требования: `.kiro/specs/pig-tracking-system/requirements-simple.md`
- Дизайн: `.kiro/specs/pig-tracking-system/design.md`