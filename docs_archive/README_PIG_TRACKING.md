# Система автоматического отслеживания свиней

Автоматический подсчет свиней при прохождении через весы с сохранением в базу данных.

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Установка зависимостей для системы отслеживания
pip install -r requirements-pig-tracking.txt
```

### 2. Настройка окружения

```bash
# Копируем файл с настройками
cp .env.example .env

# Настройки уже заполнены для локального Supabase
```

### 3. Запуск Supabase

```bash
# Запуск локального Supabase через Docker
docker-compose up -d

# Проверка что все запустилось
docker-compose ps
```

### 4. Запуск приложения

**Вариант 1: Автоматический запуск (рекомендуется)**
```bash
python start_pig_tracking.py
```

**Вариант 2: Ручной запуск**
```bash
# Интерактивный выбор видео
python console_app.py

# Обработка конкретного файла
python console_app.py --video uploads/test_video.mp4
```

## 📁 Структура проекта

```
.
├── console_app.py              # Основное консольное приложение
├── start_pig_tracking.py       # Скрипт быстрого запуска
├── pig_tracking/               # Модули системы отслеживания
│   ├── database.py            # Работа с Supabase
│   ├── crossing_counter.py    # Подсчет проходов
│   ├── act_detector.py        # Обнаружение актов взвешивания
│   └── video_processor.py     # Обработка видео
├── supabase/                   # Настройки Supabase
│   ├── migrations/            # SQL миграции
│   └── config/                # Конфигурация
├── docker-compose.yml          # Docker Compose для Supabase
└── uploads/                    # Папка для видеофайлов
```

## 🎯 Использование

### Обработка видео

1. Поместите видеофайлы в папку `uploads/`
2. Запустите `python console_app.py`
3. Выберите видео из списка
4. Дождитесь окончания обработки
5. Результаты сохранятся в базу данных Supabase

### Просмотр результатов

**Через Supabase Studio:**
- Откройте http://localhost:8000
- Перейдите в Table Editor
- Просмотрите таблицы `weighing_acts` и `crossings`

**Через Python:**
```python
from pig_tracking.database import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()

# Получить акты за последний день
acts = db.get_acts_by_period(
    start=datetime.now() - timedelta(days=1),
    end=datetime.now()
)

for act in acts:
    print(f"Акт: {act.started_at} - {act.ended_at}")
    print(f"  Слева: {act.left_count}, Справа: {act.right_count}")
    print(f"  Пик: {act.peak_count}")
```

## 🔧 Настройки

Основные параметры в `.env`:

```env
# База данных
SUPABASE_URL=http://localhost:8000
SUPABASE_KEY=...

# Модель
MODEL_PATH=models/pig_yolo11-seg.v4.pt
CONF_THRESHOLD=0.30
IMG_SIZE=960

# Линии детекции
LINE_LEFT_X=0.25
LINE_RIGHT_X=0.75

# Акты взвешивания
MIN_PIGS_FOR_ACT=3          # Минимум свиней для начала акта
MAX_INTERVAL_SEC=30.0       # Макс интервал без активности
MIN_ACT_DURATION_SEC=10.0   # Минимальная длительность акта
```

## 🧪 Тестирование

### Тест подключения к базе данных

```bash
python test_database.py
```

### Тест обработки видео

```bash
# Поместите тестовое видео в uploads/test.mp4
python console_app.py --video uploads/test.mp4 --debug
```

## 📊 Структура базы данных

### Таблица `weighing_acts`
- `id` - ID акта
- `started_at` - Время начала
- `ended_at` - Время окончания
- `duration_sec` - Длительность в секундах
- `left_count` - Количество проходов слева
- `right_count` - Количество проходов справа
- `peak_count` - Пиковое количество одновременно
- `total_weight` - Общий вес
- `avg_weight` - Средний вес
- `video_file` - Имя видеофайла

### Таблица `crossings`
- `id` - ID прохода
- `act_id` - Ссылка на акт взвешивания
- `pig_id` - ID свиньи (из трекера)
- `direction` - Направление ('left' или 'right')
- `crossed_at` - Время пересечения
- `line_x`, `line_y` - Координаты пересечения
- `weight_estimate` - Оценка веса

## 🐛 Решение проблем

### Supabase не запускается

```bash
# Остановить и удалить контейнеры
docker-compose down -v

# Запустить заново
docker-compose up -d

# Проверить логи
docker-compose logs -f
```

### Ошибка подключения к базе

```bash
# Проверить что Supabase запущен
docker-compose ps

# Проверить переменные окружения
cat .env | grep SUPABASE
```

### Видео не обрабатывается

```bash
# Проверить что модель существует
ls -lh models/pig_yolo11-seg.v4.pt

# Запустить с отладкой
python console_app.py --video uploads/test.mp4 --debug
```

## 📚 Дополнительная информация

- [Документация Supabase](https://supabase.com/docs)
- [Документация YOLO](https://docs.ultralytics.com/)
- Требования: `.kiro/specs/pig-tracking-system/requirements-simple.md`
- Дизайн: `.kiro/specs/pig-tracking-system/design.md`

## 🎯 Следующие этапы

- [ ] Этап 1: MVP с базой данных (в процессе)
- [ ] Этап 2: Excel экспорт и сверка
- [ ] Этап 3: Веб-интерфейс