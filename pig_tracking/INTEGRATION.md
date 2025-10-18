# Интеграция модулей отслеживания

## Что реализовано

✅ **Задача 4** - Модуль CrossingCounter
- Адаптирована логика из `VideoStream._update_line_counters()`
- Интерполяция Y-координат при пересечении
- Cooldown для предотвращения дублирования
- Направленный подсчет (вход/выход)

✅ **Задача 5** - Модуль ActDetector
- Определение начала акта (порог MIN_PIGS_FOR_ACT=3)
- Определение конца акта (MAX_INTERVAL_SEC=30.0)
- Игнорирование одиночных проходов
- Сбор статистики по актам

✅ **Задача 6** - Интеграция компонентов
- `IntegratedVideoProcessor` объединяет все компоненты
- Использует `UnifiedVideoProcessor` из `core/processor.py`
- Использует `SimpleTracker` из `api/app.py`
- Использует параметры из `core/config.py`

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│           IntegratedVideoProcessor                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐                                  │
│  │ UnifiedVideo     │  Детекция + Сегментация          │
│  │ Processor        │  (core/processor.py)             │
│  └────────┬─────────┘                                  │
│           │                                             │
│           ↓                                             │
│  ┌──────────────────┐                                  │
│  │ SimpleTracker    │  Отслеживание с ID               │
│  │                  │  (api/app.py)                    │
│  └────────┬─────────┘                                  │
│           │                                             │
│           ↓                                             │
│  ┌──────────────────┐                                  │
│  │ CrossingCounter  │  Подсчет пересечений             │
│  │                  │  (pig_tracking/)                 │
│  └────────┬─────────┘                                  │
│           │                                             │
│           ↓                                             │
│  ┌──────────────────┐                                  │
│  │ ActDetector      │  Определение актов               │
│  │                  │  (pig_tracking/)                 │
│  └──────────────────┘                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Использование в консольном приложении

### Базовый пример

```python
import asyncio
from pig_tracking import IntegratedVideoProcessor
from core.config import CONFIG

async def process_video(video_path: str):
    # Создание процессора
    processor = IntegratedVideoProcessor(
        stream_id="console_app",
        conf_threshold=CONFIG.CONF_THRESHOLD,
        img_size=CONFIG.IMG_SIZE,
        line_left_x=CONFIG.LINE_LEFT_X,
        line_right_x=CONFIG.LINE_RIGHT_X,
        min_pigs_for_act=3,
        max_interval_sec=30.0
    )
    
    # Инициализация
    await processor.initialize()
    
    # Обработка видео
    summary = await processor.process_video_file(video_path)
    
    return summary

# Запуск
summary = asyncio.run(process_video("video.mp4"))
```

### Интеграция с базой данных (следующий шаг)

```python
from pig_tracking import IntegratedVideoProcessor
from database_manager import DatabaseManager  # TODO: создать

async def process_and_save(video_path: str):
    # Процессор
    processor = IntegratedVideoProcessor(...)
    await processor.initialize()
    
    # База данных
    db = DatabaseManager()
    
    # Обработка с сохранением
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обработка кадра
        result = await processor.process_frame(frame)
        
        # Сохранение пересечений
        for event in result['crossing_events']:
            db.save_crossing(event)
        
        # Сохранение завершенных актов
        if result['completed_act']:
            db.save_weighing_act(result['completed_act'])
    
    cap.release()
```

## Параметры конфигурации

Все параметры берутся из `.env` через `core/config.py`:

```env
# Детекция
CONF_THRESHOLD=0.30
IMG_SIZE=960

# Линии подсчета
LINE_LEFT_X=0.25
LINE_RIGHT_X=0.75
CROSS_COOLDOWN_SEC=1.0

# Акты взвешивания
MIN_PIGS_FOR_ACT=3
MAX_INTERVAL_SEC=30.0
```

## Формат данных для базы

### Таблица `crossings`

```sql
CREATE TABLE crossings (
    id SERIAL PRIMARY KEY,
    act_id INTEGER REFERENCES weighing_acts(id),
    track_id INTEGER NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'left' или 'right'
    mode VARCHAR(10) NOT NULL,  -- 'enter' или 'exit'
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Данные из `CrossingEvent`:
```python
{
    'track_id': 1,
    'side': 'left',
    'mode': 'enter',
    'x': 0.25,
    'y': 0.5,
    'timestamp': 1234567890.123
}
```

### Таблица `weighing_acts`

```sql
CREATE TABLE weighing_acts (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    left_count INTEGER DEFAULT 0,
    right_count INTEGER DEFAULT 0,
    peak_count INTEGER DEFAULT 0,
    seen_total INTEGER DEFAULT 0,
    duration FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Данные из `WeighingAct`:
```python
{
    'act_id': 1,
    'started_at': 1234567890.0,
    'ended_at': 1234567920.0,
    'duration': 30.0,
    'left_count': 15,
    'right_count': 14,
    'peak_count': 8,
    'seen_total': 20
}
```

## Следующие шаги (из tasks.md)

### ✅ Выполнено
- [x] Задача 4: CrossingCounter
- [x] Задача 5: ActDetector
- [x] Задача 6: IntegratedVideoProcessor

### 🔄 Следующие задачи

**Задача 1-2: База данных**
```bash
# Создать файл: database_manager.py
# Реализовать:
# - DatabaseManager класс
# - save_crossing()
# - save_weighing_act()
# - get_acts_by_period()
```

**Задача 3: Консольное приложение**
```bash
# Создать файл: console_app.py
# Реализовать:
# - Парсинг аргументов (--video)
# - Интерактивный выбор видео
# - Интеграция с IntegratedVideoProcessor
# - Сохранение в базу через DatabaseManager
```

**Задача 7-8: Тестирование**
```bash
# Запуск:
python console_app.py --video uploads/test_video.mp4

# Проверка:
# - Обнаружение актов
# - Сохранение в Supabase
# - Точность подсчета
```

## Тестирование модулей

### Запуск примера

```bash
# С указанием видео
python pig_tracking/example_usage.py uploads/test_video.mp4

# Автоматический выбор
python pig_tracking/example_usage.py
```

### Ожидаемый вывод

```
Используем видео: uploads/test_video.mp4
Видео: 1500 кадров, 25.00 FPS, длительность 60.0s
Прогресс: 30/1500 (2.0%)
Прогресс: 60/1500 (4.0%)
...
🎬 Начат новый акт #1: 5 проходов за последнюю минуту
🔵 L=0.250 y=0.523 t1 ←IN (1)
🔵 L=0.250 y=0.487 t2 ←IN (2)
...
🏁 Завершен акт #1: длительность=25.3s, left=15, right=14, peak=8, seen=20

============================================================
ИТОГОВАЯ СТАТИСТИКА
============================================================
Видео: uploads/test_video.mp4
Обработано кадров: 1500/1500
Время обработки: 45.2s
Средний FPS: 33.2

ПЕРЕСЕЧЕНИЯ ЛИНИЙ:
  Вход слева: 15
  Вход справа: 14
  Всего пересечений: 29

АКТЫ ВЗВЕШИВАНИЯ:
  Завершенных актов: 1

Детали актов:
  Акт #1:
    Начало: 2024-01-15T10:30:00
    Окончание: 2024-01-15T10:30:25
    Длительность: 25.3s
    Вход слева: 15
    Вход справа: 14
    Пиковое количество: 8
    Всего уникальных: 20
```

## Отладка

### Включение подробного логирования

```python
from core.config import setup_logging

logger = setup_logging(debug=True)
```

### Проверка компонентов по отдельности

```python
# Тест CrossingCounter
from pig_tracking import CrossingCounter

counter = CrossingCounter()
events = counter.process_tracks([1, 2], [0.3, 0.8], [0.5, 0.6])
print(counter.get_stats())

# Тест ActDetector
from pig_tracking import ActDetector

detector = ActDetector()
completed = detector.update(events, current_count=2)
print(detector.get_stats())
```

## Производительность

Ожидаемая производительность на RTX 3060:
- **FPS обработки**: 30-40 кадров/сек
- **Время на кадр**: 25-35 мс
- **Память GPU**: ~2-3 GB

Оптимизация:
- Батчинг через `DynamicBatcher` (уже реализовано в `UnifiedVideoProcessor`)
- Half precision (FP16) для GPU (настраивается через `USE_HALF=true`)
- Адаптивный размер батча (настраивается через `MAX_BATCH_SIZE`)

## Поддержка

При возникновении проблем:
1. Проверьте логи в `logs/app.log`
2. Убедитесь, что модель загружена: `models/pig_yolo11-seg.v4.pt`
3. Проверьте параметры в `.env`
4. Запустите пример: `python pig_tracking/example_usage.py`
