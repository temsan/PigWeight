# Модули отслеживания свиней

Набор модулей для обработки видео, отслеживания свиней и определения актов взвешивания.

## Структура модулей

### 1. `crossing_counter.py` - Подсчет пересечений линий

Адаптирован из `VideoStream._update_line_counters()` в `api/app.py`.

**Основные возможности:**
- Отслеживание пересечений двух вертикальных линий (левая и правая)
- Интерполяция Y-координат в точке пересечения
- Cooldown между событиями для предотвращения дублирования
- Направленный подсчет (вход/выход с каждой стороны)

**Параметры:**
- `line_left_x`: позиция левой линии (0.0-1.0, по умолчанию 0.25)
- `line_right_x`: позиция правой линии (0.0-1.0, по умолчанию 0.75)
- `cooldown_sec`: минимальный интервал между событиями (по умолчанию 1.0s)

**Пример использования:**
```python
from pig_tracking import CrossingCounter

counter = CrossingCounter(line_left_x=0.25, line_right_x=0.75)

# Обработка треков
events = counter.process_tracks(
    track_ids=[1, 2, 3],
    centers_x=[0.3, 0.5, 0.8],
    centers_y=[0.5, 0.6, 0.4]
)

# Получение статистики
stats = counter.get_stats()
print(f"Вход слева: {stats['left_in']}")
print(f"Вход справа: {stats['right_in']}")
```

### 2. `act_detector.py` - Определение актов взвешивания

Определяет начало и конец актов взвешивания на основе активности пересечений.

**Основные возможности:**
- Автоматическое определение начала акта (при достижении порога активности)
- Автоматическое завершение акта (при отсутствии активности)
- Игнорирование одиночных проходов
- Сбор статистики по каждому акту

**Параметры:**
- `min_pigs_for_act`: минимальное количество проходов для начала акта (по умолчанию 3)
- `max_interval_sec`: максимальный интервал без активности для завершения (по умолчанию 30.0s)

**Пример использования:**
```python
from pig_tracking import ActDetector

detector = ActDetector(min_pigs_for_act=3, max_interval_sec=30.0)

# Обновление на основе пересечений
completed_act = detector.update(
    crossings=crossing_events,
    current_count=5,
    timestamp=time.time()
)

if completed_act:
    print(f"Завершен акт #{completed_act.act_id}")
    print(f"Длительность: {completed_act.duration:.1f}s")
    print(f"Пиковое количество: {completed_act.peak_count}")
```

### 3. `video_processor.py` - Интегрированный процессор

Объединяет все компоненты в единый пайплайн обработки.

**Компоненты:**
- `UnifiedVideoProcessor` (из `core/processor.py`) - детекция и сегментация
- `SimpleTracker` (из `api/app.py`) - отслеживание объектов
- `CrossingCounter` - подсчет пересечений
- `ActDetector` - определение актов

**Пример использования:**
```python
import asyncio
from pig_tracking import IntegratedVideoProcessor

async def main():
    # Создание процессора
    processor = IntegratedVideoProcessor(
        stream_id="my_video",
        conf_threshold=0.30,
        img_size=960
    )
    
    # Инициализация
    await processor.initialize()
    
    # Обработка видео
    summary = await processor.process_video_file(
        "path/to/video.mp4",
        progress_callback=lambda cur, tot: print(f"{cur}/{tot}")
    )
    
    print(f"Обработано кадров: {summary['frames_processed']}")
    print(f"Завершено актов: {summary['act_stats']['completed_acts_count']}")

asyncio.run(main())
```

## Быстрый старт

### Установка зависимостей

Все зависимости уже установлены в основном проекте:
- `opencv-python` - обработка видео
- `numpy` - работа с массивами
- Существующие компоненты: `UnifiedVideoProcessor`, `SimpleTracker`, `ModelAdapter`

### Запуск примера

```bash
# С указанием видеофайла
python pig_tracking/example_usage.py path/to/video.mp4

# Автоматический выбор из папки uploads/
python pig_tracking/example_usage.py
```

## Интеграция с существующими компонентами

### UnifiedVideoProcessor (core/processor.py)

Используется для детекции и сегментации:
```python
from core.processor import get_processor, ProcessingOptions

options = ProcessingOptions(conf_threshold=0.30, img_size=960)
processor = await get_processor("stream_id", options)
result = await processor.process_frame_async(frame, timestamp)
```

### SimpleTracker (api/app.py)

Используется для отслеживания объектов:
```python
from api.app import SimpleTracker

tracker = SimpleTracker(iou_threshold=0.3, max_age=30, dist_weight=0.2)
tracked_objects = tracker.update(detections)
```

### Параметры из CONFIG (core/config.py)

Все параметры берутся из глобальной конфигурации:
- `CONFIG.CONF_THRESHOLD` - порог уверенности детекции (0.30)
- `CONFIG.IMG_SIZE` - размер изображения для модели (960)
- `CONFIG.LINE_LEFT_X` - позиция левой линии (0.25)
- `CONFIG.LINE_RIGHT_X` - позиция правой линии (0.75)
- `CONFIG.CROSS_COOLDOWN_SEC` - cooldown между пересечениями (1.0)

## Архитектура обработки

```
Видео кадр
    ↓
UnifiedVideoProcessor (детекция + сегментация)
    ↓
SimpleTracker (отслеживание с ID)
    ↓
CrossingCounter (подсчет пересечений линий)
    ↓
ActDetector (определение актов взвешивания)
    ↓
Результат (статистика + события)
```

## Формат результатов

### Результат обработки кадра

```python
{
    'timestamp': 1234567890.123,
    'frame_number': 100,
    'detections': 5,
    'tracked_objects': [...],
    'current_count': 5,
    'crossing_events': [
        {
            'track_id': 1,
            'side': 'left',
            'mode': 'enter',
            'x': 0.25,
            'y': 0.5,
            'timestamp': 1234567890.123
        }
    ],
    'crossing_stats': {
        'left_in': 10,
        'right_in': 8,
        'total_crossings': 18
    },
    'act_stats': {
        'completed_acts_count': 2,
        'current_act': {...}
    },
    'completed_act': {...}  # если акт завершен на этом кадре
}
```

### Акт взвешивания

```python
{
    'act_id': 1,
    'started_at': 1234567890.0,
    'started_at_iso': '2024-01-15T10:30:00',
    'ended_at': 1234567920.0,
    'ended_at_iso': '2024-01-15T10:30:30',
    'duration': 30.0,
    'left_count': 15,
    'right_count': 14,
    'peak_count': 8,
    'seen_total': 20,
    'is_active': False,
    'crossings_count': 29
}
```

## Следующие шаги

Эти модули готовы для интеграции в консольное приложение (задачи 7-8 из tasks.md):

1. **Консольное приложение** (`console_app.py`)
   - Использует `IntegratedVideoProcessor` для обработки видео
   - Сохраняет результаты в Supabase через `DatabaseManager`

2. **База данных** (задача 2)
   - Таблица `weighing_acts` для актов
   - Таблица `crossings` для пересечений

3. **Excel экспорт** (задачи 9-11)
   - Экспорт актов из базы в Excel
   - Сверка с ручными записями

## Тестирование

Для тестирования модулей используйте `example_usage.py`:

```bash
# Тест на реальном видео
python pig_tracking/example_usage.py uploads/test_video.mp4
```

Ожидаемый вывод:
- Прогресс обработки кадров
- Статистика пересечений линий
- Список завершенных актов взвешивания
- Метрики производительности
