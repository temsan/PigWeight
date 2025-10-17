# Документ проектирования

## Обзор

Система отслеживания и взвешивания свиней состоит из двух основных компонентов:

1. **Консольное приложение** - работает в фоне, обрабатывает видео, обнаруживает акты взвешивания, сохраняет в базу данных
2. **Веб-интерфейс** - отображает показатели в реальном времени без видео, предоставляет функции экспорта и сверки

Система оптимизирована для тестирования: запуск на видео за период времени с автоматической сверкой результатов с Excel файлом от операторов.

## Архитектура

### Высокоуровневая архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Консольное приложение                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Видео вход   │→ │ YOLO детекция│→ │ Трекинг      │      │
│  │ (камера/файл)│  │ + сегментация│  │ + подсчет    │      │
│  └──────────────┘  └──────────────┘  └──────┬───────┘      │
│                                              ↓               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Обнаружение  │← │ Анализ       │← │ Пересечение  │      │
│  │ актов        │  │ активности   │  │ линий        │      │
│  └──────┬───────┘  └──────────────┘  └──────────────┘      │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │         PostgreSQL / Supabase (локально)         │      │
│  │  - weighing_acts (акты взвешивания)              │      │
│  │  - crossings (отдельные проходы внутри актов)    │      │
│  └──────────────────────┬───────────────────────────┘      │
└─────────────────────────┼───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      Веб-интерфейс                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ REST API     │  │ Показатели   │  │ Экспорт      │      │
│  │ (FastAPI)    │→ │ в реал-тайм  │  │ в Excel      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Сверка с     │  │ Отчет о      │  │ Визуализация │      │
│  │ Excel        │→ │ расхождениях │→ │ результатов  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Режимы работы

1. **Режим реального времени** - обработка с камеры, непрерывная работа
2. **Режим обработки файла** - обработка видеофайла с прогрессом
3. **Тестовый режим** - обработка + автоматическая сверка с Excel

## Компоненты и интерфейсы

### 1. Консольное приложение

#### 1.1 Модуль захвата видео (VideoCapture)

**Назначение:** Получение кадров из различных источников

**Интерфейс:**
```python
class VideoCapture:
    def __init__(self, source: str, mode: str = "realtime"):
        """
        source: путь к файлу или URL камеры
        mode: "realtime", "file", "test"
        """
        pass
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Читает следующий кадр"""
        pass
    
    def get_progress(self) -> float:
        """Возвращает прогресс обработки (0.0-1.0)"""
        pass
    
    def get_timestamp(self) -> datetime:
        """Возвращает временную метку текущего кадра"""
        pass
```

**Зависимости:**
- OpenCV для чтения видео
- PyAV для оптимизированного декодирования (опционально)

#### 1.2 Модуль детекции (использует существующий UnifiedVideoProcessor)

**Назначение:** Обнаружение свиней на кадре с использованием YOLO сегментации

**Используем существующие компоненты:**
- `core/processor.py` → `UnifiedVideoProcessor` - основной процессор с батчингом
- `services/model_adapter.py` → `ModelAdapter` - адаптер модели (PyTorch/ONNX)
- `core/preprocess.py` → `center_crop_resize()` - предобработка кадров

**Извлеченные настройки из существующей системы:**

```python
# Из core/config.py
CONF_THRESHOLD = 0.30  # Порог уверенности детекции
IMG_SIZE = 960  # Размер входного изображения для модели
DEVICE = "auto"  # Автоопределение устройства (cuda/cpu)
USE_HALF = True  # Использование half precision (FP16) на GPU
BATCH_SIZE = 4  # Размер батча для инференса
MAX_WAIT_MS = 50  # Максимальное время ожидания батча

# Preprocessing (из core/preprocess.py)
# center_crop_resize() - масштабирование с padding до квадрата
# - Сохраняет пропорции
# - Добавляет черные поля (letterbox)
# - Возвращает transform_meta для обратного маппинга масок

# Model Adapter (из services/model_adapter.py)
# - Автоопределение оптимального устройства
# - Поддержка PyTorch (.pt) и ONNX (.onnx)
# - Автоматический выбор precision (FP16/FP32)
# - Type safety для предотвращения ошибок dtype

# Processor (из core/processor.py)
# - DynamicBatcher для адаптивного батчинга
# - Асинхронная обработка через asyncio
# - Автоматический маппинг масок обратно к оригинальным координатам
```

**Интерфейс (переиспользуем существующий):**
```python
from core.processor import get_processor, ProcessingOptions, FrameResult

# Получение процессора
processor = await get_processor(
    stream_id="console_app",
    options=ProcessingOptions(
        conf_threshold=0.30,
        img_size=960
    )
)

# Обработка кадра
result: FrameResult = await processor.process_frame_async(frame, timestamp)
# result содержит: detections, confidence, masks, bboxes, centroids
```

#### 1.3 Модуль трекинга (использует существующий SimpleTracker)

**Назначение:** Отслеживание свиней между кадрами с уникальными ID

**Используем существующий компонент:**
- `api/app.py` → `SimpleTracker` - трекер с Hungarian алгоритмом

**Извлеченные параметры трекинга:**

```python
# Из api/app.py SimpleTracker
class SimpleTracker:
    def __init__(self, 
                 iou_threshold=0.3,  # Порог IoU для сопоставления
                 max_age=30,  # Максимальный возраст трека без обновлений
                 dist_weight=0.2):  # Вес расстояния в функции стоимости
        pass
    
    # Алгоритм:
    # 1. Вычисление IoU между детекциями и существующими треками
    # 2. Вычисление расстояния между центроидами (нормализованное)
    # 3. Функция стоимости: (1 - IoU) + dist_weight * normalized_distance
    # 4. Hungarian алгоритм для оптимального сопоставления
    # 5. Создание новых треков для несопоставленных детекций
    # 6. Уменьшение age для несопоставленных треков (age -= 2)
    # 7. Удаление треков с age <= 0
```

**Интерфейс (переиспользуем существующий):**
```python
from api.app import SimpleTracker

tracker = SimpleTracker(
    iou_threshold=0.3,
    max_age=30,
    dist_weight=0.2
)

# Обновление треков
tracked_pigs = tracker.update(detections)
# detections: List[{bbox, ...}]
# tracked_pigs: List[{id, bbox, cx, cy, age, ...}]
```

#### 1.4 Модуль подсчета проходов (использует логику из VideoStream)

**Назначение:** Подсчет проходов через вертикальные линии

**Используем существующую логику:**
- `api/app.py` → `VideoStream._update_line_counters()` - подсчет проходов

**Извлеченные параметры и алгоритм:**

```python
# Из core/config.py
LINE_LEFT_X = 0.25  # Позиция левой линии (0.0-1.0)
LINE_RIGHT_X = 0.75  # Позиция правой линии (0.0-1.0)
CROSS_COOLDOWN_SEC = 1.0  # Cooldown между проходами одной свиньи

# Алгоритм из api/app.py VideoStream._update_line_counters():
# 1. Для каждого трека сохраняем предыдущую позицию (prev_x, prev_y)
# 2. Определяем, был ли трек внутри зоны (между линиями) ранее
# 3. Определяем, находится ли трек внутри зоны сейчас
# 4. Детектируем события:
#    - Вход слева: prev < LINE_LEFT_X <= current (left_in++)
#    - Вход справа: prev > LINE_RIGHT_X >= current (right_in++)
#    - Выход слева: prev >= LINE_LEFT_X > current (left_flow--)
#    - Выход справа: prev <= LINE_RIGHT_X < current (right_flow--)
# 5. Интерполяция Y-координаты в точке пересечения линии
# 6. Проверка cooldown: не считать проход, если прошло < CROSS_COOLDOWN_SEC
# 7. Логирование события с временной меткой и координатами
```

**Интерфейс (адаптируем из существующего):**
```python
class CrossingCounter:
    def __init__(self, 
                 left_line_x: float = 0.25, 
                 right_line_x: float = 0.75, 
                 cooldown_sec: float = 1.0):
        self.left_line_x = left_line_x
        self.right_line_x = right_line_x
        self.cooldown_sec = cooldown_sec
        self._track_prev_x = {}  # track_id -> prev_x
        self._track_prev_y = {}  # track_id -> prev_y
        self._track_is_inside = {}  # track_id -> bool
        self._track_last_cross_time = {}  # track_id -> timestamp
    
    def process_tracks(self, tracks, current_time) -> List[CrossingEvent]:
        # Реализация на основе VideoStream._update_line_counters()
        pass
```

#### 1.5 Модуль обнаружения актов (ActDetector)

**Назначение:** Автоматическое обнаружение начала и конца актов взвешивания

**Интерфейс:**
```python
class ActDetector:
    def __init__(self, 
                 min_pigs_threshold: int = 3,
                 max_interval_sec: float = 30.0,
                 min_duration_sec: float = 10.0):
        """
        min_pigs_threshold: минимум свиней для начала акта
        max_interval_sec: макс интервал без активности для завершения акта
        min_duration_sec: минимальная длительность акта
        """
        pass
    
    def update(self, crossings: List[CrossingEvent], current_time: datetime) -> Optional[ActEvent]:
        """
        Обновляет состояние, возвращает событие акта (start/end)
        ActEvent: {type: "start"|"end", timestamp, act_id}
        """
        pass
    
    def get_current_act(self) -> Optional[WeighingAct]:
        """Возвращает текущий активный акт или None"""
        pass
```

#### 1.6 Модуль базы данных (DatabaseManager для Supabase)

**Назначение:** Сохранение данных в локальный Supabase

**Используем:**
- `supabase-py` - официальный Python клиент для Supabase
- Локальный Supabase через Docker Compose

**Интерфейс:**
```python
from supabase import create_client, Client

class DatabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        supabase_url: URL локального Supabase (http://localhost:54321)
        supabase_key: Anon key из Supabase
        """
        self.client: Client = create_client(supabase_url, supabase_key)
    
    def save_crossing(self, crossing: CrossingEvent) -> int:
        """Сохраняет проход через supabase.table('crossings').insert()"""
        result = self.client.table('crossings').insert({
            'act_id': crossing.act_id,
            'pig_id': crossing.pig_id,
            'direction': crossing.direction,
            'crossed_at': crossing.timestamp.isoformat(),
            'line_x': crossing.line_x,
            'line_y': crossing.line_y,
            'weight_estimate': crossing.weight_estimate
        }).execute()
        return result.data[0]['id']
    
    def save_weighing_act(self, act: WeighingAct) -> int:
        """Сохраняет акт взвешивания"""
        result = self.client.table('weighing_acts').insert({
            'started_at': act.started_at.isoformat(),
            'ended_at': act.ended_at.isoformat(),
            'duration_sec': act.duration_sec,
            'left_count': act.left_count,
            'right_count': act.right_count,
            'peak_count': act.peak_count,
            'total_weight': act.total_weight,
            'avg_weight': act.avg_weight,
            'stream_id': act.stream_id,
            'video_file': act.video_file
        }).execute()
        return result.data[0]['id']
    
    def get_acts_by_period(self, start: datetime, end: datetime) -> List[WeighingAct]:
        """Получает акты за период через Supabase query"""
        result = self.client.table('weighing_acts')\
            .select('*')\
            .gte('started_at', start.isoformat())\
            .lte('started_at', end.isoformat())\
            .execute()
        return [WeighingAct(**row) for row in result.data]
```

### 2. Веб-интерфейс

#### 2.1 REST API (FastAPI)

**Эндпоинты:**

```python
# Получение текущих показателей
GET /api/stats/current
Response: {
    "current_count": int,
    "left_count": int,
    "right_count": int,
    "total_weight": float,
    "avg_weight": float,
    "active_act": Optional[WeighingAct]
}

# Получение истории актов
GET /api/acts?start_date=...&end_date=...
Response: List[WeighingAct]

# Экспорт в Excel
POST /api/export/excel
Body: {"start_date": "...", "end_date": "..."}
Response: Excel файл (download)

# Сверка с Excel
POST /api/compare/excel
Body: {"file": UploadFile}
Response: {
    "matches": int,
    "discrepancies": int,
    "accuracy": float,
    "report_url": str
}

# Получение списка видео
GET /api/videos
Response: List[{"filename": str, "duration": float, "acts_count": int}]
```

#### 2.2 Модуль экспорта (ExcelExporter)

**Назначение:** Экспорт данных в Excel с точным воспроизведением формата

**Интерфейс:**
```python
class ExcelExporter:
    def __init__(self, template_path: str = "docs/Замеры 20.07 по 03.09.xlsx"):
        """Анализирует шаблон и сохраняет схему"""
        pass
    
    def analyze_template(self) -> ExcelSchema:
        """
        Извлекает структуру шаблона:
        - Определяет столбцы: секция, коэффициент, дата
        - Находит пары (вес, количество)
        - Определяет столбцы итогов: общий вес, количество, средний вес
        - Извлекает форматирование и стили
        """
        pass
    
    def parse_excel_row(self, row: List) -> ExcelRow:
        """Парсит строку Excel в структурированный объект"""
        pass
    
    def export(self, acts: List[WeighingAct], output_path: str, section: str = "6B"):
        """
        Экспортирует акты в Excel по схеме шаблона
        Группирует акты по дате, суммирует показатели
        """
        pass
    
    def convert_act_to_excel_row(self, acts_by_date: List[WeighingAct], section: str) -> ExcelRow:
        """Конвертирует акты за день в строку Excel"""
        pass
```

#### 2.3 Модуль сверки (ExcelComparator)

**Назначение:** Сравнение автоматических результатов с ручными записями

**Интерфейс:**
```python
class ExcelComparator:
    def __init__(self, time_tolerance_minutes: int = 5):
        pass
    
    def compare(self, 
                auto_acts: List[WeighingAct], 
                manual_excel: str) -> ComparisonReport:
        """
        Сравнивает данные, возвращает отчет
        ComparisonReport: {
            "matches": List[Match],
            "discrepancies": List[Discrepancy],
            "missing_in_auto": List[WeighingAct],
            "missing_in_manual": List[WeighingAct],
            "metrics": {
                "recall": float,
                "precision": float,
                "mae_count": float,
                "mape_count": float,
                "mae_weight": float,
                "mape_weight": float
            }
        }
        """
        pass
    
    def generate_report(self, comparison: ComparisonReport, output_path: str):
        """Создает Excel отчет с цветовым выделением"""
        pass
```

## Модели данных

### База данных Supabase (PostgreSQL)

**SQL миграции для локального Supabase:**

#### Таблица: weighing_acts

```sql
-- Создание таблицы актов взвешивания
CREATE TABLE weighing_acts (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    duration_sec FLOAT NOT NULL,
    left_count INTEGER NOT NULL,
    right_count INTEGER NOT NULL,
    peak_count INTEGER NOT NULL,
    total_weight FLOAT,
    avg_weight FLOAT,
    stream_id VARCHAR(255),
    video_file VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для быстрого поиска
CREATE INDEX idx_weighing_acts_started_at ON weighing_acts(started_at);
CREATE INDEX idx_weighing_acts_video_file ON weighing_acts(video_file);

-- RLS (Row Level Security) - опционально для Supabase
ALTER TABLE weighing_acts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all operations" ON weighing_acts FOR ALL USING (true);
```

#### Таблица: crossings

```sql
-- Создание таблицы проходов
CREATE TABLE crossings (
    id BIGSERIAL PRIMARY KEY,
    act_id BIGINT REFERENCES weighing_acts(id) ON DELETE CASCADE,
    pig_id INTEGER NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'left' or 'right'
    crossed_at TIMESTAMPTZ NOT NULL,
    line_x FLOAT NOT NULL,
    line_y FLOAT NOT NULL,
    weight_estimate FLOAT,
    stream_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы
CREATE INDEX idx_crossings_act_id ON crossings(act_id);
CREATE INDEX idx_crossings_crossed_at ON crossings(crossed_at);

-- RLS
ALTER TABLE crossings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all operations" ON crossings FOR ALL USING (true);
```

#### Таблица: excel_schemas (для хранения схемы шаблона)

```sql
-- Создание таблицы схем Excel
CREATE TABLE excel_schemas (
    id BIGSERIAL PRIMARY KEY,
    template_name VARCHAR(255) NOT NULL,
    schema_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE excel_schemas ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all operations" ON excel_schemas FOR ALL USING (true);
```

### Модели Python (Pydantic)

```python
from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional, List

class CrossingEvent(BaseModel):
    pig_id: int
    direction: str  # "left" or "right"
    timestamp: datetime
    line_x: float
    line_y: float
    weight_estimate: Optional[float] = None

class WeighingAct(BaseModel):
    id: Optional[int] = None
    started_at: datetime
    ended_at: datetime
    duration_sec: float
    left_count: int
    right_count: int
    peak_count: int
    total_weight: Optional[float] = None
    avg_weight: Optional[float] = None
    stream_id: str
    video_file: Optional[str] = None
    crossings: List[CrossingEvent] = []

class ExcelRow(BaseModel):
    """Строка из Excel файла с замерами"""
    section: str  # "6B"
    coefficient: float  # 4.5, 3.4, 2.3
    date: date  # 19.07.2025
    measurements: List[tuple[int, int]]  # [(вес, количество), ...]
    total_weight: int  # 4116
    total_count: int  # 666
    avg_weight: float  # 6.2
    period_markers: List[str]  # ["подсосный период 22,7", "отъем на свиноматку 12,8"]

class ExcelSchema(BaseModel):
    """Схема Excel файла"""
    columns: List[str]
    column_types: dict
    date_format: str  # "DD.MM.YYYY"
    number_format: str  # "0.0" или "0"
    column_widths: dict
    styles: dict
    # Специфичные для формата замеров
    section_column: int  # индекс столбца с секцией
    date_column: int  # индекс столбца с датой
    measurements_start_column: int  # начало пар (вес, количество)
    total_weight_column: int  # столбец с общим весом
    total_count_column: int  # столбец с общим количеством
    avg_weight_column: int  # столбец со средним весом
```

## Обработка ошибок

### Стратегии восстановления

1. **Ошибка подключения к PostgreSQL:**
   - Переключение на локальную SQLite очередь
   - Периодические попытки переподключения (каждые 30 сек)
   - Автоматическая синхронизация при восстановлении

2. **Ошибка чтения видео:**
   - Логирование ошибки
   - Попытка переподключения (для камеры)
   - Пропуск поврежденных кадров (для файла)

3. **Ошибка инференса YOLO:**
   - Логирование ошибки
   - Пропуск кадра
   - Продолжение обработки

4. **Ошибка записи в базу:**
   - Сохранение в резервный CSV файл
   - Логирование ошибки
   - Продолжение работы

## Стратегия тестирования

### Тестовый режим

**Команда запуска:**
```bash
python console_app.py --mode test \
    --video uploads/video.mp4 \
    --start-time "2024-07-20 10:00:00" \
    --end-time "2024-09-03 18:00:00" \
    --excel-reference "docs/Замеры 20.07 по 03.09.xlsx" \
    --output test_results/run_001
```

**Процесс:**
1. Загрузка видео и модели
2. Обработка видео с выводом прогресса
3. Сохранение всех актов в базу
4. Автоматическая сверка с Excel
5. Генерация отчета с метриками
6. Сохранение результатов в `test_results/`

**Метрики точности:**
- Recall (полнота): % правильно обнаруженных актов
- Precision (точность): % актов без ложных срабатываний
- MAE (средняя абсолютная ошибка) по количеству
- MAPE (средняя процентная ошибка) по количеству
- MAE по весу
- MAPE по весу
- Корреляция Пирсона между авто и ручными подсчетами

### Юнит-тесты

```python
# tests/test_act_detector.py
def test_act_detection_start():
    """Тест начала акта при превышении порога"""
    pass

def test_act_detection_end():
    """Тест завершения акта при падении активности"""
    pass

# tests/test_crossing_counter.py
def test_crossing_left_to_right():
    """Тест подсчета прохода слева направо"""
    pass

def test_crossing_cooldown():
    """Тест cooldown для предотвращения двойного подсчета"""
    pass

# tests/test_excel_comparator.py
def test_comparison_exact_match():
    """Тест сверки при точном совпадении"""
    pass

def test_comparison_with_discrepancy():
    """Тест сверки с расхождениями"""
    pass
```

## Извлеченные гиперпараметры и настройки из существующей системы

### Параметры модели и инференса (из core/config.py)

```python
# Модель
MODEL_PATH = "models/pig_yolo11-seg.v4.pt"
DETECTION_MODE = "pig-only"
PIG_CLASS_ID = 0
CONF_THRESHOLD = 0.30  # Порог уверенности детекции

# Устройство и производительность
DEVICE = "auto"  # Автоопределение cuda/cpu
USE_HALF = True  # FP16 на GPU для экономии памяти
IMG_SIZE = 960  # Размер входного изображения
BATCH_SIZE = 4  # Размер батча
MAX_WAIT_MS = 50  # Максимальное время ожидания батча

# Профили производительности (автоматический выбор по GPU)
PERFORMANCE_PROFILES = {
    'ULTRA_PERFORMANCE': {'TARGET_FPS': 50, 'BATCH_MAX_SIZE': 12, 'IMG_SIZE': 960},
    'RTX_OPTIMIZED': {'TARGET_FPS': 35, 'BATCH_MAX_SIZE': 8, 'IMG_SIZE': 832},
    'BALANCED': {'TARGET_FPS': 25, 'BATCH_MAX_SIZE': 6, 'IMG_SIZE': 768},
    'POWER_SAVING': {'TARGET_FPS': 20, 'BATCH_MAX_SIZE': 4, 'IMG_SIZE': 640},
    'CPU_ONLY': {'TARGET_FPS': 15, 'BATCH_MAX_SIZE': 2, 'IMG_SIZE': 640}
}
```

### Параметры трекинга (из api/app.py SimpleTracker)

```python
# SimpleTracker
IOU_THRESHOLD = 0.3  # Порог IoU для сопоставления треков
MAX_AGE = 30  # Максимальный возраст трека без обновлений (кадры)
DIST_WEIGHT = 0.2  # Вес расстояния в функции стоимости
AGE_DECREMENT = 2  # Уменьшение age для несопоставленных треков
```

### Параметры подсчета проходов (из core/config.py и api/app.py)

```python
# Позиции линий детекции (нормализованные 0.0-1.0)
LINE_LEFT_X = 0.25
LINE_RIGHT_X = 0.75

# Cooldown для предотвращения двойного подсчета
CROSS_COOLDOWN_SEC = 1.0

# Окна и пороги
AVG_WINDOW = 20  # Размер окна для усреднения
FRAME_SKIP = 3  # Пропуск кадров при высокой нагрузке
COUNT_WINDOW_SEC = 10.0  # Окно для подсчета
COUNT_DECAY_HALFLIFE_SEC = 4.0  # Полураспад для оценки количества
COUNT_SOFTMAX_BETA = 0.8  # Параметр softmax для оценки
```

### Параметры предобработки (из core/preprocess.py)

```python
# center_crop_resize() - основной метод предобработки
# - Масштабирование с сохранением пропорций
# - Padding до квадрата target_size x target_size
# - Черные поля (letterbox) для сохранения пропорций
# - Возврат transform_meta для обратного маппинга

# Альтернативные методы (не используются в консольном приложении):
# - letterbox_resize() - классический letterbox
# - adaptive_preprocess() - адаптивная предобработка
# - hsv_filter() - HSV фильтрация (опционально)
```

### Параметры актов взвешивания (новые, настраиваемые)

```python
# Пороги для определения акта взвешивания
MIN_PIGS_FOR_ACT = 3  # Минимум свиней для начала акта
MAX_INTERVAL_SEC = 30.0  # Макс интервал без активности для завершения
MIN_ACT_DURATION_SEC = 10.0  # Минимальная длительность акта
```

## Конфигурация

### Файл .env (расширенный с извлеченными параметрами)

```env
# База данных Supabase (локально через Docker)
SUPABASE_URL=http://localhost:54321
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0

# Модель YOLO (из существующей системы)
MODEL_PATH=models/pig_yolo11-seg.v4.pt
DETECTION_MODE=pig-only
PIG_CLASS_ID=0
CONF_THRESHOLD=0.30

# Устройство и производительность (из существующей системы)
DEVICE=auto  # auto, cuda, cpu
USE_HALF=true
IMG_SIZE=960
BATCH_SIZE=4
MAX_WAIT_MS=50

# Трекинг (из SimpleTracker)
IOU_THRESHOLD=0.3
MAX_AGE=30
DIST_WEIGHT=0.2

# Линии детекции (из существующей системы)
LINE_LEFT_X=0.25
LINE_RIGHT_X=0.75
CROSS_COOLDOWN_SEC=1.0

# Параметры актов взвешивания (новые)
MIN_PIGS_FOR_ACT=3
MAX_INTERVAL_SEC=30.0
MIN_ACT_DURATION_SEC=10.0

# Excel шаблон
EXCEL_TEMPLATE=docs/Замеры 20.07 по 03.09.xlsx

# Сверка
TIME_TOLERANCE_MINUTES=5
```

## Развертывание

### Вариант 1: Локальный PostgreSQL

```bash
# Установка PostgreSQL
# Windows: скачать с postgresql.org
# Linux: sudo apt install postgresql

# Создание базы
createdb pigweight

# Запуск приложения
python console_app.py --mode realtime --source camera
```

### Вариант 2: Локальный Supabase через Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: supabase/postgres:latest
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: your-password
    volumes:
      - postgres-data:/var/lib/postgresql/data
  
  supabase:
    image: supabase/supabase:latest
    ports:
      - "54321:8000"
    depends_on:
      - postgres
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PASSWORD: your-password

volumes:
  postgres-data:
```

```bash
# Запуск
docker-compose up -d

# Запуск приложения
python console_app.py --mode realtime --source camera
```

## Оптимизации для производительности

1. **Батчинг инференса** - использовать существующий `DynamicBatcher`
2. **Пропуск кадров** - обрабатывать каждый N-й кадр при высокой нагрузке
3. **Асинхронная запись в БД** - очередь для неблокирующей записи
4. **Кэширование схемы Excel** - загружать один раз при старте
5. **Индексы БД** - на временные метки и video_file для быстрых запросов

## Следующие шаги

После утверждения дизайна:
1. Создание структуры проекта
2. Реализация модулей по приоритету
3. Интеграция с существующим кодом
4. Тестирование на реальных видео
5. Оптимизация параметров
