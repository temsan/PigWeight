# Документ проектирования

## Обзор

Система отслеживания и взвешивания свиней состоит из двух основных компонентов:

1. **Консольное приложение** - универсальный инструмент для:
   - Работы в фоне с камерой в режиме реального времени (24/7)
   - Обработки видеофайлов с прогрессом
   - Тестирования с автоматической сверкой результатов
   - Обнаружения актов взвешивания и сохранения в базу данных

2. **Веб-интерфейс (опционально)** - отображает показатели в реальном времени без видео, предоставляет функции экспорта и сверки

Консольное приложение является основным компонентом системы и используется для всех режимов работы, включая фоновую обработку с камеры.

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

1. **Режим реального времени (фоновый)** - обработка с камеры в фоне, непрерывная работа 24/7, используется консольное приложение
2. **Режим обработки файла** - обработка видеофайла с прогрессом, используется консольное приложение
3. **Тестовый режим** - обработка видеофайла + автоматическая сверка с Excel, используется консольное приложение

## Компоненты и интерфейсы

### 1. Консольное приложение

#### 1.1 Модуль захвата видео (VideoCapture)

**Назначение:** Получение кадров из различных источников

**Обоснование дизайна:**
- Единый интерфейс для работы с видеофайлами и камерами упрощает переключение между источниками
- Отслеживание прогресса необходимо для выполнения Требования 1.2 (отображение прогресса в консоли)
- Временные метки кадров критичны для Требования 2 (обнаружение актов по времени)

**Интерфейс:**
```python
class VideoCapture:
    def __init__(self, source: str, mode: str = "realtime"):
        """
        source: путь к файлу или URL камеры
        mode: "realtime", "file", "test"
        
        Требования: 1.1 (загрузка видеофайла)
        """
        pass
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Читает следующий кадр
        
        Требования: 1.1 (обработка кадров)
        """
        pass
    
    def get_progress(self) -> float:
        """
        Возвращает прогресс обработки (0.0-1.0)
        
        Требования: 1.2 (отображение прогресса)
        """
        pass
    
    def get_timestamp(self) -> datetime:
        """
        Возвращает временную метку текущего кадра
        
        Требования: 1.5, 2.1, 2.3 (временные метки для проходов и актов)
        """
        pass
    
    def get_total_frames(self) -> int:
        """
        Возвращает общее количество кадров в видео
        
        Требования: 1.2 (расчет процента выполнения)
        """
        pass
    
    def get_current_frame_number(self) -> int:
        """
        Возвращает номер текущего кадра
        
        Требования: 1.2 (отображение количества обработанных кадров)
        """
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

**Обоснование дизайна:**
- Cooldown механизм предотвращает двойной подсчет (Требование 6.1, 6.2)
- Интерполяция Y-координаты обеспечивает точное определение позиции прохода (Требование 6.5)
- Отслеживание направления необходимо для Требования 1.5 и 6.4
- Использование уникальных ID треков обеспечивает различение свиней (Требование 6.3)

**Используем существующую логику:**
- `api/app.py` → `VideoStream._update_line_counters()` - подсчет проходов

**Извлеченные параметры и алгоритм:**

```python
# Из core/config.py
LINE_LEFT_X = 0.25  # Позиция левой линии (0.0-1.0) - Требование 8.3
LINE_RIGHT_X = 0.75  # Позиция правой линии (0.0-1.0) - Требование 8.3
CROSS_COOLDOWN_SEC = 1.0  # Cooldown между проходами одной свиньи - Требование 6.1

# Алгоритм из api/app.py VideoStream._update_line_counters():
# 1. Для каждого трека сохраняем предыдущую позицию (prev_x, prev_y)
# 2. Определяем, был ли трек внутри зоны (между линиями) ранее
# 3. Определяем, находится ли трек внутри зоны сейчас
# 4. Детектируем события (Требование 6.4):
#    - Вход слева: prev < LINE_LEFT_X <= current (left_in++)
#    - Вход справа: prev > LINE_RIGHT_X >= current (right_in++)
#    - Выход слева: prev >= LINE_LEFT_X > current (left_flow--)
#    - Выход справа: prev <= LINE_RIGHT_X < current (right_flow--)
# 5. Интерполяция Y-координаты в точке пересечения линии (Требование 6.5)
# 6. Проверка cooldown: не считать проход, если прошло < CROSS_COOLDOWN_SEC (Требование 6.2)
# 7. Логирование события с временной меткой и координатами (Требование 1.5)
```

**Интерфейс (адаптируем из существующего):**
```python
class CrossingCounter:
    def __init__(self, 
                 left_line_x: float = 0.25, 
                 right_line_x: float = 0.75, 
                 cooldown_sec: float = 1.0):
        """
        Требования: 6.1 (cooldown), 8.3 (конфигурируемые позиции линий)
        """
        self.left_line_x = left_line_x
        self.right_line_x = right_line_x
        self.cooldown_sec = cooldown_sec
        self._track_prev_x = {}  # track_id -> prev_x
        self._track_prev_y = {}  # track_id -> prev_y
        self._track_is_inside = {}  # track_id -> bool
        self._track_last_cross_time = {}  # track_id -> timestamp
    
    def process_tracks(self, tracks, current_time) -> List[CrossingEvent]:
        """
        Обрабатывает треки и возвращает события пересечения линий
        
        Требования: 1.5 (регистрация проходов), 6.1-6.5 (предотвращение двойного подсчета)
        """
        # Реализация на основе VideoStream._update_line_counters()
        pass
```

#### 1.5 Модуль обнаружения актов (ActDetector)

**Назначение:** Автоматическое обнаружение начала и конца актов взвешивания

**Обоснование дизайна:**
- Порог в 3 свиньи отфильтровывает одиночные проходы (Требование 2.1, 2.4)
- Интервал без активности 30 секунд определяет естественное завершение акта (Требование 2.3)
- Накопление статистики во время акта необходимо для Требования 2.2
- Вычисление длительности требуется для Требования 2.5 и сохранения в БД (Требование 3.2)

**Интерфейс:**
```python
class ActDetector:
    def __init__(self, 
                 min_pigs_threshold: int = 3,
                 max_interval_sec: float = 30.0,
                 min_duration_sec: float = 10.0):
        """
        min_pigs_threshold: минимум свиней для начала акта (Требование 2.1)
        max_interval_sec: макс интервал без активности для завершения акта (Требование 2.3)
        min_duration_sec: минимальная длительность акта
        
        Требования: 2.1, 2.3, 8.4 (конфигурируемые параметры)
        """
        pass
    
    def update(self, crossings: List[CrossingEvent], current_time: datetime) -> Optional[ActEvent]:
        """
        Обновляет состояние, возвращает событие акта (start/end)
        ActEvent: {type: "start"|"end", timestamp, act_id}
        
        Требования: 2.1 (начало акта), 2.3 (завершение акта)
        """
        pass
    
    def get_current_act(self) -> Optional[WeighingAct]:
        """
        Возвращает текущий активный акт или None
        
        Требования: 2.2 (накопление статистики), 9.2 (текущая статистика для API)
        """
        pass
    
    def accumulate_statistics(self, crossing: CrossingEvent):
        """
        Накапливает статистику по проходам во время активного акта
        
        Требования: 2.2 (проходы слева, справа, пиковое количество, уникальные свиньи)
        """
        pass
    
    def calculate_duration(self) -> float:
        """
        Вычисляет длительность акта в секундах
        
        Требования: 2.5 (вычисление длительности)
        """
        pass
```

#### 1.6 Модуль базы данных (DatabaseManager для Supabase)

**Назначение:** Сохранение данных в локальный Supabase

**Обоснование дизайна:**
- Supabase выбран как современная альтернатива PostgreSQL с встроенным REST API (Требование 3.1)
- Разделение на таблицы weighing_acts и crossings обеспечивает нормализацию данных (Требование 3.2, 3.3)
- Graceful degradation при недоступности БД обеспечивает непрерывность работы (Требование 3.4)
- SQL миграции обеспечивают автоматическое создание схемы (Требование 3.5)

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
        
        Требования: 3.1 (подключение к БД)
        """
        self.client: Client = create_client(supabase_url, supabase_key)
    
    def save_crossing(self, crossing: CrossingEvent) -> int:
        """
        Сохраняет проход через supabase.table('crossings').insert()
        
        Требования: 3.3 (сохранение проходов с полями act_id, pig_id, direction, crossed_at, line_x, line_y)
        """
        try:
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
        except Exception as e:
            # Требование 3.4: логирование ошибки и продолжение работы
            logger.error(f"Failed to save crossing: {e}")
            return -1
    
    def save_weighing_act(self, act: WeighingAct) -> int:
        """
        Сохраняет акт взвешивания
        
        Требования: 3.2 (сохранение актов с полями started_at, ended_at, duration_sec, 
                    left_count, right_count, peak_count, stream_id, video_file)
        """
        try:
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
        except Exception as e:
            # Требование 3.4: логирование ошибки и продолжение работы
            logger.error(f"Failed to save weighing act: {e}")
            return -1
    
    def get_acts_by_period(self, start: datetime, end: datetime) -> List[WeighingAct]:
        """
        Получает акты за период через Supabase query
        
        Требования: 4.2 (получение актов за период для экспорта), 9.1 (API для получения актов)
        """
        result = self.client.table('weighing_acts')\
            .select('*')\
            .gte('started_at', start.isoformat())\
            .lte('started_at', end.isoformat())\
            .execute()
        return [WeighingAct(**row) for row in result.data]
    
    def get_current_stats(self) -> dict:
        """
        Получает текущую статистику для веб-интерфейса
        
        Требования: 9.2 (текущая статистика: количество свиней, проходы, активный акт)
        """
        # Реализация получения последнего активного акта и статистики
        pass
    
    def run_migrations(self):
        """
        Выполняет SQL миграции для создания таблиц
        
        Требования: 3.5 (автоматическое создание таблиц при первом запуске)
        """
        # Выполнение SQL миграций из supabase/migrations/
        pass
```

### 2. Веб-интерфейс

#### 2.1 REST API (FastAPI)

**Обоснование дизайна:**
- REST API обеспечивает доступ к данным для веб-интерфейса (Требование 9)
- Разделение на эндпоинты по функциональности упрощает поддержку
- Использование FastAPI обеспечивает автоматическую документацию и валидацию

**Эндпоинты:**

```python
# Получение текущих показателей
# Требования: 9.2 (текущая статистика)
GET /api/weighing/stats
Response: {
    "current_count": int,
    "left_count": int,
    "right_count": int,
    "total_weight": float,
    "avg_weight": float,
    "active_act": Optional[WeighingAct]
}

# Получение истории актов
# Требования: 9.1 (список актов за период)
GET /api/weighing/acts?start_date=...&end_date=...
Response: List[WeighingAct]

# Экспорт в Excel
# Требования: 9.3 (экспорт с возможностью скачивания)
POST /api/weighing/export
Body: {"start_date": "...", "end_date": "..."}
Response: Excel файл (download)

# Сверка с Excel
# Требования: 5.1 (загрузка файла для сверки)
POST /api/compare/excel
Body: {"file": UploadFile}
Response: {
    "matches": int,
    "discrepancies": int,
    "accuracy": float,
    "report_url": str,
    "metrics": {
        "recall": float,
        "precision": float,
        "f1_score": float,
        "mae": float,
        "mape": float,
        "correlation": float
    }
}

# Получение списка видео
GET /api/videos
Response: List[{"filename": str, "duration": float, "acts_count": int}]
```

#### 2.2 Модуль экспорта (ExcelExporter)

**Назначение:** Экспорт данных в Excel с точным воспроизведением формата

**Обоснование дизайна:**
- Анализ шаблона обеспечивает точное воспроизведение формата (Требование 4.1, 4.4)
- Группировка по датам необходима для Требования 4.3
- Сохранение стилей обеспечивает профессиональный вид отчетов (Требование 4.4)
- Имя файла с датой упрощает организацию отчетов (Требование 4.5)

**Интерфейс:**
```python
class ExcelExporter:
    def __init__(self, template_path: str = "docs/Замеры 20.07 по 03.09.xlsx"):
        """
        Анализирует шаблон и сохраняет схему
        
        Требования: 4.1 (анализ структуры шаблона)
        """
        pass
    
    def analyze_template(self) -> ExcelSchema:
        """
        Извлекает структуру шаблона:
        - Определяет столбцы: секция, коэффициент, дата
        - Находит пары (вес, количество)
        - Определяет столбцы итогов: общий вес, количество, средний вес
        - Извлекает форматирование и стили
        
        Требования: 4.1 (определение формата столбцов и стилей)
        """
        pass
    
    def parse_excel_row(self, row: List) -> ExcelRow:
        """
        Парсит строку Excel в структурированный объект
        
        Требования: 4.1 (анализ структуры)
        """
        pass
    
    def export(self, acts: List[WeighingAct], output_path: str, section: str = "6B"):
        """
        Экспортирует акты в Excel по схеме шаблона
        Группирует акты по дате, суммирует показатели
        
        Требования: 4.2 (получение актов), 4.3 (группировка по дате и суммирование),
                    4.4 (структура и стили), 4.5 (сохранение с датой)
        """
        pass
    
    def group_acts_by_date(self, acts: List[WeighingAct]) -> Dict[date, List[WeighingAct]]:
        """
        Группирует акты по дате
        
        Требования: 4.3 (группировка по дате)
        """
        pass
    
    def summarize_by_date(self, acts: List[WeighingAct]) -> dict:
        """
        Суммирует метрики для актов за один день
        
        Требования: 4.3 (суммирование количества проходов, общего веса, среднего веса)
        """
        pass
    
    def convert_act_to_excel_row(self, acts_by_date: List[WeighingAct], section: str) -> ExcelRow:
        """
        Конвертирует акты за день в строку Excel
        
        Требования: 4.3, 4.4 (форматирование данных)
        """
        pass
    
    def generate_filename_with_date(self, base_path: str) -> str:
        """
        Генерирует имя файла с датой экспорта
        
        Требования: 4.5 (имя файла с датой)
        """
        pass
```

#### 2.3 Веб-страница для мониторинга

**Назначение:** Адаптивный веб-интерфейс для мониторинга системы с мобильных устройств

**Обоснование дизайна:**
- Адаптивный дизайн обеспечивает удобство использования на смартфонах (Требование 9.4)
- Автоматическое обновление каждые 5 секунд обеспечивает актуальность данных (Требование 9.5)
- Минималистичный интерфейс без видео снижает нагрузку на сеть
- Использование WebSocket или polling для real-time обновлений

**Компоненты интерфейса:**

```html
<!-- Требования: 9.4 (адаптивный дизайн для мобильных устройств) -->
<div class="mobile-dashboard">
    <!-- Текущие показатели -->
    <div class="stats-panel">
        <div class="stat-card">
            <h3>Текущее количество</h3>
            <span id="current-count">0</span>
        </div>
        <div class="stat-card">
            <h3>Проходы слева</h3>
            <span id="left-count">0</span>
        </div>
        <div class="stat-card">
            <h3>Проходы справа</h3>
            <span id="right-count">0</span>
        </div>
        <div class="stat-card">
            <h3>Средний вес</h3>
            <span id="avg-weight">0.0</span> кг
        </div>
    </div>
    
    <!-- Активный акт взвешивания -->
    <div class="active-act" id="active-act">
        <h3>Активный акт</h3>
        <p>Начало: <span id="act-start">-</span></p>
        <p>Длительность: <span id="act-duration">-</span></p>
        <p>Пиковое количество: <span id="peak-count">-</span></p>
    </div>
    
    <!-- История актов -->
    <div class="acts-history">
        <h3>История актов</h3>
        <div id="acts-list"></div>
    </div>
</div>
```

**JavaScript для автообновления:**

```javascript
// Требования: 9.5 (автоматическое обновление каждые 5 секунд)
class DashboardUpdater {
    constructor(updateInterval = 5000) {
        this.updateInterval = updateInterval;
        this.intervalId = null;
    }
    
    start() {
        // Первое обновление сразу
        this.updateStats();
        
        // Периодическое обновление
        this.intervalId = setInterval(() => {
            this.updateStats();
        }, this.updateInterval);
    }
    
    async updateStats() {
        try {
            // Требование 9.2: получение текущей статистики
            const response = await fetch('/api/weighing/stats');
            const data = await response.json();
            
            // Обновление UI
            document.getElementById('current-count').textContent = data.current_count;
            document.getElementById('left-count').textContent = data.left_count;
            document.getElementById('right-count').textContent = data.right_count;
            document.getElementById('avg-weight').textContent = data.avg_weight.toFixed(1);
            
            // Обновление активного акта
            if (data.active_act) {
                this.updateActiveAct(data.active_act);
            }
        } catch (error) {
            console.error('Failed to update stats:', error);
        }
    }
    
    updateActiveAct(act) {
        document.getElementById('act-start').textContent = 
            new Date(act.started_at).toLocaleTimeString();
        document.getElementById('act-duration').textContent = 
            `${Math.floor(act.duration_sec / 60)} мин`;
        document.getElementById('peak-count').textContent = act.peak_count;
    }
    
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
        }
    }
}

// Запуск при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    const updater = new DashboardUpdater(5000); // 5 секунд
    updater.start();
});
```

**CSS для адаптивного дизайна:**

```css
/* Требования: 9.4 (адаптивный дизайн) */
.mobile-dashboard {
    max-width: 100%;
    padding: 1rem;
}

.stats-panel {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: #fff;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.stat-card h3 {
    font-size: 0.875rem;
    color: #666;
    margin-bottom: 0.5rem;
}

.stat-card span {
    font-size: 2rem;
    font-weight: bold;
    color: #333;
}

/* Адаптация для маленьких экранов */
@media (max-width: 480px) {
    .stats-panel {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .stat-card span {
        font-size: 1.5rem;
    }
}
```

#### 2.4 Модуль сверки (ExcelComparator)

**Назначение:** Сравнение автоматических результатов с ручными записями

**Обоснование дизайна:**
- Допуск ±5 минут учитывает возможные расхождения в синхронизации времени (Требование 5.2)
- Четыре листа отчета обеспечивают полную картину сверки (Требование 5.5)
- Цветовое выделение упрощает визуальный анализ (Требование 5.5)
- Метрики точности позволяют количественно оценить работу системы (Требование 5.4, 7.4)

**Интерфейс:**
```python
class ExcelComparator:
    def __init__(self, time_tolerance_minutes: int = 5):
        """
        time_tolerance_minutes: допуск для сопоставления по времени
        
        Требования: 5.2 (допуск ±5 минут)
        """
        pass
    
    def compare(self, 
                auto_acts: List[WeighingAct], 
                manual_excel: str) -> ComparisonReport:
        """
        Сравнивает данные, возвращает отчет
        
        Требования: 5.1 (загрузка ручных записей), 5.2 (сопоставление по времени),
                    5.3 (сравнение количества и веса), 5.4 (вычисление метрик)
        
        ComparisonReport: {
            "matches": List[Match],
            "discrepancies": List[Discrepancy],
            "missing_in_auto": List[WeighingAct],
            "missing_in_manual": List[WeighingAct],
            "metrics": {
                "recall": float,
                "precision": float,
                "f1_score": float,  # Требование 7.4
                "mae_count": float,
                "mape_count": float,
                "mae_weight": float,
                "mape_weight": float,
                "correlation": float  # Требование 5.4
            }
        }
        """
        pass
    
    def load_manual_records(self, excel_path: str) -> List[WeighingAct]:
        """
        Загружает ручные записи из Excel
        
        Требования: 5.1 (загрузка ручных записей)
        """
        pass
    
    def match_acts_by_time(self, auto_acts: List[WeighingAct], 
                           manual_acts: List[WeighingAct]) -> List[tuple]:
        """
        Сопоставляет акты по времени с допуском
        
        Требования: 5.2 (сопоставление с допуском ±5 минут)
        """
        pass
    
    def compare_metrics(self, auto_act: WeighingAct, manual_act: WeighingAct) -> dict:
        """
        Сравнивает метрики двух актов
        
        Требования: 5.3 (сравнение количества проходов и веса)
        """
        pass
    
    def calculate_accuracy_metrics(self, comparison: ComparisonReport) -> dict:
        """
        Вычисляет метрики точности
        
        Требования: 5.4 (Recall, Precision, MAE, MAPE, корреляция), 7.4 (F1-score)
        """
        pass
    
    def generate_report(self, comparison: ComparisonReport, output_path: str):
        """
        Создает Excel отчет с цветовым выделением
        
        Требования: 5.5 (четыре листа: совпадения зеленый, расхождения желтый/красный,
                    пропущенные серый, метрики точности), 5.6 (сохранение в test_results)
        """
        pass
    
    def generate_filename_with_timestamp(self, base_path: str) -> str:
        """
        Генерирует имя файла с датой и временем
        
        Требования: 5.6 (имя файла с датой и временем)
        """
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

**Обоснование дизайна:**
- Автоматическое тестирование обеспечивает быструю оценку точности (Требование 7)
- Сохранение результатов позволяет отслеживать улучшения со временем (Требование 7.5)
- Вывод метрик в консоль обеспечивает быструю обратную связь (Требование 7.4)
- JSON файл с метриками позволяет автоматизировать анализ (Требование 7.5)

### Тестовый режим

**Команда запуска:**
```bash
# Требования: 7.1 (активация тестового режима)
python console_app.py --mode test \
    --video uploads/video.mp4 \
    --excel-reference "docs/Замеры 20.07 по 03.09.xlsx" \
    --output test_results/run_001
```

**Процесс:**
1. Загрузка видео и модели (Требование 7.1)
2. Обработка видео с выводом прогресса (Требование 7.2)
3. Сохранение всех актов в базу (Требование 7.2)
4. Автоматическая сверка с Excel (Требование 7.3)
5. Генерация отчета с метриками (Требование 7.3)
6. Вывод метрик в консоль (Требование 7.4)
7. Сохранение результатов в `test_results/` (Требование 7.5)

**Метрики точности (Требования 7.4, 5.4):**
- Recall (полнота): % правильно обнаруженных актов
- Precision (точность): % актов без ложных срабатываний
- F1-score: гармоническое среднее Recall и Precision (Требование 7.4)
- MAE (средняя абсолютная ошибка) по количеству
- MAPE (средняя процентная ошибка) по количеству
- MAE по весу
- MAPE по весу
- Корреляция Пирсона между авто и ручными подсчетами

**Выходные файлы (Требование 7.5):**
- `comparison_report_YYYY-MM-DD_HH-MM-SS.xlsx` - отчет сверки с цветовым выделением
- `metrics_YYYY-MM-DD_HH-MM-SS.json` - метрики точности в JSON формате

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

**Обоснование дизайна:**
- Файл .env обеспечивает простую настройку без изменения кода (Требование 8.1)
- Все критические параметры вынесены в конфигурацию (Требования 8.2, 8.3, 8.4)
- Автоопределение устройства упрощает развертывание (Требование 8.5)
- Комментарии в файле помогают администраторам понять назначение параметров

### Файл .env (расширенный с извлеченными параметрами)

```env
# База данных Supabase (локально через Docker)
# Требования: 3.1 (подключение к БД)
SUPABASE_URL=http://localhost:54321
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0

# Модель YOLO (из существующей системы)
# Требования: 1.3 (детекция с YOLO), 8.1 (загрузка из конфигурации)
MODEL_PATH=models/pig_yolo11-seg.v4.pt
DETECTION_MODE=pig-only
PIG_CLASS_ID=0

# Порог уверенности детекции
# Требования: 1.3 (порог 0.30), 8.2 (конфигурируемый порог)
CONF_THRESHOLD=0.30

# Устройство и производительность (из существующей системы)
# Требования: 8.5 (автоопределение устройства)
DEVICE=auto  # auto, cuda, cpu - автоматический выбор оптимального устройства
USE_HALF=true
IMG_SIZE=960
BATCH_SIZE=4
MAX_WAIT_MS=50

# Трекинг (из SimpleTracker)
# Требования: 1.4 (трекинг с уникальными ID), 6.3 (использование ID треков)
IOU_THRESHOLD=0.3
MAX_AGE=30
DIST_WEIGHT=0.2

# Линии детекции (из существующей системы)
# Требования: 8.3 (конфигурируемые позиции линий)
LINE_LEFT_X=0.25  # Позиция левой линии (0.0-1.0)
LINE_RIGHT_X=0.75  # Позиция правой линии (0.0-1.0)

# Cooldown для предотвращения двойного подсчета
# Требования: 6.1 (период ожидания 1.0 секунды)
CROSS_COOLDOWN_SEC=1.0

# Параметры актов взвешивания (новые)
# Требования: 8.4 (конфигурируемые параметры актов)
MIN_PIGS_FOR_ACT=3  # Минимум свиней для начала акта (Требование 2.1)
MAX_INTERVAL_SEC=30.0  # Макс интервал без активности для завершения (Требование 2.3)
MIN_ACT_DURATION_SEC=10.0  # Минимальная длительность акта

# Excel шаблон
# Требования: 4.1 (анализ структуры шаблона)
EXCEL_TEMPLATE=docs/Замеры 20.07 по 03.09.xlsx

# Сверка
# Требования: 5.2 (допуск ±5 минут)
TIME_TOLERANCE_MINUTES=5
```

### Загрузка конфигурации

```python
# Требования: 8.1 (загрузка параметров из .env при запуске)
from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()
        
        # База данных
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        # Модель
        self.model_path = os.getenv('MODEL_PATH')
        self.conf_threshold = float(os.getenv('CONF_THRESHOLD', 0.30))
        
        # Устройство (Требование 8.5: автоопределение)
        self.device = self._determine_device(os.getenv('DEVICE', 'auto'))
        self.use_half = os.getenv('USE_HALF', 'true').lower() == 'true'
        self.img_size = int(os.getenv('IMG_SIZE', 960))
        
        # Линии детекции (Требование 8.3)
        self.line_left_x = float(os.getenv('LINE_LEFT_X', 0.25))
        self.line_right_x = float(os.getenv('LINE_RIGHT_X', 0.75))
        self.cross_cooldown_sec = float(os.getenv('CROSS_COOLDOWN_SEC', 1.0))
        
        # Параметры актов (Требование 8.4)
        self.min_pigs_for_act = int(os.getenv('MIN_PIGS_FOR_ACT', 3))
        self.max_interval_sec = float(os.getenv('MAX_INTERVAL_SEC', 30.0))
        self.min_act_duration_sec = float(os.getenv('MIN_ACT_DURATION_SEC', 10.0))
        
        # Excel
        self.excel_template = os.getenv('EXCEL_TEMPLATE')
        self.time_tolerance_minutes = int(os.getenv('TIME_TOLERANCE_MINUTES', 5))
    
    def _determine_device(self, device_config: str) -> str:
        """
        Автоматически определяет оптимальное устройство
        
        Требования: 8.5 (автоопределение CUDA GPU или CPU)
        """
        if device_config == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_config
```

## Развертывание

### Вариант 1: Локальный PostgreSQL

```bash
# Установка PostgreSQL
# Windows: скачать с postgresql.org
# Linux: sudo apt install postgresql

# Создание базы
createdb pigweight

# Запуск консольного приложения в фоне с камерой
python console_app.py --mode realtime --source camera

# Для Windows: создать задачу в планировщике задач для автозапуска при старте системы
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
# Запуск Supabase
docker-compose up -d

# Запуск консольного приложения в фоне с камерой
python console_app.py --mode realtime --source camera

# Для Windows: создать задачу в планировщике задач для автозапуска при старте системы
# Для Linux: создать systemd service для автозапуска
```

## Оптимизации для производительности

1. **Батчинг инференса** - использовать существующий `DynamicBatcher`
2. **Пропуск кадров** - обрабатывать каждый N-й кадр при высокой нагрузке
3. **Асинхронная запись в БД** - очередь для неблокирующей записи
4. **Кэширование схемы Excel** - загружать один раз при старте
5. **Индексы БД** - на временные метки и video_file для быстрых запросов

## Покрытие требований

Эта таблица показывает, как каждое требование из документа требований реализовано в дизайне:

| Требование | Компоненты дизайна | Статус |
|------------|-------------------|--------|
| **1. Обработка видеофайлов** | | |
| 1.1 Загрузка видео с --video | VideoCapture.__init__() | ✅ |
| 1.2 Отображение прогресса | VideoCapture.get_progress(), get_current_frame_number() | ✅ |
| 1.3 Детекция с YOLO (порог 0.30) | UnifiedVideoProcessor, CONF_THRESHOLD | ✅ |
| 1.4 Трекинг с уникальными ID | SimpleTracker | ✅ |
| 1.5 Регистрация проходов | CrossingCounter.process_tracks() | ✅ |
| **2. Обнаружение актов** | | |
| 2.1 Начало акта (≥3 свиньи) | ActDetector.update(), MIN_PIGS_FOR_ACT | ✅ |
| 2.2 Накопление статистики | ActDetector.accumulate_statistics() | ✅ |
| 2.3 Завершение акта (30 сек) | ActDetector.update(), MAX_INTERVAL_SEC | ✅ |
| 2.4 Игнорирование <3 проходов | ActDetector логика фильтрации | ✅ |
| 2.5 Вычисление длительности | ActDetector.calculate_duration() | ✅ |
| **3. Хранение в БД** | | |
| 3.1 Подключение к Supabase | DatabaseManager.__init__() | ✅ |
| 3.2 Сохранение актов | DatabaseManager.save_weighing_act() | ✅ |
| 3.3 Сохранение проходов | DatabaseManager.save_crossing() | ✅ |
| 3.4 Graceful degradation | try/except в DatabaseManager | ✅ |
| 3.5 SQL миграции | DatabaseManager.run_migrations(), SQL схемы | ✅ |
| **4. Экспорт в Excel** | | |
| 4.1 Анализ шаблона | ExcelExporter.analyze_template() | ✅ |
| 4.2 Получение актов за период | DatabaseManager.get_acts_by_period() | ✅ |
| 4.3 Группировка и суммирование | ExcelExporter.group_acts_by_date(), summarize_by_date() | ✅ |
| 4.4 Структура и стили | ExcelExporter.export() | ✅ |
| 4.5 Имя файла с датой | ExcelExporter.generate_filename_with_date() | ✅ |
| **5. Сверка с Excel** | | |
| 5.1 Загрузка ручных записей | ExcelComparator.load_manual_records() | ✅ |
| 5.2 Сопоставление (±5 мин) | ExcelComparator.match_acts_by_time() | ✅ |
| 5.3 Сравнение метрик | ExcelComparator.compare_metrics() | ✅ |
| 5.4 Вычисление метрик точности | ExcelComparator.calculate_accuracy_metrics() | ✅ |
| 5.5 Отчет с 4 листами | ExcelComparator.generate_report() | ✅ |
| 5.6 Сохранение в test_results | ExcelComparator.generate_filename_with_timestamp() | ✅ |
| **6. Предотвращение двойного подсчета** | | |
| 6.1 Cooldown 1.0 сек | CrossingCounter.cooldown_sec, CROSS_COOLDOWN_SEC | ✅ |
| 6.2 Игнорирование в cooldown | CrossingCounter.process_tracks() логика | ✅ |
| 6.3 Использование ID треков | SimpleTracker, CrossingCounter | ✅ |
| 6.4 Определение направления | CrossingCounter алгоритм детекции | ✅ |
| 6.5 Интерполяция Y-координаты | CrossingCounter алгоритм интерполяции | ✅ |
| **7. Тестовый режим** | | |
| 7.1 Активация с --mode test | Консольное приложение CLI | ✅ |
| 7.2 Полная обработка и сохранение | Интеграция всех компонентов | ✅ |
| 7.3 Автоматическая сверка | ExcelComparator.compare() | ✅ |
| 7.4 Вывод метрик в консоль | Консольное приложение, Rich UI | ✅ |
| 7.5 Сохранение результатов | Excel отчет + JSON метрики | ✅ |
| **8. Конфигурирование** | | |
| 8.1 Загрузка из .env | Config класс, load_dotenv() | ✅ |
| 8.2 CONF_THRESHOLD | Config.conf_threshold | ✅ |
| 8.3 LINE_LEFT_X, LINE_RIGHT_X | Config.line_left_x, line_right_x | ✅ |
| 8.4 MIN_PIGS_FOR_ACT, MAX_INTERVAL_SEC | Config.min_pigs_for_act, max_interval_sec | ✅ |
| 8.5 Автоопределение устройства | Config._determine_device() | ✅ |
| **9. Веб-интерфейс** | | |
| 9.1 GET /api/weighing/acts | REST API endpoint | ✅ |
| 9.2 GET /api/weighing/stats | REST API endpoint | ✅ |
| 9.3 POST /api/weighing/export | REST API endpoint | ✅ |
| 9.4 Адаптивная веб-страница | HTML/CSS mobile-dashboard | ✅ |
| 9.5 Автообновление каждые 5 сек | DashboardUpdater JavaScript | ✅ |

**Итого: 47/47 требований покрыто дизайном (100%)**

## Следующие шаги

После утверждения дизайна:
1. Создание структуры проекта
2. Реализация модулей по приоритету
3. Интеграция с существующим кодом
4. Тестирование на реальных видео
5. Оптимизация параметров
