# 🔧 Руководство по интеграции сервисов (Фаза 2)

**Задача:** Интегрировать созданные сервисы в `api/app.py`  
**Цель:** Сократить размер с 3444 до 1500-2000 строк  
**Время:** 4-6 часов  
**Приоритет:** Критический

---

## 📋 План действий

### Шаг 1: Инициализация сервисов (30 мин)

**Файл:** `api/app.py`

**Добавить в начало файла:**
```python
from api.services import StreamService, ActService, MetricsService

# Глобальные сервисы (инициализируются при startup)
stream_service: Optional[StreamService] = None
act_service: Optional[ActService] = None
metrics_service: Optional[MetricsService] = None
```

**В функции startup добавить:**
```python
@app.on_event("startup")
async def startup_event():
    global stream_service, act_service, metrics_service
    
    # Инициализация сервисов
    stream_service = StreamService(STREAM_MANAGER)
    act_service = ActService(db_manager)
    metrics_service = MetricsService(stream_service, act_service, db_manager)
    
    logger.info("✅ Сервисы инициализированы")
```

---

### Шаг 2: Замена прямых обращений к STREAM_MANAGER (2 часа)

**Найти и заменить паттерны:**

#### Паттерн 1: Получение потока
```python
# Было:
stream = STREAM_MANAGER.streams.get(stream_id)
if not stream:
    raise HTTPException(404, "Stream not found")

# Стало:
if not stream_service.stream_exists(stream_id):
    raise HTTPException(404, "Stream not found")
stream = stream_service.get_stream(stream_id)
```

#### Паттерн 2: Список потоков
```python
# Было:
streams = list(STREAM_MANAGER.streams.keys())

# Стало:
streams = stream_service.get_active_streams()
```

#### Паттерн 3: Метрики потока
```python
# Было:
metrics = {
    "current_count": stream.reported_count,
    "left_count": stream.left_in,
    "right_count": stream.right_in
}

# Стало:
metrics = stream_service.get_stream_metrics(stream_id)
```

**Команда для поиска:**
```bash
# Найти все обращения к STREAM_MANAGER.streams
rg "STREAM_MANAGER\.streams" api/app.py
```

---

### Шаг 3: Замена логики актов (1.5 часа)

#### Паттерн 1: Получение активного акта
```python
# Было:
if hasattr(stream, 'current_act') and stream.current_act:
    act = stream.current_act
    # ... обработка

# Стало:
active_act = act_service.get_active_act(stream)
if active_act:
    # ... обработка
```

#### Паттерн 2: Завершение акта
```python
# Было:
act = stream.current_act
act['ended_at'] = datetime.now()
# ... сохранение в БД
stream.current_act = None

# Стало:
finalized_act = act_service.finalize_act(stream, manual=False)
```

#### Паттерн 3: Получение актов из БД
```python
# Было:
acts = db_manager.get_acts_by_period(start, end, stream_id)

# Стало:
acts = act_service.get_acts_by_period(start, end, stream_id)
```

**Команда для поиска:**
```bash
# Найти все обращения к current_act
rg "current_act" api/app.py
```

---

### Шаг 4: Замена вычисления метрик (1 час)

#### Паттерн 1: Текущие метрики
```python
# Было:
stats = {
    "current_count": stream.reported_count,
    "total_weight": calculate_weight(...),
    "avg_weight": calculate_avg(...),
    # ...
}

# Стало:
stats = metrics_service.get_current_metrics(stream_id)
```

#### Паттерн 2: Агрегация данных
```python
# Было:
total_pigs = sum(act['left_count'] + act['right_count'] for act in acts)
total_weight = sum(act['total_weight'] for act in acts)
# ...

# Стало:
aggregated = metrics_service.aggregate_daily_stats(acts)
```

#### Паттерн 3: Health check
```python
# Было:
db_status = "connected" if db_manager.test_connection() else "error"
stream_status = "active" if STREAM_MANAGER.streams else "idle"
# ...

# Стало:
health = metrics_service.get_system_health()
```

---

### Шаг 5: Удаление дублированного кода (1 час)

**Найти повторяющиеся блоки:**

1. **Проверка существования потока:**
```python
# Создать утилиту
def require_stream(stream_id: str):
    if not stream_service.stream_exists(stream_id):
        raise HTTPException(404, f"Stream {stream_id} not found")
    return stream_service.get_stream(stream_id)

# Использовать везде
stream = require_stream(stream_id)
```

2. **Обработка ошибок БД:**
```python
# Создать декоратор
def handle_db_errors(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"DB error: {e}")
            raise HTTPException(500, "Database error")
    return wrapper

# Использовать
@handle_db_errors
async def get_acts(...):
    ...
```

3. **Валидация дат:**
```python
# Создать утилиту
def parse_date_range(start_date: Optional[str], end_date: Optional[str]):
    start = datetime.fromisoformat(start_date) if start_date else datetime.now() - timedelta(days=7)
    end = datetime.fromisoformat(end_date) if end_date else datetime.now()
    return start, end
```

---

### Шаг 6: Проверка и тестирование (30 мин)

**После каждого изменения:**

1. **Проверить размер файла:**
```bash
(Get-Content api/app.py | Measure-Object -Line).Lines
```

2. **Запустить сервер:**
```bash
python main.py
```

3. **Проверить endpoints:**
```bash
# Health check
curl http://localhost:8000/api/health

# Текущие метрики
curl http://localhost:8000/api/stats/current

# Список актов
curl http://localhost:8000/api/weighing/acts
```

4. **Проверить логи:**
- Нет ошибок при старте
- Сервисы инициализированы
- Endpoints отвечают корректно

---

## 📊 Контрольные точки

| Шаг | Целевой размер | Время |
|-----|----------------|-------|
| Начало | 3444 строки | - |
| После Шага 1 | 3450 строк | +30 мин |
| После Шага 2 | ~2800 строк | +2 часа |
| После Шага 3 | ~2200 строк | +1.5 часа |
| После Шага 4 | ~1800 строк | +1 час |
| После Шага 5 | ~1500 строк | +1 час |
| **ИТОГО** | **1500-2000 строк** | **~6 часов** |

---

## 🎯 Критерии успеха

✅ Размер `api/app.py` ≤ 2000 строк  
✅ Все endpoints работают корректно  
✅ Нет ошибок при запуске  
✅ Тесты проходят (если есть)  
✅ Логи чистые  
✅ Health check возвращает "healthy"  

---

## ⚠️ Частые ошибки

1. **Забыть инициализировать сервисы** - добавить в startup
2. **Circular imports** - импортировать сервисы после определения зависимостей
3. **None вместо сервиса** - проверять, что сервисы инициализированы
4. **Изменить сигнатуру без обновления вызовов** - искать все использования

---

## 💡 Советы

1. **Работать постепенно** - по одному endpoint'у за раз
2. **Коммитить часто** - после каждого успешного изменения
3. **Тестировать сразу** - не накапливать изменения
4. **Использовать поиск** - `rg`, `grep` для нахождения паттернов
5. **Измерять прогресс** - проверять размер файла после каждого шага

---

## 📝 Чеклист

- [ ] Шаг 1: Инициализация сервисов
- [ ] Шаг 2: Замена STREAM_MANAGER (2 часа)
- [ ] Шаг 3: Замена логики актов (1.5 часа)
- [ ] Шаг 4: Замена метрик (1 час)
- [ ] Шаг 5: Удаление дублирования (1 час)
- [ ] Шаг 6: Проверка и тестирование (30 мин)
- [ ] Финальная проверка размера (≤ 2000 строк)
- [ ] Обновить tasks.md (отметить Задачу 18 как завершённую)

---

**Автор:** Kiro AI  
**Дата:** 9 ноября 2025  
**Версия:** 1.0
