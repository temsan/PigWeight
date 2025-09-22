# Улучшения надежности av_worker

## Обзор

av_worker был значительно улучшен для повышения надежности и устойчивости к сбоям. Добавлены механизмы retry с экспоненциальным backoff, проверки здоровья и автоматическое восстановление соединений.

## Ключевые улучшения

### 1. Retry механизм с экспоненциальным backoff

```python
@retry_with_backoff(max_retries=3, base_delay=0.1, max_delay=2.0)
def operation():
    # Операция с автоматическим retry при сбоях
    pass
```

**Особенности:**
- Экспоненциальное увеличение задержки между попытками
- Добавление jitter (до 10%) для предотвращения thundering herd
- Различные стратегии для разных типов операций
- Логирование всех попыток и неудач

### 2. Проверки здоровья worker процесса

```python
# Автоматическая проверка каждые 30 секунд
health_stats = av_worker.get_health_stats()
ping_result = av_worker.ping()
```

**Возможности:**
- Периодическая проверка жизнеспособности процесса
- Ping-команда для быстрой проверки связи
- Отслеживание количества последовательных сбоев
- Автоматический перезапуск при критических сбоях

### 3. Автоматическое восстановление соединений

```python
# Автоматический перезапуск при превышении лимита сбоев
if consecutive_failures >= max_consecutive_failures:
    restart_worker()
```

**Механизмы:**
- Graceful termination старого процесса
- Создание нового worker процесса с теми же параметрами
- Сброс счетчиков сбоев после успешного восстановления
- Сохранение конфигурации для восстановления

## Конфигурация retry

### Глобальные константы

```python
MAX_RETRIES = 3           # Максимальное количество попыток
BASE_DELAY = 0.1          # Базовая задержка в секундах
MAX_DELAY = 5.0           # Максимальная задержка
BACKOFF_MULTIPLIER = 2.0  # Множитель для экспоненциального роста
```

### Специализированные декораторы

```python
@retry_with_backoff(max_retries=2, base_delay=0.2, max_delay=3.0)
def open_file(self, sid: str, path: str):
    # Операции открытия файлов с умеренным retry
    pass

@health_check_retry  # max_retries=5, более агрессивный retry
def ping(self):
    # Health check операции с быстрым retry
    pass
```

## Мониторинг и диагностика

### Health Stats

```python
health_stats = av_worker.get_health_stats()
# Возвращает:
{
    'process_alive': True,
    'consecutive_failures': 0,
    'last_health_check': 1758545880.67,
    'max_consecutive_failures': 3,
    'health_check_interval': 30.0
}
```

### Логирование

Система предоставляет подробное логирование:

```
WARNING - Function open_file failed (attempt 1/3): av_worker timeout on open_file. Retrying in 0.12s...
INFO - av_worker process restarted successfully
ERROR - Function _req failed after 4 attempts: Connection lost
```

## Типы ошибок и обработка

### Retryable ошибки
- `TimeoutError` - таймауты операций
- `ConnectionError` - проблемы соединения
- `RuntimeError` - ошибки worker процесса

### Non-retryable ошибки
- `ValueError` - неверные параметры
- `FileNotFoundError` - отсутствующие файлы
- Другие исключения, не связанные с временными сбоями

## Производительность

### Оптимизации
- Кэширование результатов health check
- Адаптивные интервалы проверки
- Минимальные задержки для быстрых операций
- Jitter для предотвращения синхронных retry

### Метрики
- Отслеживание времени восстановления
- Статистика успешности операций
- Мониторинг частоты перезапусков

## Использование

### Базовое использование
```python
from api.av_worker import AVIsolate

# Создание с автоматическим retry
av_worker = AVIsolate(jpeg_quality=80, target_fps=12.0)

# Операции автоматически используют retry
result = av_worker.open_file("stream1", "/path/to/video.mp4")
```

### Мониторинг здоровья
```python
# Проверка состояния
health = av_worker.get_health_stats()
if health['consecutive_failures'] > 0:
    logger.warning(f"Worker has {health['consecutive_failures']} consecutive failures")

# Ручная проверка связи
try:
    ping_result = av_worker.ping()
    logger.info(f"Worker is alive: {ping_result}")
except Exception as e:
    logger.error(f"Worker ping failed: {e}")
```

## Исправленные проблемы

### ✅ TARGET_FPS undefined
- Исправлена ошибка `name 'TARGET_FPS' is not defined`
- Используется `config.TARGET_FPS` вместо неопределенной переменной

### ✅ av_worker timeout
- Добавлен retry механизм для операций с таймаутами
- Автоматическое восстановление при критических сбоях

### ✅ Connection reliability
- Проверки здоровья соединения
- Автоматический перезапуск при потере связи

## Совместимость

Все изменения обратно совместимы. Существующий код продолжит работать без изменений, но получит дополнительную надежность автоматически.