# 📚 PigWeight API - Полная документация

**Версия:** 1.1.0  
**Последнее обновление:** 2025-11-03

---

## 🎯 Оглавление

1. [Общее описание](#общее-описание)
2. [Диагностика (Debug API)](#диагностика)
3. [Обработка видео (Queue API)](#обработка-видео)
4. [Управление камерами](#управление-камерами)
5. [Записи актов](#записи-актов)
6. [WebSocket подключения](#websocket)
7. [Примеры использования](#примеры)
8. [Решение проблем](#решение-проблем)

---

## Общее описание

### Базовый URL
```
http://localhost:8000
```

### Аутентификация
Текущая версия не требует аутентификации. В production рекомендуется добавить API keys.

### Формат ответов
Все эндпоинты возвращают JSON:
```json
{
  "status": "success|error",
  "timestamp": "2025-11-03T12:00:00",
  "data": {}
}
```

---

## Диагностика

### GET /debug/health
Проверка здоровья всей системы

**Ответ:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-03T12:00:00",
  "server": {
    "host": "localhost",
    "port": 8000,
    "debug": true
  },
  "components": {
    "unified_processor": true,
    "event_logger": true
  },
  "rtsp_diagnostics": {
    "connection_attempts": 5,
    "successful_connections": 5,
    "failed_connections": 0,
    "timeouts": 0
  }
}
```

### GET /debug/rtsp
Подробная диагностика RTSP подключений

**Ответ:**
```json
{
  "rtsp_status": "diagnostic",
  "diagnostics": {
    "stages": {
      "rtsp_open": {
        "status": "success",
        "duration_ms": 250.5
      },
      "frame_read": {
        "status": "success",
        "duration_ms": 33.2
      }
    }
  },
  "summary": {
    "success_rate": 100.0
  }
}
```

### GET /debug/infer_status
Статус инференса и нагрузка системы

**Ответ:**
```json
{
  "inference_status": "running",
  "process_info": {
    "pid": 12345,
    "memory_mb": 2048.5,
    "cpu_percent": 45.2,
    "num_threads": 8
  },
  "system_info": {
    "cpu_percent": 60.0,
    "memory_percent": 55.5,
    "available_memory_gb": 4.5
  }
}
```

### POST /debug/test_rtsp/{camera_id}
Тест подключения к конкретной камере

**Параметры:**
- `camera_id` (string): ID камеры (например, "cam101")

**Ответ (успех):**
```json
{
  "status": "success",
  "camera_id": "cam101",
  "connection_time_sec": 0.85,
  "timestamp": "2025-11-03T12:00:00"
}
```

**Ответ (таймаут):**
```json
{
  "status": "timeout",
  "camera_id": "cam101",
  "timeout_sec": 5.0,
  "timestamp": "2025-11-03T12:00:00"
}
```

---

## Обработка видео

### POST /api/processing/queue/add
Добавить видео в очередь обработки

**Запрос:**
```json
{
  "video_path": "uploads/test_video.mp4"
}
```

**Ответ:**
```json
{
  "status": "success",
  "task_id": "task_a1b2c3d4",
  "video_path": "uploads/test_video.mp4",
  "message": "Video added to processing queue"
}
```

### GET /api/processing/queue/tasks
Получить список всех задач

**Параметры (опциональные):**
- `status`: фильтр по статусу (pending, processing, completed, failed)

**Ответ:**
```json
{
  "status": "success",
  "tasks": [
    {
      "task_id": "task_a1b2c3d4",
      "video_path": "uploads/test_video.mp4",
      "status": "completed",
      "progress": 100.0,
      "created_at": 1699005600,
      "started_at": 1699005605,
      "completed_at": 1699005720,
      "result": {
        "frames_processed": 2150,
        "acts_detected": 5
      }
    }
  ],
  "count": 1
}
```

### GET /api/processing/queue/task/{task_id}
Получить информацию о конкретной задаче

**Параметры:**
- `task_id` (string): ID задачи

**Ответ:**
```json
{
  "status": "success",
  "task": {
    "task_id": "task_a1b2c3d4",
    "video_path": "uploads/test_video.mp4",
    "status": "processing",
    "progress": 45.5
  }
}
```

### GET /api/processing/queue/stats
Статистика очереди обработки

**Ответ:**
```json
{
  "status": "success",
  "stats": {
    "total_tasks": 10,
    "pending": 2,
    "processing": 1,
    "completed": 6,
    "failed": 1,
    "success_rate": 0.857
  }
}
```

---

## Управление камерами

### GET /api/cameras
Получить список доступных камер

**Ответ:**
```json
{
  "cam101": {
    "name": "Камера 101",
    "url": "rtsp://192.168.1.10:554/stream"
  },
  "cam102": {
    "name": "Камера 102",
    "url": "rtsp://192.168.1.11:554/stream"
  }
}
```

---

## Записи актов

### GET /api/records
Получить список всех актов взвешивания

**Ответ:**
```json
{
  "items": [
    {
      "id": "act_file_1234567890_20251103-120000",
      "date": "2025-11-03",
      "time": "12:00:00",
      "filename": "act_file_1234567890_20251103-120000.json"
    }
  ],
  "count": 15
}
```

### GET /api/records/{act_name}
Получить детали конкретного акта

**Параметры:**
- `act_name` (string): имя файла акта

**Ответ:**
```json
{
  "act_id": 1,
  "started_at": 1699005600.0,
  "ended_at": 1699005720.0,
  "duration_sec": 120.0,
  "left_count": 15,
  "right_count": 14,
  "peak_count": 8,
  "crossings": [
    {
      "timestamp": 1699005610.0,
      "track_id": 42,
      "side": "left",
      "x": 0.25,
      "y": 0.5,
      "weight_estimate": 120.5
    }
  ]
}
```

---

## WebSocket

### WS /ws/processing/progress
Подписка на обновления прогресса обработки

**Параметры:**
- `task_id` (string): ID задачи для отслеживания

**События:**

**status** (получается при подключении):
```json
{
  "event": "status",
  "task": {
    "task_id": "task_a1b2c3d4",
    "status": "processing",
    "progress": 25.0
  }
}
```

**progress** (периодические обновления):
```json
{
  "event": "progress",
  "task": {
    "progress": 45.5,
    "status": "processing"
  }
}
```

**complete** (успешное завершение):
```json
{
  "event": "complete",
  "task": {
    "status": "completed",
    "result": {
      "frames_processed": 2150,
      "acts_detected": 5
    }
  }
}
```

**error** (ошибка обработки):
```json
{
  "event": "error",
  "task": {
    "status": "failed",
    "error": "Video file not found or corrupted"
  }
}
```

---

## Примеры использования

### cURL

#### Добавить видео в очередь
```bash
curl -X POST http://localhost:8000/api/processing/queue/add \
  -H "Content-Type: application/json" \
  -d '{"video_path": "uploads/test.mp4"}'
```

#### Проверить статус очереди
```bash
curl http://localhost:8000/api/processing/queue/stats
```

#### Тестировать RTSP камеру
```bash
curl -X POST http://localhost:8000/debug/test_rtsp/cam101
```

### Python

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Добавить видео
response = requests.post(
    f"{BASE_URL}/api/processing/queue/add",
    json={"video_path": "uploads/test.mp4"}
)
task_id = response.json()["task_id"]

# Проверить прогресс
response = requests.get(f"{BASE_URL}/api/processing/queue/task/{task_id}")
task_info = response.json()["task"]
print(f"Progress: {task_info['progress']}%")

# Получить статистику
response = requests.get(f"{BASE_URL}/api/processing/queue/stats")
stats = response.json()["stats"]
print(f"Completed: {stats['completed']}/{stats['total_tasks']}")
```

### JavaScript

```javascript
// Добавить видео
const taskResponse = await fetch('/api/processing/queue/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: 'uploads/test.mp4' })
});
const { task_id } = await taskResponse.json();

// WebSocket для прогресса
const ws = new WebSocket(`ws://localhost:8000/ws/processing/progress?task_id=${task_id}`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Event: ${data.event}, Progress: ${data.task.progress}%`);
};
```

---

## Решение проблем

### RTSP подключение не работает

**Диагностика:**
```bash
curl http://localhost:8000/debug/rtsp
```

**Решение:**
1. Проверьте доступность камеры: `curl http://localhost:8000/debug/test_rtsp/cam101`
2. Проверьте URL камеры в `.env`
3. Проверьте сетевое подключение и firewall

### Видео не обрабатывается

**Диагностика:**
```bash
curl http://localhost:8000/api/processing/queue/tasks?status=failed
```

**Решение:**
1. Проверьте путь к видеофайлу
2. Проверьте формат видео (mp4, avi, mov поддерживаются)
3. Проверьте лог в `logs/app.log`

### Низкая производительность

**Диагностика:**
```bash
curl http://localhost:8000/debug/infer_status
```

**Рекомендации:**
1. Увеличьте параметры в `.env`:
   - `FRAME_SKIP=2` (пропускать каждый второй кадр)
   - `USE_HALF=true` (использовать FP16)
2. Уменьшите `TARGET_FPS` для менее нагруженной обработки
3. Проверьте использование GPU

---

## Коды ошибок

| Код | Описание |
|-----|----------|
| 200 | OK - запрос успешен |
| 400 | Bad Request - неверные параметры |
| 404 | Not Found - ресурс не найден |
| 500 | Server Error - ошибка сервера |
| 504 | Gateway Timeout - таймаут подключения |

---

## Рекомендации

### Production-ready checklist

- [ ] Включить HTTPS
- [ ] Добавить аутентификацию (API keys)
- [ ] Настроить rate limiting
- [ ] Включить CORS политику
- [ ] Добавить логирование всех запросов
- [ ] Настроить мониторинг (Prometheus/Grafana)
- [ ] Настроить backup для результатов
- [ ] Протестировать нагрузку (load testing)

---

**Последнее обновление:** 2025-11-03  
**Автор:** Kiro AI Assistant
