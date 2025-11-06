# 🛠️ ГАЙД ПО РЕАЛИЗАЦИИ

**Дата:** 7 ноября 2025  
**Для:** Разработчиков и AI агентов  
**Статус:** Готов к использованию

---

## 🎯 КРИТИЧЕСКИЕ ЗАДАЧИ

### ЗАДАЧА 11: API Standardization (~3-4 часа)

#### Шаг 1: Переименовать существующие endpoints

**Файл:** `api/endpoints/metrics.py`

```python
# БЫЛО:
@app.get("/api/metrics/current")
async def get_current_metrics():
    return STREAM_MANAGER.get_stats()

# СТАЛО:
@app.get("/api/stats/current")
async def get_current_stats():
    from pig_tracking.database_manager import DatabaseManager
    db = DatabaseManager(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_KEY")
    )
    return db.get_stats_summary()

# Добавить редирект для обратной совместимости
@app.get("/api/metrics/current")
async def get_current_metrics_legacy():
    return await get_current_stats()
```

#### Шаг 2: Создать новые endpoints

**Файл:** `api/endpoints/weighing.py` (создать новый)

```python
from fastapi import APIRouter, Query, UploadFile, File
from datetime import datetime
import os

from pig_tracking.database_manager import DatabaseManager
from pig_tracking.excel_exporter import ExcelExporter
from pig_tracking.excel_comparator import ExcelComparator

router = APIRouter(prefix="/api/weighing", tags=["weighing"])

# Глобальный экземпляр БД
db = DatabaseManager(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

@router.get("/acts")
async def get_weighing_acts(
    start_date: str = Query(...),
    end_date: str = Query(...),
    stream_id: str = Query(None)
):
    """Получить список актов взвешивания за период"""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    acts = db.get_acts_by_period(start, end)
    
    if stream_id:
        acts = [a for a in acts if a.stream_id == stream_id]
    
    return {"acts": [a.to_dict() for a in acts]}

@router.get("/stats")
async def get_weighing_stats(
    start_date: str = Query(None),
    end_date: str = Query(None)
):
    """Получить агрегированную статистику"""
    if start_date and end_date:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        acts = db.get_acts_by_period(start, end)
    else:
        # Последние 7 дней
        acts = db.get_recent_acts(days=7)
    
    total_acts = len(acts)
    total_crossings = sum(a.left_count + a.right_count for a in acts)
    avg_weight = sum(a.avg_weight or 0 for a in acts) / total_acts if total_acts > 0 else 0
    
    return {
        "total_acts": total_acts,
        "total_crossings": total_crossings,
        "avg_weight": round(avg_weight, 1),
        "period": {
            "start": acts[0].started_at if acts else None,
            "end": acts[-1].ended_at if acts else None
        }
    }
```

#### Шаг 3: Создать endpoints для экспорта

**Файл:** `api/endpoints/export.py` (создать новый)

```python
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from datetime import datetime
from pathlib import Path
import os

from pig_tracking.database_manager import DatabaseManager
from pig_tracking.excel_exporter import ExcelExporter

router = APIRouter(prefix="/api/export", tags=["export"])

db = DatabaseManager(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

@router.post("/excel")
async def export_to_excel(
    start_date: str = Query(...),
    end_date: str = Query(...),
    stream_id: str = Query(None)
):
    """Экспорт актов в Excel"""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    # Получить акты
    acts = db.get_acts_by_period(start, end)
    if stream_id:
        acts = [a for a in acts if a.stream_id == stream_id]
    
    # Экспорт
    exporter = ExcelExporter()
    output_path = Path("temp") / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_path.parent.mkdir(exist_ok=True)
    
    exporter.export_to_excel(acts, str(output_path))
    
    return FileResponse(
        path=str(output_path),
        filename=f"weighing_acts_{start_date}_{end_date}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
```

#### Шаг 4: Зарегистрировать новые роутеры

**Файл:** `api/app.py`

```python
# Добавить импорты
from api.endpoints import weighing, export

# Зарегистрировать роутеры
app.include_router(weighing.router)
app.include_router(export.router)
```

#### Шаг 5: Обновить frontend

**Файл:** `static/mobile-dashboard.html`

```javascript
// БЫЛО:
const response = await fetch('/api/metrics/current');

// СТАЛО:
const response = await fetch('/api/stats/current');
```

---

### ЗАДАЧА 2: Интеграция API с БД (~2-3 часа)

#### Шаг 1: Добавить DatabaseManager в зависимости

**Файл:** `api/app.py`

```python
from pig_tracking.database_manager import DatabaseManager
import os

# Создать глобальный экземпляр
DB_MANAGER = DatabaseManager(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)
```

#### Шаг 2: Заменить STREAM_MANAGER на DB_MANAGER

**Файл:** `api/app.py` - в обработчике WebSocket

```python
# БЫЛО:
@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    # ...
    STREAM_MANAGER.update_stats(stream_id, stats)

# СТАЛО:
@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    # ...
    # Сохранять в БД при завершении акта
    if completed_act:
        DB_MANAGER.save_weighing_act(completed_act)
    
    # Сохранять пересечения
    for crossing in crossing_events:
        DB_MANAGER.save_crossing(crossing)
```

#### Шаг 3: Сохранять STREAM_MANAGER только для real-time

```python
# STREAM_MANAGER - только для WebSocket broadcast
# DB_MANAGER - для персистентного хранения

# При обработке кадра:
# 1. Обновить STREAM_MANAGER (для WebSocket)
STREAM_MANAGER.update_stats(stream_id, stats)

# 2. Сохранить в БД (для истории)
if completed_act:
    DB_MANAGER.save_weighing_act(completed_act)
```

---

## 🟡 ВЫСОКИЙ ПРИОРИТЕТ

### ЗАДАЧА 9: WebSocket оптимизация (~2 часа)

**Файл:** `api/app.py`

```python
import time
from collections import deque

class WebSocketThrottler:
    def __init__(self, max_fps: int = 10, max_clients: int = 5):
        self.max_fps = max_fps
        self.max_clients = max_clients
        self.min_interval = 1.0 / max_fps
        self.last_send_time = {}
        self.active_clients = set()
    
    def can_send(self, client_id: str) -> bool:
        current_time = time.time()
        last_time = self.last_send_time.get(client_id, 0)
        
        if current_time - last_time >= self.min_interval:
            self.last_send_time[client_id] = current_time
            return True
        return False
    
    def can_connect(self) -> bool:
        return len(self.active_clients) < self.max_clients
    
    def add_client(self, client_id: str):
        self.active_clients.add(client_id)
    
    def remove_client(self, client_id: str):
        self.active_clients.discard(client_id)
        self.last_send_time.pop(client_id, None)

# Использование
throttler = WebSocketThrottler(max_fps=10, max_clients=5)

@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    client_id = f"{stream_id}_{id(websocket)}"
    
    # Проверить лимит клиентов
    if not throttler.can_connect():
        await websocket.close(code=1008, reason="Too many clients")
        return
    
    await websocket.accept()
    throttler.add_client(client_id)
    
    try:
        while True:
            # Получить данные
            data = await get_frame_data(stream_id)
            
            # Throttling
            if throttler.can_send(client_id):
                await websocket.send_json(data)
            
            await asyncio.sleep(0.01)  # 100 FPS loop, но отправка ≤10 FPS
    finally:
        throttler.remove_client(client_id)
```

---

### ЗАДАЧА 10: av_worker устойчивость (~2 часа)

**Файл:** `api/av_worker.py`

```python
import time
import random

class RetryConfig:
    MAX_RETRIES = 3
    BASE_DELAY = 0.1
    MAX_DELAY = 5.0
    BACKOFF_MULTIPLIER = 2.0

def retry_with_backoff(func):
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(RetryConfig.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (TimeoutError, ConnectionError) as e:
                last_exception = e
                
                if attempt == RetryConfig.MAX_RETRIES - 1:
                    raise e
                
                # Exponential backoff with jitter
                delay = min(
                    RetryConfig.BASE_DELAY * (RetryConfig.BACKOFF_MULTIPLIER ** attempt),
                    RetryConfig.MAX_DELAY
                )
                jitter = random.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)
        
        raise last_exception
    return wrapper

class AVIsolate:
    def __init__(self, jpeg_quality: int = 80, target_fps: float = 12.0):
        # ...
        self._health_check_interval = 30.0
        self._last_health_check = time.time()
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3
    
    @retry_with_backoff
    def open_rtsp(self, sid: str, url: str, timeout: float = 10.0):
        return self._req('open_rtsp', {'id': sid, 'url': url}, timeout=timeout)
    
    @retry_with_backoff
    def read_jpeg(self, sid: str, timeout: float = 2.0):
        return self._req('read_jpeg', {'id': sid}, timeout=timeout)
    
    def _check_worker_health(self) -> bool:
        """Проверка здоровья worker процесса"""
        try:
            if not self.proc.is_alive():
                return False
            
            # Ping
            self.conn.send(('ping', {}))
            if not self.conn.poll(1.0):
                return False
            
            ok, data = self.conn.recv()
            return ok
        except Exception:
            return False
    
    def _restart_worker(self):
        """Перезапуск worker процесса"""
        if hasattr(self, 'proc') and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5.0)
        
        # Создать новый процесс
        parent_conn, child_conn = mp.Pipe()
        self.conn = parent_conn
        self.proc = _Worker(child_conn, self._jpeg_quality, self._target_fps)
        self.proc.start()
        
        self._consecutive_failures = 0
```

---

## 📝 ЧЕКЛИСТ РЕАЛИЗАЦИИ

### Задача 11: API Standardization
- [ ] Переименовать `/api/metrics/current` → `/api/stats/current`
- [ ] Создать `api/endpoints/weighing.py`
- [ ] Создать `api/endpoints/export.py`
- [ ] Зарегистрировать роутеры в `api/app.py`
- [ ] Обновить frontend (`static/mobile-dashboard.html`)
- [ ] Добавить редиректы для обратной совместимости
- [ ] Протестировать все endpoints

### Задача 2: Интеграция с БД
- [ ] Добавить `DB_MANAGER` в `api/app.py`
- [ ] Заменить `STREAM_MANAGER` на `DB_MANAGER` в WebSocket
- [ ] Сохранять акты в БД при завершении
- [ ] Сохранять пересечения в БД
- [ ] Оставить `STREAM_MANAGER` только для real-time
- [ ] Протестировать персистентность данных

### Задача 9: WebSocket оптимизация
- [ ] Создать `WebSocketThrottler` класс
- [ ] Добавить throttling в WebSocket endpoint
- [ ] Добавить лимит клиентов (макс 5)
- [ ] Добавить мониторинг метрик
- [ ] Протестировать под нагрузкой

### Задача 10: av_worker устойчивость
- [ ] Добавить `retry_with_backoff` декоратор
- [ ] Добавить таймауты для всех операций
- [ ] Реализовать `_check_worker_health()`
- [ ] Реализовать `_restart_worker()`
- [ ] Протестировать на нестабильном RTSP

---

## 🧪 ТЕСТИРОВАНИЕ

### После каждой задачи:

```bash
# 1. Перезапустить сервер
pkill -f "python main.py"
python main.py

# 2. Проверить endpoints
curl http://localhost:8000/api/stats/current
curl http://localhost:8000/api/weighing/acts?start_date=2025-11-01&end_date=2025-11-07

# 3. Проверить WebSocket
# Открыть http://localhost:8000/mobile в браузере

# 4. Проверить БД
python -c "
from pig_tracking.database_manager import DatabaseManager
import os
db = DatabaseManager(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
print(db.get_stats_summary())
"
```

---

**Версия:** 1.0  
**Обновлено:** 7 ноября 2025  
**Статус:** ✅ Готов к использованию
