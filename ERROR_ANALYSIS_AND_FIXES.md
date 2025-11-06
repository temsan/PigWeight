# 🔴 АНАЛИЗ КРИТИЧЕСКИХ ОШИБОК И РЕШЕНИЯ

**Дата:** 2025-11-06 13:30:15  
**Статус:** АКТИВНОЕ ТЕСТИРОВАНИЕ НА СЕРВЕРЕ КЛИЕНТА

---

## 🚨 ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ

### Проблема #1: Model result has no masks

```
2025-11-06 13:30:15,014 - services.model_adapter - INFO - 
🎭 Model result has no masks - hasattr(r, 'masks'): True, r.masks: None
```

**Анализ:**
- YOLO v11 модель загружена и работает
- Модель **поддерживает маски** (hasattr = True)
- Но маски **не генерируются** (r.masks = None)

**Причины:**
1. ❌ Модель может быть загружена **в режиме detection-only** (без segmentation)
2. ❌ YOLO v11 может быть базовой версией без поддержки segmentation
3. ❌ Параметры инициализации модели могут отключать маски

**Решение:**
```python
# Правильная инициализация YOLO v11 с поддержкой segmentation
from ultralytics import YOLO

# ✅ ПРАВИЛЬНО - загружаем segmentation модель
model = YOLO('yolov11-seg.pt')  # -seg означает segmentation!

# ❌ НЕПРАВИЛЬНО - базовая модель без segmentation
model = YOLO('yolo11.pt')
```

**Статус кода:** `services/model_adapter.py:537` - маски установлены как пустой список `[]` (заглушка)

---

### Проблема #2: Function _req failed after 3 attempts: None

```
2025-11-06 13:30:15,187 - api.av_worker - ERROR - 
Function _req failed after 3 attempts: None

2025-11-06 13:30:15,505 - api.av_worker - ERROR - 
Function _req failed after 3 attempts: None
```

**Анализ:**
- av_worker (видео обработчик) **не отвечает** на запрос
- Ошибка **None** - это исключение без сообщения об ошибке
- **3 попытки** не помогли - сервис падает

**Места в коде:**
```python
# api/av_worker.py:381-419
def _req(self, cmd: str, payload: Dict[str, Any], timeout: float = 3.0):
    # ...
    if not ok:
        self._consecutive_failures += 1
        raise RuntimeError(str(data))  # ← data может быть None!
    # ...
```

**Проблемы:**
1. ❌ Когда `data = None`, `str(None)` = `'None'` (строка)
2. ❌ RuntimeError выбрасывается с текстом `'None'`
3. ❌ Декоратор `@retry_with_backoff` ловит её и логирует как `"Function _req failed: None"`
4. ❌ Работник процесс может быть **dead** или зависнуть

**Код воркера который отправляет ошибку:**
```python
# api/av_worker.py должен отправить (ok, data)
# Если что-то пошло не так:
conn.send((False, None))  # ← Вот откуда None!
```

---

### Проблема #3: Worker ping failed: None

```
2025-11-06 13:30:15,659 - api.av_worker - WARNING - 
Worker ping failed: None
```

**Анализ:**
- Проверка здоровья воркера не прошла
- `data = None` - нет ответа на ping

**Код:**
```python
# api/av_worker.py:342
logger.warning(f"Worker ping failed: {data}")  # data = None
```

**Цепь проблем:**
1. av_worker процесс **завис** или **упал**
2. Ping вернул `None` вместо результата
3. Система **не может перезапустить** процесс

---

## ✅ РЕШЕНИЕ ПРОБЛЕМ

### Решение #1: Проверить и зафиксировать модель YOLO

**Шаг 1: Проверить какая модель используется**

```bash
# Посмотреть config
cat models/pig_yolo11-seg.v4.pt

# Это должна быть -seg модель (с segmentation)!
# Если используется базовая - переменовать:
mv models/pig_yolo11.pt models/pig_yolo11-seg.pt.bak
```

**Шаг 2: Обновить код обработки масок**

```python
# services/model_adapter.py:502-540

def infer(self, imgs: List[np.ndarray]) -> List[Dict[str, Any]]:
    # ... 
    for result in results:
        # ✅ ИСПРАВЛЕНО: Проверяем наличие масок правильно
        if hasattr(result, 'masks') and result.masks is not None:
            logger.info(f"✅ Маски найдены: {len(result.masks)}")
            # Обработка масок
            masks = result.masks.data.cpu().numpy()
        else:
            logger.warning("⚠️ Маски недоступны, используем только bbox")
            masks = []
        
        # Используем bbox как fallback если нет масок
        if result.boxes:
            bboxes = result.boxes.xyxy.cpu().numpy()
        else:
            bboxes = []
        
        # Возвращаем то что есть
        frame_data = {
            'detections': len(bboxes),
            'confidence': float(result.boxes.conf.cpu().mean().numpy()) if result.boxes else 0.0,
            'masks': masks if masks else [],
            'bboxes': bboxes.tolist() if len(bboxes) > 0 else []
        }
```

---

### Решение #2: Исправить обработку ошибок в av_worker

**Шаг 1: Улучшить обработку None значений**

```python
# api/av_worker.py:381-419

def _req(self, cmd: str, payload: Dict[str, Any], timeout: float = 3.0):
    # ... health check ...
    
    # Send request
    t0 = time.time()
    try:
        self.conn.send((cmd, payload))
    except Exception as e:
        logger.error(f"❌ Failed to send {cmd}: {e}", exc_info=True)
        self._consecutive_failures += 1
        raise ConnectionError(f"Failed to send command {cmd}: {e}")
    
    # Wait for response
    while not self.conn.poll(0.05):
        if (time.time() - t0) > timeout:
            logger.error(f"⏱️ Timeout on {cmd} after {timeout}s")
            self._consecutive_failures += 1
            raise TimeoutError(f"av_worker timeout on {cmd} after {timeout}s")
    
    # Receive response
    try:
        response = self.conn.recv()
        if not isinstance(response, tuple) or len(response) != 2:
            logger.error(f"❌ Invalid response format: {type(response)}")
            self._consecutive_failures += 1
            raise RuntimeError(f"Invalid response from worker: {response}")
        
        ok, data = response
    except Exception as e:
        logger.error(f"❌ Failed to receive {cmd}: {e}", exc_info=True)
        self._consecutive_failures += 1
        raise ConnectionError(f"Failed to receive response for {cmd}: {e}")
    
    # Check if command succeeded
    if not ok:
        error_msg = str(data) if data is not None else "Unknown error"
        logger.error(f"❌ Command {cmd} failed: {error_msg}")
        self._consecutive_failures += 1
        raise RuntimeError(f"Worker error for {cmd}: {error_msg}")
    
    # Success!
    self._consecutive_failures = 0
    logger.debug(f"✅ {cmd} succeeded")
    return data
```

**Шаг 2: Улучшить декоратор retry**

```python
# api/av_worker.py:94-140

@retry_with_backoff(max_retries=MAX_RETRIES, ...)
def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    delay = min(base_delay * (backoff_multiplier ** attempt) + 
                               random.uniform(0, 0.1), max_delay)
                    logger.warning(
                        f"⚠️ Function {func.__name__} failed "
                        f"(attempt {attempt + 1}/{max_retries + 1}): "
                        f"{type(e).__name__}: {str(e) or 'No error message'}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"❌ Function {func.__name__} failed after {max_retries + 1} attempts. "
                        f"Last error: {type(e).__name__}: {str(e) or 'No error message'}",
                        exc_info=True
                    )
        
        raise last_exception
    
    return wrapper
```

---

### Решение #3: Добавить health check для av_worker

**Шаг 1: Улучшить проверку здоровья воркера**

```python
# api/av_worker.py:300-360

def _check_worker_health(self) -> bool:
    """
    ✅ Проверить здоровье av_worker процесса
    """
    try:
        # Проверка 1: Жив ли процесс?
        if not self.proc.is_alive():
            logger.error("❌ Worker process is dead!")
            return False
        
        # Проверка 2: Может ли процесс обработать запрос?
        if not self.conn.poll(0.5):
            # Ничего не пришло за 0.5s
            try:
                # Пытаемся отправить ping
                self.conn.send(('ping', {}))
            except Exception as e:
                logger.error(f"❌ Cannot send ping: {e}")
                return False
            
            # Ждём ответ на ping
            if not self.conn.poll(2.0):  # 2 секунды
                logger.error("❌ No response to ping")
                return False
            
            try:
                ok, data = self.conn.recv()
                if not ok:
                    logger.error(f"❌ Ping failed: {data}")
                    return False
                logger.debug("✅ Worker ping succeeded")
                return True
            except Exception as e:
                logger.error(f"❌ Cannot receive ping response: {e}")
                return False
        else:
            # Что-то осталось в буфере от предыдущего запроса
            try:
                stale = self.conn.recv()
                logger.warning(f"⚠️ Stale data in buffer: {stale}")
            except:
                pass
            return False
    
    except Exception as e:
        logger.error(f"❌ Health check error: {e}", exc_info=True)
        return False
```

---

## 🔧 ПЛАН ДЕЙСТВИЙ

### Немедленно (Critical):

1. **Проверить модель YOLO**
   ```bash
   # Убедиться что используется -seg модель
   ls -la models/pig_yolo*
   ```

2. **Обновить av_worker error handling** (api/av_worker.py)
   ```python
   # Заменить неправильную обработку None на правильную
   ```

3. **Перезапустить тестирование**
   ```bash
   python console_app.py --video test_cam.mp4
   ```

### На этой неделе:

4. Улучшить логирование с детальными сообщениями об ошибках
5. Добавить graceful shutdown при падении воркера
6. Написать smoke-тесты для проверки всех компонентов

### На следующую неделю:

7. Интеграция с реальными IP весами
8. Calibration параметров под видеопотоки камер
9. Load testing на полном наборе камер

---

## 📊 СТАТУС КОМПОНЕНТОВ

| Компонент | Статус | Проблема | Fix |
|-----------|--------|---------|-----|
| **YOLO Model** | ⚠️ Частично | Нет масок | Проверить тип модели (-seg) |
| **av_worker** | 🔴 Broken | Timeout/None errors | Улучшить error handling |
| **Pipe Communication** | ⚠️ Нестабильно | Стейл данные в буфере | Добавить flush |
| **Health Check** | ❌ Неполная | Неправильный ping | Переписать |
| **Retry Logic** | ⚠️ Работает | Плохие логи | Улучшить messages |

---

## 🎯 ОЖИДАЕМЫЙ РЕЗУЛЬТАТ

**После применения fixes:**

✅ Модель будет возвращать маски (или fallback на bbox)  
✅ av_worker будет обрабатывать ошибки gracefully  
✅ Health check будет работать корректно  
✅ Логи будут понятными (вместо "None")  
✅ Система будет автоматически перезапускать упавшие компоненты  

---

## 📝 ПРИМЕЧАНИЯ

- **Маски нужны для:** точной сегментации свиней (более точно чем bbox)
- **Если нет -seg модели:** система может работать с bbox (bounding boxes)
- **av_worker critical:** если упадёт воркер, вся обработка видео остановится
- **Тестирование на сервере:** нужно смотреть реальные видеопотоки с камер (cam101, cam102)

---

**Версия:** 1.0  
**Дата:** 2025-11-06 13:30  
**Автор:** Error Analysis System

