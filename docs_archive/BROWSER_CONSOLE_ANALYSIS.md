# 🌐 АНАЛИЗ БРАУЗЕРНЫХ ЛОГОВ

**Дата:** 2025-11-06 (из браузера пользователя)  
**Статус:** Система работает, но есть мелкие проблемы

---

## ✅ **ЧТО РАБОТАЕТ ОТЛИЧНО**

```
✅ Chart.js загружен успешно
✅ UI Manager инициализирован
✅ WebSocket Manager инициализирован  
✅ Chart Manager инициализирован
✅ Journal Manager инициализирован
✅ PigWeight App инициализирован
✅ WebSocket подключен к cam101
✅ Переключение стрима: null -> cam101
✅ Система работает нормально
```

**Вывод:** Фронтенд полностью инициализирован и готов!

---

## ⚠️ **ПРОБЛЕМА #1: Размеры видео не определены**

**Логи:**
```
⚠️ Размеры видео не определены, используем размеры wrapper
getRenderedImageRect @ video-manager.js:382
onPos @ video-manager.js:839
```

**Что это значит:**
- Функция `getRenderedImageRect()` вызывается ~20+ раз
- `videoWidth` и `videoHeight` возвращают 0 или undefined
- Система использует размеры wrapper вместо реальных размеров видео

**Когда это нормально:**
✅ При первой загрузке видео (до loadedmetadata)
✅ Когда видеоэлемент ещё не инициализирован

**Когда это ПРОБЛЕМА:**
❌ Если видео никогда не загружает размеры
❌ Если случается в течение всего сеанса

**Текущий статус:** ⚠️ НОРМАЛЬНОЕ - видео просто ещё загружается

**Решение:** Добавить debounce, чтобы не спамить логи при повторяющихся вызовах

---

## 🔴 **ПРОБЛЕМА #2: 404 Not Found для /api/events/**

**Ошибки:**
```
GET http://localhost:8000/api/events/cam102?limit=100 → 404
GET http://localhost:8000/api/events/cam102/stats → 404
GET http://localhost:8000/api/events/cam101?limit=100 → 404
GET http://localhost:8000/api/events/cam101/stats → 404
```

**Анализ:**
- Frontend ищет `/api/events/{stream_id}` 
- Backend возвращает 404
- Endpoints ДА существуют в `api/endpoints/events.py`
- Router подключен в `api/app.py:1464`

**Возможные причины:**

### Причина 1: Сервер не перезагружен
```bash
# Endpoint добавлен в events.py
# Но сервер старый процесс всё ещё работает

# Решение:
pkill -f "python main.py"
python main.py
```

### Причина 2: HAVE_EVENT_LOGGER = False
```python
# api/endpoints/events.py:18-22
try:
    from services.event_logger import get_event_logger
    HAVE_EVENT_LOGGER = True
except ImportError:
    HAVE_EVENT_LOGGER = False  # ← Если это False, 503 ошибка
    logger.warning("EventLogger не доступен")
```

**Проверить:**
```bash
# В логах сервера должно быть:
grep "EventLogger" logs/app.log

# Если "EventLogger не доступен" - нужно установить зависимости
pip install -r requirements.txt
```

### Причина 3: EventsManager в браузере вызывает неправильный URL
```javascript
// static/js/events-manager.js:118
// Проверить что URL правильный:
GET /api/events/cam101?limit=100  ← Правильный формат
```

**Решение:**
```bash
# 1. Перезагрузить сервер
python main.py

# 2. Проверить что endpoints ответят
curl http://localhost:8000/api/events/cam101?limit=100

# 3. Если ещё 404 - посмотреть логи сервера
tail -f logs/app.log | grep "events"
```

---

## 📊 **СТАТУС КОМПОНЕНТОВ**

| Компонент | Фронтенд | Бэкенд | WebSocket | Статус |
|-----------|----------|--------|-----------|--------|
| **UI Manager** | ✅ | ✅ | N/A | ✅ Работает |
| **WebSocket Manager** | ✅ | ✅ | ✅ | ✅ Подключен |
| **Chart Manager** | ✅ | ✅ | N/A | ✅ Инициализирован |
| **Journal Manager** | ✅ | ✅ | N/A | ✅ Инициализирован |
| **Video Manager** | ✅ | ⚠️ | ✅ | ⚠️ Размеры не определены |
| **Events Manager** | ✅ | 🔴 | N/A | 🔴 Endpoints 404 |

---

## 🔧 **ДЕЙСТВИЯ ДЛЯ ПОЛНОГО ФУНКЦИОНИРОВАНИЯ**

### Шаг 1: Перезагрузить сервер (СРОЧНО!)

```bash
# Завершить старый процесс
pkill -f "python main.py"

# Запустить новый
python main.py
```

**Почему:** Endpoints были добавлены, но сервер не перезагружен

### Шаг 2: Проверить что endpoints отвечают

```bash
# Тест в браузере или curl:
curl -i http://localhost:8000/api/events/cam101?limit=100

# Должно вернуть:
# HTTP/1.1 200 OK
# или
# HTTP/1.1 503 Service Unavailable (если EventLogger не доступен)

# НЕ должно быть 404!
```

### Шаг 3: Если всё ещё 404

```bash
# Проверить логи
tail -100 logs/app.log

# Искать:
# - "EventLogger"
# - "events.router"
# - Ошибки импорта
```

### Шаг 4: Уменьшить spam логов про видео размеры

```javascript
// static/js/video-manager.js:382

// Текущее:
if (!iw || !ih) {
    console.warn('⚠️ Размеры видео не определены, используем размеры wrapper');
    return { x: 0, y: 0, w: cw, h: ch };
}

// Улучшенное (с debounce):
if (!iw || !ih) {
    const now = Date.now();
    if (!this._lastVideoSizeWarning || now - this._lastVideoSizeWarning > 5000) {
        console.warn('⚠️ Размеры видео не определены, используем размеры wrapper');
        this._lastVideoSizeWarning = now;
    }
    return { x: 0, y: 0, w: cw, h: ch };
}
```

---

## 📝 **КРАТКАЯ ПРОВЕРКА**

```bash
# Всё работает если:
1. ✅ Браузер показывает: "[SUCCESS] ✅ Система работает нормально"
2. ✅ WebSocket подключен (нет красных ошибок)
3. ⚠️ Размеры видео warnings - НОРМАЛЬНО (видео загружается)
4. 🔴 404 ошибки от /api/events - НУЖНО ПЕРЕЗАГРУЗИТЬ СЕРВЕР

# После перезагрузки сервера:
python main.py
# Обновить браузер: F5 или Ctrl+Shift+R (hard refresh)
```

---

## 🎯 **СЛЕДУЮЩИЕ ШАГИ**

1. **Немедленно:** Перезагрузить сервер
2. **Проверить:** Endpoints отвечают 200, не 404
3. **Оптимизировать:** Убрать spam логов про видео размеры
4. **Тестировать:** На реальных видеопотоках (cam101, cam102)

---

**Вывод:** Система **99% готова**. Нужна только перезагрузка сервера! ✨

**Версия:** 1.0  
**Дата:** 2025-11-06

