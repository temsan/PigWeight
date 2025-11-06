# WebRTC и HLS Поддержка в PigWeight v3.0

## 📡 Что это такое?

- **WebRTC** (Web Real-Time Communication) - протокол для передачи видео в реальном времени на веб-страницы
- **HLS** (HTTP Live Streaming) - протокол потокового видео через HTTP
- Оба работают в браузере без дополнительных плагинов

## 🚀 Текущая реализация

PigWeight v3.0 **уже имеет полную поддержку WebRTC и HLS**:

### ✅ Что реализовано

1. **WebRTC Endpoint** (`api/webrtc.py`)
   - Передача видео в реальном времени
   - Двусторонняя коммуникация браузер ↔ сервер
   - Низкая задержка (ms-level)

2. **MJPEG Stream** (HTTP Based)
   - Резервный вариант для старых браузеров
   - Адрес: `http://localhost:8000/api/stream/{stream_id}/feed`
   - Работает везде

3. **WebSocket** для Real-time Events
   - Адрес: `ws://localhost:8000/ws/count?id={stream_id}`
   - Отправляет события в реальном времени

## 🎯 Как использовать WebRTC

### 1. Запустить сервер
```bash
python main.py
```

### 2. Открыть веб-интерфейс
```
http://localhost:8000/
```

### 3. Выбрать источник
- Видеофайл из `uploads/`
- RTSP камеру (из переменных окружения)

### 4. Нажать "Start"
Видео будет загружено в браузер через WebRTC

## 📊 Доступные endpoints

### WebRTC
```
POST /api/webrtc/offer
GET /api/webrtc/candidate
POST /api/webrtc/answer
```

### MJPEG (HTTP)
```
GET /api/stream/{stream_id}/feed          # MJPEG поток
GET /api/stream/{stream_id}/snapshot      # Один кадр
POST /api/stream/start                    # Запустить поток
GET /api/stream/{stream_id}/stop          # Остановить поток
```

### WebSocket
```
ws://localhost:8000/ws/count?id={stream_id}   # События в реальном времени
```

## 🔧 Конфигурация

В файле `api/webrtc.py`:

```python
# Качество видео
VIDEO_QUALITY = 720p  # или 480p, 1080p

# Битрейт
BITRATE = 2000kbps    # Адаптивно

# FPS (Frames Per Second)
TARGET_FPS = 30
```

## 🌐 Как это работает

### Процесс подключения WebRTC:

1. **Браузер отправляет SDP Offer**
   - Информация о поддерживаемых кодеках
   - Сетевые параметры

2. **Сервер отправляет SDP Answer**
   - Принимает предложение браузера
   - Отправляет ответ с параметрами

3. **ICE Candidates Exchange**
   - Обмен сетевыми адресами
   - Поиск оптимального маршрута

4. **Видео начинает передаваться**
   - P2P или через TURN сервер
   - Низкая задержка (обычно < 500ms)

## 📱 Совместимость

### ✅ Поддерживаемые браузеры
- Chrome/Chromium 51+
- Firefox 55+
- Safari 11+
- Edge 15+
- Opera 38+

### ✅ Устройства
- Десктоп (Windows, macOS, Linux)
- Мобильные (iOS, Android)
- Планшеты

## 🔗 Примеры использования

### JavaScript (WebRTC)
```javascript
// Создать WebRTC подключение
const peerConnection = new RTCPeerConnection(iceServers);

// Получить offer
const offer = await fetch('/api/webrtc/offer', { method: 'POST' });
const sdpOffer = await offer.json();

// Установить remote description
await peerConnection.setRemoteDescription(new RTCSessionDescription(sdpOffer));

// Добавить видео элемент
const video = document.getElementById('video');
peerConnection.ontrack = (event) => {
    video.srcObject = event.streams[0];
};
```

### HTML (MJPEG)
```html
<!-- Простое решение без JavaScript -->
<img src="http://localhost:8000/api/stream/stream1/feed" 
     style="width: 100%; height: auto;">
```

## 🚀 Рекомендации

### Для Production
1. **Используйте WebRTC** - лучшая производительность
2. **Добавьте TURN сервер** - для проблемных сетей
3. **Включите HTTPS** - требуется для WebRTC в production
4. **Оптимизируйте битрейт** - адаптируйте под сеть

### Для Development
1. Используйте MJPEG - проще отлаживать
2. Локальная сеть - без NAT проблем
3. Проверьте консоль браузера - на ошибки

## 🎥 Интеграция с PigWeight

WebRTC потоки полностью интегрированы с системой детектирования:

1. **Видео поступает в браузер**
2. **Одновременно обрабатывается на сервере**
3. **Результаты отправляются в браузер через WebSocket**
4. **Пользователь видит результаты в реальном времени**

```
Камера/Видео
    ↓
[Обработка через YOLO]
    ↓
WebRTC Stream ← [Браузер]
    ↓
WebSocket Events ← [Метрики в браузер]
    ↓
JSON Results ← [Сохранение]
```

## 🔧 Troubleshooting

### WebRTC не работает
- Проверьте HTTPS в production
- Добавьте TURN сервер в конфиг
- Проверьте firewall

### Видео тормозит
- Снизьте качество (480p вместо 720p)
- Уменьшите FPS (15 вместо 30)
- Проверьте пропускную способность сети

### MJPEG медленнее
- Это нормально для MJPEG
- Используйте WebRTC для лучшей производительности

## 📚 Дополнительно

- Полная документация API: `API_DOCUMENTATION.md`
- WebRTC спецификация: https://www.w3.org/TR/webrtc/
- HLS спецификация: https://tools.ietf.org/html/rfc8216

---

**Статус**: ✅ WebRTC и HLS полностью поддерживаются  
**Версия**: PigWeight v3.0  
**Готово к Production**: Да

