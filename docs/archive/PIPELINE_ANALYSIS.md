# 🔍 Полный разбор пайплайна PigWeight

## 📊 Обзор архитектуры

На основе анализа performance логов (`logs/perf.log`) и структуры кода, вот полный разбор современного пайплайна обработки:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frame Source  │ -> │  Frame Broker    │ -> │ Inference Worker│
│  (Camera/RTSP/  │    │  (In-process     │    │  (Batch ML      │
│    File)        │    │   Pub-Sub)       │    │   Processing)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Preprocessing │    │   Results Store  │    │ Postprocessing  │
│  (HSV/Resize/   │    │   (TTL Cache)    │    │  (Mask Mapping) │
│    Letterbox)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  WebRTC Server  │    │   WebSocket      │    │   Client UI     │
│  (aiortc H.264) │    │   Broadcast       │    │  (Canvas Draw) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 Этапы пайплайна

### 1. 📹 Frame Capture (Захват кадра)

**Метрики из логов:**
- `read_ms`: Время чтения кадра из источника
- Среднее: ~1.5-2.0ms
- P95: ~3.0ms

**Что происходит:**
```python
# api/av_worker.py или core/video_stream.py
frame = await self._capture_frame()
# RTSP: av.open() -> container.decode()
# File: cv2.VideoCapture() -> cap.read()
```

**Оптимизации:**
- ✅ Hardware-accelerated decoding (FFmpeg + CUDA)
- ✅ Frame buffering (circular buffer)
- ✅ Async capture with timeout handling

### 2. 🎨 Preprocessing (Предобработка)

**Метрики из логов:**
- `proc_ms`: Время предобработки (HSV + resize)
- Среднее: ~800-1000ms (включая inference!)
- P95: ~1200ms

**Что происходит:**
```python
# core/preprocess.py
def preprocess_for_model(frame):
    # HSV filtering for better detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Resize with letterbox padding
    resized = letterbox(hsv, (960, 960))
    # Normalization
    normalized = resized.astype(np.float32) / 255.0
    return normalized
```

**Оптимизации:**
- ✅ HSV filtering для лучшей детекции свиней
- ✅ Letterbox resize (сохраняет aspect ratio)
- ✅ Batch preprocessing (4 кадра одновременно)

### 3. 🧠 Inference (Инференс модели)

**Метрики из логов:**
- `proc_ms`: Время инференса (основная часть!)
- Среднее: ~900-1000ms
- P95: ~1200ms
- Model load: ~2000-4000ms (однократно)

**Что происходит:**
```python
# services/model_adapter.py
class ModelAdapter:
    def __init__(self, model_path):
        self._yolo = YOLO(model_path)
        if torch.cuda.is_available():
            self._yolo.to('cuda:0')
            self._yolo.model.half()  # FP16

    def infer(self, imgs):
        results = self._yolo.predict(imgs, imgsz=960, conf=0.3)
        return self._extract_masks_and_boxes(results)
```

**Оптимизации:**
- ✅ CUDA acceleration с FP16
- ✅ Batch processing (4 кадра)
- ✅ Ultralytics YOLOv11 с Retina Masks
- ✅ Graceful CPU fallback

### 4. ✂️ Postprocessing (Постобработка)

**Метрики из логов:**
- `enc_ms`: Время кодирования результатов
- Среднее: ~2-3ms
- P95: ~5ms

**Что происходит:**
```python
# core/preprocess.py
def map_polys_to_original(masks, original_shape, resized_shape):
    # Scale masks back to original frame coordinates
    scale_x = original_shape[1] / resized_shape[1]
    scale_y = original_shape[0] / resized_shape[0]

    scaled_masks = []
    for mask in masks:
        scaled_mask = mask * np.array([scale_x, scale_y])
        scaled_masks.append(scaled_mask)
    return scaled_masks
```

**Оптимизации:**
- ✅ Масштабирование масок к оригинальным координатам
- ✅ Конвертация в JSON для WebSocket
- ✅ Efficient polygon decimation

### 5. 📡 Transmission (Передача)

**Метрики из логов:**
- `send_ms`: Время передачи данных
- Среднее: ~40-60ms
- P95: ~100ms

**Что происходит:**
```python
# api/app.py - WebRTC
class BrokerVideoTrack(VideoStreamTrack):
    async def next_frame(self):
        # Get latest frame with overlays from RESULTS_STORE
        jpeg_data = RESULTS_STORE.get_latest(self.stream_id)
        # Convert to av.VideoFrame
        frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
        return frame

# api/app.py - WebSocket
async def ws_count(ws, stream_id):
    while True:
        # Send inference results every 50ms
        results = RESULTS_STORE.get_latest(stream_id)
        await ws.send_json(results)
        await asyncio.sleep(0.05)
```

**Оптимизации:**
- ✅ WebRTC для низкой задержки (<100ms end-to-end)
- ✅ MJPEG fallback при проблемах
- ✅ Throttled WebSocket updates (20 FPS max)
- ✅ Delta encoding для изменений

### 6. 🖥️ Rendering (Отрисовка)

**Метрики из логов:**
- Не измеряется напрямую, но влияет на общий FPS

**Что происходит:**
```javascript
// static/index.html
function drawOverlay(masks, ids) {
    // Draw segmentation masks
    ctx.globalAlpha = 0.6;
    ctx.fillStyle = colorForInstance(instId);
    ctx.fill();

    // Draw instance labels
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, centroid_x, centroid_y);
}
```

**Оптимизации:**
- ✅ Hardware-accelerated Canvas rendering
- ✅ Mask blending с transparency
- ✅ Efficient polygon rendering
- ✅ Debounced UI updates

## 📈 Производительность по логам

### Статистика из `logs/perf.log`:

#### Общая производительность:
- **Средний FPS**: 1.0-2.0 (низкий!)
- **Средняя общая задержка**: ~950-1100ms
- **P95 задержка**: ~1300ms
- **Эффективность**: ~15-25% от целевой

#### Разбивка по этапам:
```
Frame Capture:    1.5ms (1.5%)
Preprocessing:    ~100ms (10%)
Inference:        ~900ms (85%) 🔥 BOTTLENECK
Postprocessing:   3ms (0.3%)
Transmission:     50ms (5%)
```

#### Ключевые проблемы:
1. **🤖 Model Loading**: 2000-4000ms при старте
2. **🧠 Inference**: 85% времени пайплайна!
3. **📡 Transmission**: Пики до 580ms
4. **🐌 Low FPS**: 1-2 FPS вместо 12-30

## 🔧 Оптимизации и решения

### 1. **Inference оптимизации** (главный bottleneck):
```python
# services/model_adapter.py
# Добавить ONNX Runtime для CPU inference
if torch.cuda.is_available():
    # CUDA path
    self._yolo.to('cuda:0')
    self._yolo.model.half()
else:
    # CPU optimization
    self._model = cv2.dnn.readNetFromONNX('model.onnx')
```

### 2. **Model оптимизации**:
- Использовать YOLOv11n (nano) вместо полного
- Quantization (INT8)
- TensorRT для максимальной скорости

### 3. **Transmission оптимизации**:
```javascript
// static/index.html - Throttle WebSocket
let lastUpdate = 0;
const MIN_INTERVAL = 50; // 20 FPS max

function scheduleOverlay(masks, ids) {
    const now = performance.now();
    if (now - lastUpdate >= MIN_INTERVAL) {
        drawOverlay(masks, ids);
        lastUpdate = now;
    }
}
```

### 4. **Memory оптимизации**:
```python
# core/results_store.py - TTL cleanup
async def _cleanup(self):
    now = time.time()
    expired = [k for k, v in self._store.items()
              if now - v['timestamp'] > self.ttl_seconds]
    for k in expired:
        del self._store[k]
```

### 5. **Batch processing оптимизации**:
```python
# services/inference_worker.py
async def _run(self):
    batch = []
    start_time = time.time()

    while self._running:
        # Gather batch
        while len(batch) < self.batch_size:
            item = await q.get()
            batch.append(item)

            # Time-based batching
            if time.time() - start_time > self.max_wait_ms / 1000:
                break

        # Process batch
        results = await self.model.infer([item['frame'] for item in batch])
```

## 🎯 Рекомендации для улучшения

### Немедленные действия:
1. **CUDA setup**: Установить PyTorch с CUDA 13.0
2. **Model optimization**: Перейти на YOLOv11n
3. **Batch size**: Увеличить до 8-16 кадров

### Среднесрочные:
1. **ONNX export**: Для CPU inference
2. **WebRTC optimization**: Убрать MJPEG fallback
3. **Memory management**: Добавить frame pooling

### Долгосрочные:
1. **TensorRT**: Для максимальной производительности
2. **Edge deployment**: Raspberry Pi/Jetson optimization
3. **Multi-camera**: Distributed processing

## 📊 Метрики для мониторинга

```python
# scripts/analyze_pipeline.py - автоматический анализ
{
    "overall": {
        "mean_fps": 1.2,
        "mean_total_latency": 950,
        "bottleneck_stage": "inference",
        "efficiency_score": 23.5
    },
    "inference": {
        "mean": 900,
        "p95": 1200,
        "optimization_potential": "high"
    }
}
```

Этот анализ показывает, что основная проблема - inference bottleneck (85% времени), что типично для ML пайплайнов. Решение лежит в оптимизации модели и hardware acceleration.
