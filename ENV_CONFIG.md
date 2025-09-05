# Конфигурация и логирование PigWeight

## 📊 Новый формат логов производительности

Система теперь записывает детальные метрики в human-readable формате:

### Формат записей batch performance:
```json
{
  "timestamp": 1756906880.215,
  "datetime": "2025-09-03 16:41:20.215",
  "stream_id": "cam101",
  "batch_size": 8,
  "detections": 5,
  "batch_time_ms": 250.5,
  "preprocess_time_ms": 15.3,
  "inference_time_ms": 180.2,
  "postprocess_time_ms": 55.0,
  "fps": 32.0
}
```

### Формат summary записей:
```json
{
  "timestamp": 1756906880.215,
  "datetime": "2025-09-03 16:41:20.215",
  "type": "performance_summary",
  "stream_id": "cam101",
  "total_batches": 150,
  "avg_batch_time_ms": 245.8,
  "avg_inference_time_ms": 175.2,
  "avg_preprocess_time_ms": 14.8,
  "avg_postprocess_time_ms": 55.8,
  "batch_size": 8,
  "throughput_fps": 31.5
}
```

### Консольный вывод:
```
[16:41:20] cam101: batch_size=8, detections=5, fps=32.0, inference=180.2ms
```

## Конфигурация (.env)

Создайте файл `.env` в корне проекта со следующими настройками для оптимизации:

## Рекомендуемая конфигурация для высокой производительности

```env
# Model Configuration
MODEL_PATH=models/pig_yolo11-seg.v4.pt
# Для CPU оптимизации используйте ONNX:
# MODEL_PATH=models/pig_yolo11-seg.onnx
DETECTION_MODE=pig-only
PIG_CLASS_ID=0
CONF_THRESHOLD=0.30

# Device Configuration
DEVICE=cuda:0  # Или 'cpu' для CPU-only
USE_HALF=true  # FP16 для GPU

# Inference Configuration
IMG_SIZE=960
BATCH_SIZE=8   # Увеличен для лучшей производительности
MAX_WAIT_MS=50

# Broker and Results Configuration
FRAME_BROKER_CACHE=16
RESULTS_TTL_SECONDS=30
BROADCAST_MIN_INTERVAL=0.05

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
RELOAD=false

# Video Configuration
FPS=12
JPEG_QUALITY=80

# Lines for Counting
LINE_LEFT_X=0.25
LINE_RIGHT_X=0.75
```

## Оптимизации для разных сценариев

### Для максимальной производительности на GPU:
```env
DEVICE=cuda:0
USE_HALF=true
BATCH_SIZE=16
MODEL_PATH=models/pig_yolo11-seg.v4.pt
```

### Для CPU оптимизации:
```env
DEVICE=cpu
USE_HALF=false
BATCH_SIZE=4
MODEL_PATH=models/pig_yolo11-seg.onnx
```

### Для низкой задержки (WebRTC):
```env
BROADCAST_MIN_INTERVAL=0.03  # 30 FPS updates
BATCH_SIZE=8
MAX_WAIT_MS=30
```

## Конвертация модели в ONNX

Для CPU оптимизации конвертируйте модель:

```bash
# Установите ONNX Runtime
pip install onnxruntime onnx onnxoptimizer

# Конвертируйте модель
python scripts/convert_to_onnx.py --model_path models/pig_yolo11-seg.v4.pt --output_path models/pig_yolo11-seg.onnx --img_size 960
```

## Мониторинг производительности

После запуска система будет логировать перформанс в `logs/perf.log`:

```json
{
  "timestamp": 1703123456.789,
  "stream_id": "cam101",
  "batch_size": 8,
  "detections": 3,
  "batch_time_ms": 450.5,
  "inference_time_ms": 380.2,
  "preprocess_time_ms": 45.3,
  "postprocess_time_ms": 25.0,
  "fps": 17.8
}
```

Используйте анализатор для разбора логов:
```bash
python scripts/analyze_pipeline.py --log_file logs/perf.log
```
