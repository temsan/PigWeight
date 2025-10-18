# Конфигурация и логирование PigWeight

## 📊 Упрощенное логирование

Система использует простое логирование без лишних деталей:

### Логирование производительности:
```
cam101: размер=8, обнаружено=5, fps=32.1, инференс=180мс
Сводка для cam101: батчей=150, средний_fps=31.5, средний_инференс=175мс
```

### Общие логи:
```
2025-01-15 16:41:20 - INFO - Запуск сервера на http://0.0.0.0:8000
2025-01-15 16:41:20 - INFO - Режим отладки: False, Горячая перезагрузка: False
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

# Оптимизированная предобработка (НОВОЕ!)
PREPROCESSING_METHOD=adaptive      # adaptive, center_crop, letterbox
ANTI_LETTERBOX=false              # Отключить для избежания конфликтов

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
# Оптимизированная предобработка
PREPROCESSING_METHOD=center_crop
ANTI_LETTERBOX=false
```

### Для CPU оптимизации:
```env
DEVICE=cpu
USE_HALF=false
BATCH_SIZE=4
MODEL_PATH=models/pig_yolo11-seg.onnx
# Оптимизированная предобработка
PREPROCESSING_METHOD=adaptive
ANTI_LETTERBOX=false
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

Система логирует производительность в простом формате в консоль и файл `logs/app.log`.

Для анализа производительности используйте:
```bash
python scripts/analyze_pipeline.py --log_file logs/app.log
```
