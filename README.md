# 🐷 PigWeight - Система отслеживания свиней

Система для анализа и взвешивания свиней с использованием компьютерного зрения, YOLO11 и ByteTrack. 

**Статус:** ✅ MVP готов к тестированию  
**Версия:** 1.0.0  
**Дата:** 2025-10-18

---

## 🚀 Быстрый старт

```bash
# 1. Проверка системы
python scripts/setup/check_system.py

# 2. Запуск Supabase
docker-compose up -d

# 3. Запуск консольного приложения
python console_app.py

# 4. Запуск API (в отдельном терминале)
python -m uvicorn api.app:app --port 8080 --reload
```

📖 **Подробнее:** [docs/guides/QUICKSTART.md](docs/guides/QUICKSTART.md)

---

## Архитектура

### Ключевые компоненты

- **WebRTC Streaming**: Низколатентная передача видео с поддержкой H.264
- **Frame Broker**: In-process pub-sub для асинхронной передачи кадров
- **Inference Service**: Отдельный воркер для батчинга и ML-инференса
- **Results Store**: Кеширование результатов с TTL для оптимизации
- **HSV Pipeline**: Предварительная обработка для улучшения детекции

### Преимущества новой архитектуры

- ✅ Разделение видео-потока и ML-инференса (независимая работа)
- ✅ Батчинг кадров для эффективного GPU-использования
- ✅ WebRTC для низкой задержки (<100ms end-to-end)
- ✅ Graceful fallback на MJPEG при проблемах
- ✅ Масштабируемость: поддержка множественных стримов
- ✅ Оптимизация памяти с TTL и ограниченными очередями

## Установка

### Базовая установка

```bash
# Клонирование репозитория
git clone https://github.com/username/PigWeight.git
cd PigWeight

# Создание виртуального окружения
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Установка зависимостей
python main.py --install

# Или вручную
pip install -r requirements.txt
```

### CUDA поддержка (для GPU ускорения)

Если у вас есть NVIDIA GPU, PyTorch автоматически установится с CUDA поддержкой.

Проверка CUDA:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### Запуск

```bash
python main.py
```

Приложение будет доступно по адресу: http://localhost:8000

### Режимы работы

- **WebRTC** (по умолчанию): Низкая задержка, оптимально для реального времени
- **MJPEG**: Совместимость, работает везде, выше задержка

## Конфигурация

Создайте файл `.env` в корне проекта:

```env
# Модель
MODEL_PATH=models/pig_yolo11-seg.v4.pt
DETECTION_MODE=pig-only
PIG_CLASS_ID=0

# Устройство (автоопределение CUDA/CPU)
DEVICE=cuda:0
USE_HALF=true

# Инференс
IMG_SIZE=960
BATCH_SIZE=4
MAX_WAIT_MS=50
CONF_THRESHOLD=0.30

# Брокер и кеширование
FRAME_BROKER_CACHE=16
RESULTS_TTL_SECONDS=30

# Сервер
HOST=0.0.0.0
PORT=8000
DEBUG=false
RELOAD=false

# Камеры (для RTSP)
CAM_CH101=rtsp://user:pass@camera-ip/live
CAM_CH102=rtsp://user:pass@camera-ip/live
```

## Тестирование производительности

### Запуск тестов

```bash
# Тест производительности стрима
python scripts/stream_performance_test.py --stream_id cam101 --transport webrtc --duration 60

# Тест с MJPEG
python scripts/stream_performance_test.py --stream_id cam101 --transport mjpeg --duration 60
```

### Метрики производительности

Тест измеряет:
- **Задержка инференса** (inference latency)
- **FPS обработки** (frames per second)
- **Использование CPU/памяти**
- **Сравнение WebRTC vs MJPEG**
- **Влияние батчинга**

## Развертывание

### Production-сервер

```bash
# Установка production зависимостей
pip install gunicorn uvicorn[standard]

# Запуск с Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Или напрямую через uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker развертывание

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

```bash
# Сборка и запуск
docker build -t pigweight .
docker run -p 8000:8000 -v /path/to/models:/app/models pigweight
```

### Оптимизации для production

1. **CUDA оптимизации**:
   - Установите PyTorch с CUDA
   - Настройте `USE_HALF=true` для FP16
   - Мониторьте использование GPU

2. **Сетевая оптимизация**:
   - Используйте WebRTC для низкой задержки
   - Настройте STUN/TURN серверы при необходимости
   - Оптимизируйте размер батчей

3. **Мониторинг**:
   - Логи пишутся в `logs/api.log`
   - Используйте `/api/health` для проверки состояния
   - Мониторьте метрики производительности

## Структура проекта

- `api/` - API эндпоинты и бэкенд логика
- `core/` - Ядро системы (конфиг, брокер, препроцессинг)
- `services/` - Сервисы (инференс, адаптер модели)
- `models/` - Модели машинного обучения
- `static/` - Статические файлы (CSS, JS, изображения)
- `uploads/` - Директория для загруженных видео файлов
- `scripts/` - Вспомогательные скрипты и тесты
- `docs/` - Документация

## Обслуживание

### Очистка директории uploads

В директории `uploads` хранятся файлы с оригинальными именами для кеширования. Для очистки этой директории можно использовать:

1. BAT-файл в корне проекта:
   ```
   clean_uploads.bat
   ```

2. Python-скрипт напрямую:
   ```bash
   python scripts/clean_uploads.py
   ```

Подробная документация по очистке доступна в [docs/uploads_cleanup.md](docs/uploads_cleanup.md).

## Лицензия

MIT

## Конфигурация (.env)

В корне проекта создайте файл `.env` и задайте как минимум путь к модели. Пример минимальной конфигурации:

```
PIG_MODEL_PATH=models/pig_yolo11-seg.v3.pt
# Для ultra-fast эндпоинтов можно переопределить отдельным путём:
# ULTRA_MODEL=models/pig_yolo11-seg.v3.pt

DETECTION_MODE=pig-only
FPS=12
JPEG_QUALITY=80
LINE_LEFT_X=0.25
LINE_RIGHT_X=0.75
```

Если переменные не заданы или файл модели отсутствует, бэкенд автоматически использует `models/best.pt` как резервный вариант.

Пример расширенной конфигурации (`.env`):

```
# Model
MODEL_PATH=models/pig_yolo11-seg.pt
DETECTION_MODE=pig-only
PIG_CLASS_ID=0

# Device
DEVICE=cuda:0
USE_HALF=true

# Inference
IMG_SIZE=960
BATCH_SIZE=4
MAX_WAIT_MS=50

# Broker
FRAME_BROKER_CACHE=16

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false
RELOAD=false
```

После правки `.env` перезапустите сервер.