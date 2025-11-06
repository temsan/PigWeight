# 🚀 Быстрый старт - PigWeight v1.1.0

**Версия:** 1.1.0  
**Дата:** 2025-11-03

---

## 📋 Требования

- **Python:** 3.9+
- **GPU:** рекомендуется (работает и на CPU)
- **RAM:** минимум 4GB (рекомендуется 8GB+)
- **Диск:** минимум 10GB свободного места

---

## 🎯 За 5 минут до первого результата

### 1️⃣ Установка зависимостей

```bash
# Установить Python зависимости
pip install -r requirements.txt

# Установить зависимости для обработки видео (опционально)
pip install -r requirements-pig-tracking.txt
```

### 2️⃣ Конфигурация

```bash
# Скопировать пример конфигурации
cp config.env.example .env

# Отредактировать .env - добавить URL камер
nano .env
```

**Основные переменные:**
```bash
# Камеры
CAM_CH101=rtsp://192.168.1.10:554/stream
CAM_CH102=rtsp://192.168.1.11:554/stream

# Модель
MODEL_PATH=models/best.pt
DEVICE=auto  # или 'cuda:0', 'cpu'

# Параметры обработки
CONF_THRESHOLD=0.30
FRAME_SKIP=0
TARGET_FPS=25
```

### 3️⃣ Проверить конфигурацию

```bash
# Проверить здоровье сервера
curl http://localhost:8000/debug/health

# Тестировать RTSP камеру
curl -X POST http://localhost:8000/debug/test_rtsp/cam101
```

### 4️⃣ Добавить видео на обработку

```bash
# Способ 1: Через API
curl -X POST http://localhost:8000/api/processing/queue/add \
  -H "Content-Type: application/json" \
  -d '{"video_path": "uploads/test.mp4"}'

# Способ 2: Через консольное приложение
python console_app.py --video uploads/test.mp4

# Способ 3: Интерактивный режим
python console_app.py
```

### 5️⃣ Проверить результаты

```bash
# Получить список актов взвешивания
curl http://localhost:8000/api/records

# Получить детали конкретного акта
curl http://localhost:8000/api/records/act_file_1234567890_20251103-120000

# Проверить статистику очереди
curl http://localhost:8000/api/processing/queue/stats
```

---

## 🖥️ Запуск приложения

### Веб-сервер (API + Web UI)

```bash
# Режим разработки (с hot reload)
python main.py --debug

# Production режим
python main.py

# С выбором рантайма
python main.py --runtime auto
python main.py --runtime pytorch
python main.py --runtime onnx-gpu
python main.py --runtime onnx-cpu
python main.py --runtime cpu
```

Откройте браузер: **http://localhost:8000**

### Консольное приложение

```bash
# Интерактивный выбор видео
python console_app.py

# Обработка конкретного видео
python console_app.py --video uploads/test.mp4

# Тестовый режим с проверкой
python console_app.py --mode test --video uploads/test.mp4 --excel-reference docs/manual.xlsx
```

---

## 📊 Мониторинг и диагностика

### Статус сервера

```bash
curl http://localhost:8000/debug/health
```

Проверяет:
- ✅ Статус сервера
- ✅ Загруженные компоненты
- ✅ Диагностику RTSP
- ✅ Параметры производительности

### RTSP диагностика

```bash
# Подробная информация о RTSP подключениях
curl http://localhost:8000/debug/rtsp

# Тест конкретной камеры
curl -X POST http://localhost:8000/debug/test_rtsp/cam101
```

### Статус инференса

```bash
curl http://localhost:8000/debug/infer_status
```

Показывает:
- 💾 Использование памяти
- ⚙️ Нагрузка CPU/GPU
- 🔋 Количество активных потоков

---

## 🎬 Работа с видео

### Структура проекта

```
PigWeight/
├── uploads/              # Входящие видеофайлы
├── results/              # JSON результаты
├── records/
│   ├── events/          # JSONL события
│   ├── frames/          # Кадры событий (опционально)
│   └── queue/           # Состояние очереди
├── logs/                # Логи приложения
├── static/              # Веб-интерфейс
├── api/                 # API код
├── core/                # Ядро обработки
├── pig_tracking/        # Модули отслеживания
└── services/            # Сервисы (логирование, БД)
```

### Формат результатов

**JSON результаты:** `results/video_name_timestamp_results.json`
```json
{
  "frames_processed": 2150,
  "act_stats": {
    "completed_acts_count": 3,
    "peak_concurrent": 8,
    "completed_acts": [
      {
        "act_id": 1,
        "started_at": 1699005600.0,
        "ended_at": 1699005720.0,
        "left_count": 12,
        "right_count": 11,
        "peak_count": 7
      }
    ]
  },
  "crossing_stats": {
    "total_crossings": 47,
    "left_crossings": 24,
    "right_crossings": 23
  }
}
```

---

## 📈 WebSocket для Real-time обновлений

### JavaScript пример

```javascript
// Добавить видео в очередь
const response = await fetch('/api/processing/queue/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: 'uploads/test.mp4' })
});
const { task_id } = await response.json();

// Подключиться к WebSocket для прогресса
const ws = new WebSocket(`ws://localhost:8000/ws/processing/progress?task_id=${task_id}`);

ws.onopen = () => console.log('✅ Подключено к очереди');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`${data.event}: ${data.task.progress}%`);
    
    if (data.event === 'complete') {
        console.log('✅ Обработка завершена!');
        console.log(data.task.result);
    }
};

ws.onerror = (error) => console.error('❌ Ошибка:', error);
```

---

## 🔧 Оптимизация производительности

### Для слабого железа

```bash
# .env
DEVICE=cpu
USE_HALF=false
FRAME_SKIP=2        # Пропускать каждый второй кадр
TARGET_FPS=15       # Снизить FPS обработки
BATCH_SIZE=1        # Обработка по одному кадру
```

### Для среднего железа

```bash
DEVICE=cuda:0
USE_HALF=true       # FP16 precision
FRAME_SKIP=1        # Обрабатывать все кадры
TARGET_FPS=25
BATCH_SIZE=2
```

### Для мощного GPU

```bash
DEVICE=cuda:0
USE_HALF=true
FRAME_SKIP=0        # Все кадры, без пропусков
TARGET_FPS=30
BATCH_SIZE=4        # Batch processing
```

---

## 🗄️ Работа с базой данных

### Запустить Supabase локально

```bash
docker-compose up -d

# Веб-интерфейс Supabase: http://localhost:8000
# API: http://localhost:3000
```

### Просмотр данных

```bash
# Все акты взвешивания
SELECT * FROM weighing_acts;

# Все пересечения за день
SELECT * FROM crossings 
WHERE timestamp >= NOW() - INTERVAL '1 day';

# Статистика по камерам
SELECT stream_id, COUNT(*) as count
FROM crossings
GROUP BY stream_id;
```

---

## ❌ Устранение неполадок

### Сервер не запускается

```bash
# Проверьте, свободен ли порт 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Может быть занят - используйте другой:
python main.py --port 9000
```

### RTSP камера не подключается

```bash
# 1. Проверьте доступность
ping 192.168.1.10

# 2. Тестируйте RTSP URL с VLC
vlc rtsp://192.168.1.10:554/stream

# 3. Используйте диагностику
curl -X POST http://localhost:8000/debug/test_rtsp/cam101
```

### Видео не обрабатывается

```bash
# 1. Проверьте формат видео
ffprobe uploads/test.mp4

# 2. Проверьте логи
tail -f logs/app.log

# 3. Проверьте статус очереди
curl http://localhost:8000/api/processing/queue/tasks?status=failed
```

### Низкая производительность

```bash
# Диагностика нагрузки
curl http://localhost:8000/debug/infer_status

# Оптимизация:
# - Увеличить FRAME_SKIP
# - Снизить TARGET_FPS
# - Включить FP16 (USE_HALF=true)
# - Уменьшить размер модели
```

---

## 📚 Документация

- **[API документация](API_DOCUMENTATION.md)** - полная документация всех эндпоинтов
- **[TODO.md](TODO.md)** - список текущих и будущих задач
- **[СТАТУС_ПРОЕКТА.md](СТАТУС_ПРОЕКТА.md)** - подробный статус компонентов
- **[api/swagger_docs.py](api/swagger_docs.py)** - Swagger документация

---

## 🎓 Примеры использования

### Python скрипт для массовой обработки

```python
#!/usr/bin/env python3
import requests
from pathlib import Path
import time

BASE_URL = "http://localhost:8000"
UPLOADS_DIR = Path("uploads")

# Найти все видеофайлы
video_files = list(UPLOADS_DIR.glob("*.mp4")) + list(UPLOADS_DIR.glob("*.avi"))

# Добавить в очередь
task_ids = []
for video_file in video_files:
    response = requests.post(
        f"{BASE_URL}/api/processing/queue/add",
        json={"video_path": str(video_file)}
    )
    task_id = response.json()["task_id"]
    task_ids.append(task_id)
    print(f"✅ Добавлено: {video_file.name} ({task_id})")

# Ждать завершения
print(f"⏳ Ожидание обработки {len(task_ids)} видео...")
while True:
    response = requests.get(f"{BASE_URL}/api/processing/queue/stats")
    stats = response.json()["stats"]
    
    print(f"Статус: {stats['completed']}/{stats['total_tasks']} завершено")
    
    if stats['processing'] == 0 and stats['pending'] == 0:
        break
    
    time.sleep(5)

print("✅ Все видео обработаны!")
```

---

## 🚀 Production Deployment

### Docker контейнеризация

```bash
# Собрать образ
docker build -t pigweight:1.1.0 .

# Запустить контейнер
docker run -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -e CAM_CH101=rtsp://... \
  pigweight:1.1.0
```

### Systemd сервис

```bash
# /etc/systemd/system/pigweight.service
[Unit]
Description=PigWeight Service
After=network.target

[Service]
Type=simple
User=pigweight
WorkingDirectory=/opt/pigweight
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📞 Техническая поддержка

При проблемах:
1. Проверьте `/logs/app.log`
2. Запустите диагностику: `/debug/health`, `/debug/rtsp`, `/debug/infer_status`
3. Обратитесь к документации или создайте issue

---

**Готовы к работе! 🎉**

Начните с простого: загрузите видео, нажмите "Обработать" и дождитесь результатов.

Удачи! 🚀
