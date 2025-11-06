# PigWeight v3.0 - Руководство по развертыванию

## 🎯 Статус системы

✅ **Все основные функции работают**
- Видеофайлы обрабатываются
- RTSP камеры поддерживаются
- Веб-интерфейс запущен
- Консольное приложение готово

## 🖥️ Требования к серверу

### Минимальные:
- Windows Server 2019 и выше
- Python 3.10+
- 4GB RAM
- 2GB свободного места

### Рекомендуемые:
- Windows Server 2022
- Python 3.11+
- 8GB RAM
- 10GB свободного места
- GPU (опционально, для ускорения)

## 📦 Установка зависимостей

### 1. Клонирование репозитория
```bash
git clone <repo-url>
cd PigWeight
```

### 2. Создание виртуального окружения
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Установка опциональных пакетов (для полной функциональности)
```bash
# WebRTC поддержка (для реального времени в браузер)
pip install aiortc av

# Excel сверка (для проверки данных)
pip install openpyxl

# Мониторинг системы
pip install psutil
```

## ⚙️ Конфигурация

### 1. Создание файла `.env`

```env
# === КАМЕРЫ ===
CAM_CH101=rtsp://user:password@192.168.1.100:554/stream
CAM_CH102=rtsp://user:password@192.168.1.101:554/stream

# === ПАРАМЕТРЫ ОБРАБОТКИ ===
DEVICE=cpu
PREFERRED_RUNTIME=cpu
TARGET_FPS=15
IMG_SIZE=640
CONF_THRESHOLD=0.5

# === ПАПКИ ===
UPLOADS_DIR=uploads
RECORDS_DIR=records
MODELS_DIR=models

# === ОПЦИОНАЛЬНЫЕ ===
# SUPABASE_URL=https://...
# SUPABASE_KEY=...
# REDIS_URL=redis://localhost:6379
```

### 2. Структура папок

```
PigWeight/
├── uploads/          # Видеофайлы для обработки
├── records/          # Результаты обработки
├── models/           # YOLO модели
├── logs/             # Логи приложения
└── .env              # Конфигурация
```

## 🚀 Запуск системы

### Вариант 1: Веб-интерфейс (рекомендуется)

```bash
python main.py
```

Откройте браузер: `http://localhost:8000`

**Возможности:**
- Выбор источника (видео/камера)
- WebRTC трансляция в реальном времени
- Просмотр метрик
- Сохранение результатов

### Вариант 2: Консольное приложение

```bash
python console_app.py
```

**Режимы:**
1. **Process** - обработка одного видео/камеры
2. **Monitor** - фоновый мониторинг с параметрами
3. **Test** - обработка с Excel проверкой

### Вариант 3: Командная строка

```bash
# Обработать видео
python console_app.py --video uploads/video.mp4

# Мониторить RTSP поток
python console_app.py --mode monitor --rtsp rtsp://camera.url

# Непрерывный мониторинг
python console_app.py --mode monitor --rtsp rtsp://camera.url --continuous

# С кастомными параметрами
python console_app.py --mode monitor \
  --rtsp rtsp://camera.url \
  --confidence 0.6 \
  --min-pigs 2 \
  --max-interval 45 \
  --continuous
```

## 🔍 Диагностика

### Проверка логов
```bash
# Последние 20 строк
tail -n 20 logs/app.log

# Или в Windows PowerShell
Get-Content logs\app.log -Tail 20
```

### Типичные логи и решения

#### 1. Warning: openpyxl не установлен
```
openpyxl не установлен. Функция сверки с Excel недоступна.
```
**Решение:**
```bash
pip install openpyxl
```

#### 2. Warning: aiortc не установлена
```
aiortc не установлена, WebRTC функционал отключен
```
**Решение:**
```bash
pip install aiortc av
```
**Примечание:** Система работает с MJPEG fallback

#### 3. RTSP поток timeout
```
Stream timeout triggered after 30056.571000 ms
```
**Решение:**
- Проверьте RTSP URL
- Убедитесь что камера доступна из сервера
- Проверьте firewall

#### 4. GPU недоступен
```
CPU: CUDA недоступен
```
**Нормально для CPU серверов**
- Система автоматически переходит на CPU
- Обработка медленнее, но стабильна

## 📊 Мониторинг

### Web Dashboard
```
http://localhost:8000/metrics
```

**Показывает:**
- Статус сервера
- Активные потоки
- Результаты обработки
- История актов

### API Endpoints

#### Здоровье сервера
```bash
curl http://localhost:8000/api/health
```

#### Список камер
```bash
curl http://localhost:8000/api/cameras
```

#### Список результатов
```bash
curl http://localhost:8000/api/records
```

## 🔧 Оптимизация

### Для слабого сервера
```env
IMG_SIZE=416        # Меньше разрешение
TARGET_FPS=10       # Меньше кадров
CONF_THRESHOLD=0.3  # Менее чувствительно
```

### Для мощного сервера
```env
IMG_SIZE=640        # Стандартное
TARGET_FPS=30       # Больше кадров
CONF_THRESHOLD=0.5  # Более чувствительно
```

## 🔒 Безопасность

### На production сервере:
1. Используйте HTTPS вместо HTTP
2. Добавьте аутентификацию
3. Ограничьте доступ IP адресами
4. Обновляйте пакеты регулярно
5. Используйте strong пароли для RTSP

### Пример HTTPS конфиг
```bash
# Генерируем самоподписанный сертификат
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Запускаем с HTTPS
python main.py --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

## 📝 Логирование

Логи сохраняются в `logs/app.log`

### Уровни логирования
- INFO - основные события
- WARNING - предупреждения (опциональные пакеты)
- ERROR - ошибки (нужно исправить)
- DEBUG - подробная информация (для разработки)

### Ротация логов
Автоматически создаются новые логи по дате

## 🆘 Troubleshooting

### Сервер не запускается
1. Проверьте Python версию: `python --version`
2. Проверьте установку зависимостей: `pip list`
3. Проверьте порт 8000: `netstat -ano | findstr :8000`

### Камеры не видны
1. Проверьте .env файл
2. Убедитесь что переменные `CAM_CH*` установлены
3. Перезагрузите приложение

### RTSP не подключается
1. Проверьте URL: `rtsp://user:pass@host:port/stream`
2. Проверьте доступность из сервера: `ping <host>`
3. Проверьте credentials (пользователь/пароль)
4. Посмотрите логи: `tail logs/app.log`

### Медленная обработка
1. Уменьшите IMG_SIZE в .env
2. Уменьшите TARGET_FPS
3. Проверьте нагрузку на CPU: `wmic path winmgmts: get loadpercentage`
4. Попробуйте GPU если доступна

## 📚 Документация

- **README.md** - основная документация
- **QUICKSTART.md** - быстрый старт
- **API_DOCUMENTATION.md** - описание API
- **WEBRTC_HLS_GUIDE.md** - WebRTC и HLS
- **DATABASE_SETUP.md** - интеграция БД

## 🎯 Следующие шаги

1. ✅ Установка зависимостей
2. ✅ Конфигурация .env
3. ✅ Запуск приложения
4. ✅ Тестирование с видео
5. ✅ Подключение RTSP камер
6. ⏭️ Настройка Supabase (опционально)
7. ⏭️ Развертывание на production

## 📞 Поддержка

Если что-то не работает:
1. Проверьте логи: `logs/app.log`
2. Проверьте конфигурацию: `.env`
3. Попробуйте обновить зависимости: `pip install --upgrade -r requirements.txt`
4. Перезагрузите приложение

---

**Версия:** PigWeight v3.0
**Дата:** Ноябрь 2025
**Статус:** Production Ready ✅
