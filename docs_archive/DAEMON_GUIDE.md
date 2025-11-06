# 🚀 Руководство по запуску PigWeight Daemon

## Обзор

**PigWeight Daemon** - это фоновый сервис для непрерывного мониторинга камер. Позволяет запустить обработку нескольких видеопотоков одновременно и автоматически перезапускает их при падении.

---

## 📋 Быстрый старт

### Запуск с мониторингом (интерактивно)

```bash
# Запустить демоны с отображением статуса
python run_daemon.py --start --monitor
```

**Что происходит:**
- ✅ Запускаются демоны для всех камер (cam101, cam102, ...)
- ✅ Каждый демон обрабатывает видеопоток независимо
- ✅ Система проверяет здоровье демонов каждые 30 сек
- ✅ При падении - автоматический перезапуск
- ✅ Статус выводится каждые 5 минут

**Выход:** Нажмите `Ctrl+C`

---

### Запуск в фоне (без интерактивного мониторинга)

```bash
# Запустить демоны и вернуть контроль консоли
python run_daemon.py --start
```

**Проверка статуса:**
```bash
ps aux | grep "console_app.py"
tail -f logs/daemon.log
```

---

## ⚙️ Конфигурация

### По умолчанию

Читается из переменных окружения в `.env`:

```env
# .env
RTSP_URL_CAM101=rtsp://localhost:8554/cam101
RTSP_URL_CAM102=rtsp://localhost:8554/cam102
```

### Сохранение пользовательской конфигурации

```bash
# Сохранить текущую конфигурацию в JSON
python run_daemon.py --save-config my_daemon_config.json

# Результат: my_daemon_config.json
# {
#   "cam101": {
#     "rtsp": "rtsp://localhost:8554/cam101",
#     "mode": "monitor",
#     "confidence": 0.30,
#     "min_pigs": 3,
#     "max_interval": 30.0,
#     "continuous": true
#   },
#   "cam102": { ... }
# }
```

### Запуск с пользовательской конфигурацией

```bash
# Загрузить конфигурацию из файла
python run_daemon.py --load-config my_daemon_config.json --start --monitor
```

---

## 📊 Мониторинг статуса

### Во время работы демона

Если запустили с `--monitor`, статус выводится автоматически:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 📊 Статус демонов       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Камера  │ Статус      │ PID  │ Перезапусков │
├─────────┼─────────────┼──────┼──────────────┤
│ cam101  │ 🟢 Активен  │ 1234 │ 0            │
│ cam102  │ 🟢 Активен  │ 5678 │ 0            │
└─────────┴─────────────┴──────┴──────────────┘
```

### Проверка логов

```bash
# Просмотр логов в реальном времени
tail -f logs/daemon.log

# Поиск ошибок
grep ERROR logs/daemon.log

# Последние события
tail -20 logs/daemon.log
```

### Linux/Mac: проверка процессов

```bash
# Все процессы демона
ps aux | grep console_app.py

# Подробная информация по камере
ps aux | grep "console_app.py.*cam101"

# Использование ресурсов
top -p <PID>
```

---

## 🔄 Параметры камер

### Понимание конфигурации

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|---------|
| `rtsp` | rtsp://localhost:8554/cam101 | URL видеопотока RTSP |
| `mode` | monitor | Режим работы (monitor, process, test) |
| `confidence` | 0.30 | Порог уверенности YOLO (0.0-1.0) |
| `min_pigs` | 3 | Минимум свиней для определения акта |
| `max_interval` | 30.0 | Макс интервал между свиньями (сек) |
| `continuous` | true | Непрерывный мониторинг |

### Настройка для конкретной камеры

Отредактируйте конфиг JSON:

```json
{
  "cam101": {
    "rtsp": "rtsp://192.168.1.100:554/cam101",
    "mode": "monitor",
    "confidence": 0.35,
    "min_pigs": 4,
    "max_interval": 25.0,
    "continuous": true
  }
}
```

Затем запустите:
```bash
python run_daemon.py --load-config custom_config.json --start --monitor
```

---

## 🐛 Решение проблем

### Демон не запускается

**Проверка 1: Доступ к RTSP**
```bash
# Проверить доступность потока
ffmpeg -rtsp_transport tcp -i rtsp://localhost:8554/cam101 -t 1 -f null -
```

**Проверка 2: Права доступа**
```bash
# Проверить права на файлы
ls -la console_app.py run_daemon.py
chmod +x console_app.py run_daemon.py
```

**Проверка 3: Python окружение**
```bash
# Активировать виртуальное окружение
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Демон часто перезапускается

**Возможные причины:**
1. Нестабильный видеопоток
2. Высокие требования к ресурсам (CPU/RAM)
3. Ошибки в обработке

**Решение:**
```bash
# Посмотреть подробные логи
tail -50 logs/daemon.log

# Запустить одну камеру в отладочном режиме
python console_app.py --rtsp rtsp://localhost:8554/cam101 --debug
```

### Слишком много памяти

**Решение 1: Уменьшить resolution**
```json
{
  "cam101": {
    "confidence": 0.35,  // Больше порог = быстрее, меньше памяти
    "max_interval": 20.0  // Меньше интервал = чаще очищает память
  }
}
```

**Решение 2: Перезагрузка по расписанию**
```bash
# В cron (Linux/Mac):
0 3 * * * pkill -f "console_app.py" && sleep 10 && python run_daemon.py --start --monitor
```

---

## 🔐 Безопасность и Надежность

### Переменные окружения для credentials

```env
# .env
RTSP_URL_CAM101=rtsp://user:password@192.168.1.10:554/cam101
RTSP_URL_CAM102=rtsp://user:password@192.168.1.11:554/cam102
```

### Логирование и аудит

Все события логируются:
```
logs/daemon.log - полные логи всех демонов
```

### Автоматический перезапуск

- Если демон упадёт, система попытается перезапустить его
- Максимум перезапусков: **10**
- Задержка перезапуска: **30 сек**
- После 10 перезапусков: демон отключается

---

## 📈 Продвинутое использование

### Запуск разных режимов для разных камер

Создайте свою конфигурацию:

```json
{
  "cam101": {
    "rtsp": "rtsp://localhost:8554/cam101",
    "mode": "monitor",
    "continuous": true
  },
  "cam102": {
    "rtsp": "rtsp://192.168.1.100:554/test",
    "mode": "process",
    "continuous": false
  }
}
```

### Integration с systemd (Linux)

Создайте `/etc/systemd/system/pigweight.service`:

```ini
[Unit]
Description=PigWeight Daemon
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/PigWeight
ExecStart=/usr/bin/python3 /path/to/PigWeight/run_daemon.py --start --monitor
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Включить:
```bash
sudo systemctl enable pigweight
sudo systemctl start pigweight
sudo systemctl status pigweight
```

### Integration с Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "run_daemon.py", "--start", "--monitor"]
```

Запуск:
```bash
docker build -t pigweight-daemon .
docker run -d --name pigweight-daemon pigweight-daemon
```

---

## 📊 Примеры использования

### Тестирование одной камеры

```bash
# Запустить обработку одного видеофайла (не демон)
python console_app.py --video test.mp4

# Или через интерактивное меню
python console_app.py
# → выбрать "Обработка видео"
```

### Сверка с Excel

```bash
# В режиме test (автоматическая сверка)
python console_app.py --mode test --video test.mp4 --excel reference.xlsx
```

### Запуск всех инструментов

```bash
# 1. Запустить демон в фоне
python run_daemon.py --start

# 2. В другом окне - веб-интерфейс
python main.py

# 3. В браузере
http://localhost:8000/mobile
```

---

## 🎯 Контроль-лист развертывания

- [ ] Установлены зависимости: `pip install -r requirements.txt`
- [ ] `.env` файл создан и заполнен
- [ ] RTSP URLs проверены и доступны
- [ ] Папка `logs/` создана: `mkdir -p logs`
- [ ] Запущен первый раз в интерактивном режиме: `python run_daemon.py --start --monitor`
- [ ] Проверены логи: `tail -f logs/daemon.log`
- [ ] Статус хороший (no errors)
- [ ] Веб-интерфейс доступен: `http://localhost:8000/mobile`
- [ ] Настроено автоматическое перезагрузка (systemd или cron)

---

## 📞 Поддержка

Если возникают проблемы:

1. **Проверьте логи:** `tail -f logs/daemon.log`
2. **Проверьте конфигурацию:** `cat daemon_config.json`
3. **Запустите в отладочном режиме:** `python console_app.py --debug --rtsp <url>`
4. **Свяжитесь с поддержкой:**
   - Email: support@pigweight.local
   - Documentation: `README.md`, `API_DOCUMENTATION.md`

---

**Версия:** 3.0  
**Дата:** Ноябрь 2025  
**Статус:** Production Ready ✅

