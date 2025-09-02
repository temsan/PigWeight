# PigWeight

Система для анализа и взвешивания свиней с использованием компьютерного зрения.

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/username/PigWeight.git
cd PigWeight

# Создание виртуального окружения
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Установка зависимостей
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

Приложение будет доступно по адресу: http://localhost:8000

## Структура проекта

- `api/` - API эндпоинты и бэкенд логика
- `models/` - Модели машинного обучения
- `static/` - Статические файлы (CSS, JS, изображения)
- `uploads/` - Директория для загруженных видео файлов
- `scripts/` - Вспомогательные скрипты
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