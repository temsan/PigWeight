# Работа с медиа файлами

## Игнорируемые файлы

Следующие типы файлов **НЕ** добавляются в Git репозиторий:

### Видео файлы
- `*.mp4`, `*.avi`, `*.mov`, `*.mkv`, `*.flv`, `*.wmv`, `*.webm`
- `*.m4v`, `*.mpg`, `*.mpeg`, `*.3gp`

### Модели машинного обучения
- `*.pt`, `*.onnx`, `*.pth`, `*.weights`, `*.h5`, `*.pb`

### Аудио файлы
- `*.mp3`, `*.wav`, `*.flac`, `*.aac`, `*.ogg`, `*.wma`, `*.m4a`

### Большие изображения
- `*.psd`, `*.ai`, `*.raw`, `*.cr2`, `*.nef`, `*.dng`

## Структура папок

```
project/
├── uploads/          # Видеофайлы для обработки (игнорируются)
├── models/           # Модели ML (игнорируются)
├── records/          # Записи актов (JSON, SVG, MD - сохраняются)
├── results/          # Результаты обработки (игнорируются)
└── test_results/     # Результаты тестов (игнорируются)
```

## Где хранить медиа файлы

### Локальная разработка
Храните видео и модели локально в соответствующих папках:
- `uploads/` - для видеофайлов
- `models/` - для моделей YOLO

### Продакшн
Используйте внешнее хранилище:
- **Видео**: S3, Google Cloud Storage, или локальный NAS
- **Модели**: Git LFS, S3, или специализированные хранилища моделей

## Как добавить модель в проект

### Вариант 1: Скачать вручную
```bash
# Скачайте модель и поместите в папку models/
wget https://example.com/model.pt -O models/pig_yolo11-seg.v4.pt
```

### Вариант 2: Git LFS (для больших файлов)
```bash
# Установите Git LFS
git lfs install

# Отслеживайте конкретную модель
git lfs track "models/pig_yolo11-seg.v4.pt"

# Добавьте и закоммитьте
git add .gitattributes models/pig_yolo11-seg.v4.pt
git commit -m "Add model via Git LFS"
```

### Вариант 3: Скрипт загрузки
Создайте скрипт `download_models.py`:
```python
import urllib.request
from pathlib import Path

MODELS = {
    "pig_yolo11-seg.v4.pt": "https://example.com/models/pig_yolo11-seg.v4.pt"
}

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

for filename, url in MODELS.items():
    filepath = models_dir / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ {filename} downloaded")
```

## Проверка игнорируемых файлов

```bash
# Проверить, какие файлы будут игнорироваться
git status --ignored

# Проверить конкретный файл
git check-ignore -v uploads/video.mp4

# Посмотреть размер репозитория
git count-objects -vH
```

## Очистка случайно добавленных файлов

Если медиа файлы уже попали в репозиторий:

```bash
# Удалить из индекса, но оставить локально
git rm --cached uploads/*.mp4
git rm --cached models/*.pt

# Удалить из истории (осторожно!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch uploads/*.mp4" \
  --prune-empty --tag-name-filter cat -- --all

# Или используйте BFG Repo-Cleaner (рекомендуется)
# https://rtyley.github.io/bfg-repo-cleaner/
```

## Рекомендации

1. **Никогда не коммитьте** видеофайлы и большие модели напрямую
2. **Используйте** `.gitkeep` для сохранения структуры пустых папок
3. **Документируйте** где и как получить необходимые медиа файлы
4. **Используйте** переменные окружения для путей к файлам
5. **Рассмотрите** Git LFS для моделей, если они часто обновляются

## Пример .env для путей

```env
# Пути к медиа файлам
UPLOADS_DIR=./uploads
MODELS_DIR=./models
RESULTS_DIR=./results

# Модель
MODEL_PATH=models/pig_yolo11-seg.v4.pt

# Внешнее хранилище (опционально)
S3_BUCKET=my-pig-tracking-videos
S3_MODELS_PREFIX=models/
```

## Размеры файлов

Типичные размеры для справки:
- Видео (1080p, 1 час): ~2-5 GB
- Модель YOLO11-seg: ~50-100 MB
- Результаты обработки (JSON): ~1-10 MB

**Итого**: Один проект с видео может занимать 5-10 GB, поэтому важно правильно настроить `.gitignore`.
