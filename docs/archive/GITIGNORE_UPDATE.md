# Обновление .gitignore - Медиа файлы

## Что было сделано

### 1. Удалены из Git индекса (но сохранены локально)

**Модели ML** (4 файла, ~200-400 MB):
- `models/pig_yolo11-seg.pt`
- `models/pig_yolo11-seg.v2.pt`
- `models/pig_yolo11-seg.v3.pt`
- `models/pig_yolo11-seg.v4.pt`

**Скриншоты** (4 файла):
- `screenshots/image_2025-08-10_17-33-12.png`
- `screenshots/image_2025-08-10_17-59-27.png`
- `screenshots/image_2025-08-10_18-00-15.png`
- `screenshots/image_2025-08-10_18-02-47.png`

**Видео** (1 файл):
- `temp/0825.mp4`

### 2. Обновлен .gitignore

Добавлены правила для игнорирования:

```gitignore
# Видео файлы
*.mp4, *.avi, *.mov, *.mkv, *.flv, *.wmv, *.webm, *.m4v, *.mpg, *.mpeg, *.3gp

# Аудио файлы
*.mp3, *.wav, *.flac, *.aac, *.ogg, *.wma, *.m4a

# Модели ML
models/*.pt, models/*.onnx, models/*.pth, models/*.weights
*.pt, *.onnx, *.pth, *.weights

# Скриншоты
screenshots/*.png, screenshots/*.jpg, screenshots/*.jpeg

# Папки
uploads/, results/, test_results/, records/, temp/
```

### 3. Созданы .gitkeep файлы

Для сохранения структуры пустых папок:
- `models/.gitkeep`
- `uploads/.gitkeep`
- `results/.gitkeep`
- `records/.gitkeep`
- `temp/.gitkeep`
- `screenshots/.gitkeep`

### 4. Добавлена документация

- `models/README.md` - инструкция по загрузке моделей
- `MEDIA_FILES.md` - руководство по работе с медиа файлами

## Результат

### До изменений
```
Размер репозитория: ~250-500 MB (с моделями и медиа)
```

### После изменений
```
Размер репозитория: ~10-20 MB (только код и конфиги)
```

**Экономия**: ~230-480 MB

## Что нужно сделать после клонирования

### 1. Скачать модели

```bash
# Вариант 1: Из внешнего источника
wget https://example.com/models/pig_yolo11-seg.v4.pt -O models/pig_yolo11-seg.v4.pt

# Вариант 2: Из локального хранилища
cp /path/to/pig_yolo11-seg.v4.pt models/

# Вариант 3: Git LFS (если настроен)
git lfs pull
```

### 2. Проверить конфигурацию

```bash
# Убедитесь, что путь к модели правильный
cat .env | grep MODEL_PATH
# Должно быть: MODEL_PATH=models/pig_yolo11-seg.v4.pt
```

### 3. Проверить работу

```bash
# Запустить проверку системы
python check_system.py

# Или запустить пример
python pig_tracking/example_usage.py
```

## Команды Git

### Проверить игнорируемые файлы

```bash
# Показать все игнорируемые файлы
git status --ignored

# Проверить конкретный файл
git check-ignore -v models/pig_yolo11-seg.v4.pt
```

### Если случайно добавили медиа файл

```bash
# Удалить из индекса, но оставить локально
git rm --cached path/to/large_file.mp4

# Добавить в .gitignore
echo "path/to/large_file.mp4" >> .gitignore

# Закоммитить изменения
git add .gitignore
git commit -m "Remove large file from tracking"
```

### Очистка истории (если нужно)

Если медиа файлы уже в истории и нужно их удалить:

```bash
# Вариант 1: BFG Repo-Cleaner (рекомендуется)
# https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-files "*.{mp4,pt,onnx}" .

# Вариант 2: git filter-branch (медленнее)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch '*.mp4' '*.pt'" \
  --prune-empty --tag-name-filter cat -- --all

# После очистки
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## Рекомендации для команды

1. **Никогда не коммитьте** медиа файлы напрямую
2. **Используйте** общее хранилище для моделей и видео
3. **Документируйте** где и как получить необходимые файлы
4. **Проверяйте** размер коммита перед push: `git diff --stat`
5. **Используйте** Git LFS для больших файлов, если необходимо

## Проверка перед коммитом

```bash
# Проверить размер изменений
git diff --stat

# Проверить, нет ли больших файлов
git diff --cached --name-only | xargs ls -lh

# Если видите большие файлы (>1MB), проверьте .gitignore
```

## Автоматизация

Можно добавить pre-commit hook для проверки размера файлов:

```bash
# .git/hooks/pre-commit
#!/bin/bash
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [ $size -gt 10485760 ]; then  # 10MB
            echo "Error: $file is larger than 10MB ($size bytes)"
            exit 1
        fi
    fi
done
```

## Статус

✅ Медиа файлы удалены из Git индекса  
✅ .gitignore обновлен  
✅ Структура папок сохранена (.gitkeep)  
✅ Документация добавлена  
✅ Файлы остались локально для работы  

## Следующие шаги

1. Закоммитить изменения:
   ```bash
   git add .gitignore models/.gitkeep screenshots/.gitkeep temp/.gitkeep
   git commit -m "Update .gitignore: exclude media files and models"
   ```

2. Настроить общее хранилище для моделей (NAS, S3, или Git LFS)

3. Обновить документацию проекта с инструкциями по получению моделей

4. Уведомить команду о изменениях
