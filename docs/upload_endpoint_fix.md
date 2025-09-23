# Исправление проблемы с загрузкой файлов

## Проблема

Пользователи получали ошибку "Неизвестная ошибка" при попытке загрузить видеофайлы через веб-интерфейс.

## Причина

Обнаружены множественные конфликтующие endpoints для загрузки файлов:

1. **Дублированные endpoints в app.py**: Два идентичных `/api/upload` endpoint'а в основном файле приложения
2. **Конфликт модульных router'ов**: 
   - `api/endpoints/video.py` - `/api/upload` 
   - `api/endpoints/files.py` - `/api/upload`
   - `api/simple_endpoints.py` - `/api/video/upload`
3. **Middleware конфликты**: Неправильная обработка ошибок из-за множественных обработчиков

## Решение

### 1. Устранение дублированных endpoints

**Отключены дублированные endpoints в app.py:**
```python
# === API для загрузки видеофайлов ===
# ОТКЛЮЧЕН: используется модульный endpoint из api/endpoints/video.py

# @app.post("/api/upload")
# async def upload_video_file_disabled1(file: UploadFile = File(...)):
```

### 2. Разрешение конфликтов router'ов

**Изменены пути endpoints:**
- `api/endpoints/files.py` - `/api/upload` (основной для фронтенда)
- `api/endpoints/video.py` - `/api/video/upload` (альтернативный)
- `api/simple_endpoints.py` - отключен

### 3. Улучшение middleware

**Добавлены новые middleware:**
- **CORS**: Гибкая настройка через переменные окружения
- **Error Handling**: Централизованная обработка с уникальными ID ошибок
- **Request Logging**: Логирование всех HTTP запросов
- **Security Headers**: Заголовки безопасности

## Текущая архитектура

### Активные upload endpoints

1. **Основной endpoint**: `/api/upload`
   - Файл: `api/endpoints/files.py`
   - Используется фронтендом
   - Полная валидация и обработка ошибок

2. **Альтернативный endpoint**: `/api/video/upload`
   - Файл: `api/endpoints/video.py`
   - Резервный endpoint

### Middleware Stack

```python
# Порядок выполнения middleware:
1. Security Headers - добавляет заголовки безопасности
2. Request Logging - логирует запросы и время выполнения  
3. CORS - обрабатывает CORS заголовки
4. Error Handling - обрабатывает исключения
```

## Конфигурация

### Переменные окружения для CORS

```bash
# Ограничить домены в продакшене
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CORS_METHODS=GET,POST,PUT,DELETE
CORS_HEADERS=*
CORS_CREDENTIALS=true

# Режим отладки для детальных ошибок
DEBUG=true
```

## Валидация файлов

### Поддерживаемые форматы
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`, `.flv`, `.wmv`

### Ограничения
- Максимальный размер: 500MB
- Обязательное имя файла
- Проверка на пустые файлы

## Обработка ошибок

### Типы ошибок
- **400**: Неверный формат файла, пустой файл, отсутствие имени
- **413**: Файл слишком большой
- **500**: Внутренние ошибки сервера

### Структура ответа об ошибке
```json
{
  "error": "Описание ошибки",
  "error_id": "ERR-1234",
  "path": "/api/upload"
}
```

## Логирование

### Успешная загрузка
```
📁 Video uploaded: 20250922_174500_video.mp4, size: 15.2MB, duration: 30.5s
```

### Ошибки
```
❌ Error uploading video: [ERR-1234] Permission denied
```

## Тестирование

Для проверки работы endpoint'а:

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test_video.mp4" \
  -H "Accept: application/json"
```

## Мониторинг

- Все запросы логируются с временем выполнения
- Ошибки получают уникальные ID для отслеживания
- Middleware добавляет заголовок `X-Process-Time`
- Security headers автоматически добавляются ко всем ответам