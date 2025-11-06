# Middleware Configuration

Система middleware для API PigWeight обеспечивает централизованную обработку запросов, ошибок, безопасности и логирования.

## Структура middleware

```
api/middleware/
├── __init__.py          # Экспорт всех middleware
├── cors.py             # CORS конфигурация
├── error.py            # Обработка ошибок
├── logging.py          # Логирование запросов
└── security.py         # Заголовки безопасности
```

## Компоненты

### 1. CORS Middleware (`cors.py`)

Настраивает Cross-Origin Resource Sharing для веб-приложения.

**Переменные окружения:**
- `CORS_ORIGINS` - разрешенные домены (по умолчанию: "*")
- `CORS_METHODS` - разрешенные HTTP методы (по умолчанию: "*")
- `CORS_HEADERS` - разрешенные заголовки (по умолчанию: "*")
- `CORS_CREDENTIALS` - разрешить credentials (по умолчанию: "true")

**Пример конфигурации:**
```bash
CORS_ORIGINS=http://localhost:3000,https://myapp.com
CORS_METHODS=GET,POST,PUT,DELETE
CORS_CREDENTIALS=true
```

### 2. Error Handling Middleware (`error.py`)

Централизованная обработка всех типов ошибок.

**Обрабатываемые типы ошибок:**
- `HTTPException` - HTTP ошибки FastAPI
- `StarletteHTTPException` - HTTP ошибки Starlette
- `RequestValidationError` - ошибки валидации запросов
- `Exception` - все остальные исключения

**Особенности:**
- Генерация уникальных ID ошибок для отслеживания
- Детальная информация в debug режиме
- Структурированное логирование

### 3. Request Logging Middleware (`logging.py`)

Логирование всех HTTP запросов с метриками производительности.

**Логируемая информация:**
- Входящие запросы (метод, путь, IP клиента)
- Исходящие ответы (статус, время выполнения)
- Заголовок `X-Process-Time` в ответе

### 4. Security Headers Middleware (`security.py`)

Добавляет заголовки безопасности ко всем ответам.

**Добавляемые заголовки:**
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy` - базовая CSP политика
- `Strict-Transport-Security` - только в продакшене

## Интеграция

Middleware автоматически подключаются в `api/app.py`:

```python
# Setup middleware
from api.middleware import setup_cors, setup_error_handling, setup_request_logging, setup_security_headers

setup_cors(app)
setup_error_handling(app)
setup_request_logging(app)
setup_security_headers(app)
```

## Порядок выполнения

Middleware выполняются в следующем порядке:

1. **Security Headers** - добавляет заголовки безопасности
2. **Request Logging** - логирует запросы и время выполнения
3. **CORS** - обрабатывает CORS заголовки
4. **Error Handling** - обрабатывает исключения (exception handlers)

## Конфигурация для продакшена

Для продакшена рекомендуется:

```bash
# Ограничить CORS домены
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CORS_CREDENTIALS=true

# Включить HTTPS
ENVIRONMENT=production

# Отключить debug режим
DEBUG=false
```

## Мониторинг

Все middleware логируют свою активность:

- **CORS**: информация о конфигурации при запуске
- **Error Handling**: все ошибки с уникальными ID
- **Request Logging**: все HTTP запросы с временем выполнения
- **Security**: подтверждение настройки при запуске

## Расширение

Для добавления нового middleware:

1. Создайте файл в `api/middleware/`
2. Реализуйте функцию `setup_*`
3. Добавьте импорт в `__init__.py`
4. Подключите в `api/app.py`

Пример:

```python
# api/middleware/custom.py
def setup_custom_middleware(app: FastAPI):
    @app.middleware("http")
    async def custom_middleware(request: Request, call_next):
        # Ваша логика
        response = await call_next(request)
        return response
```