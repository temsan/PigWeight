# Исправление проблем CSP и зависимостей

## Проблемы

1. **CSP блокирует внешние скрипты**: Chart.js и HLS.js не загружались из CDN
2. **Chart is not defined**: Ошибка при инициализации графиков
3. **Неправильная проверка зависимостей**: opencv-python и pillow показывались как отсутствующие

## Исправления

### 1. Обновление Content Security Policy

**Файл**: `api/middleware/security.py`

**Было**:
```python
csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; ..."
```

**Стало**:
```python
csp = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; ..."
```

**Изменения**:
- Добавлен `'unsafe-eval'` для поддержки Chart.js
- Добавлен `https://cdn.jsdelivr.net` для загрузки CDN скриптов
- Обновлен `style-src` для поддержки CDN стилей

### 2. Защита от ошибок Chart.js

**Файл**: `static/index.html`

**Добавлена проверка загрузки**:
```javascript
function initChart() {
    // Проверяем, загружен ли Chart.js
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js не загружен, график недоступен');
        return;
    }
    
    // Остальной код...
}
```

**Улучшенная загрузка Chart.js**:
```javascript
function loadChartJS() {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
    script.onload = function() {
        console.log('Chart.js загружен успешно');
    };
    script.onerror = function() {
        console.warn('Не удалось загрузить Chart.js из CDN');
    };
    document.head.appendChild(script);
}
```

### 3. Исправление проверки зависимостей

**Файл**: `api/endpoints/diagnostics.py`

**Было**:
```python
required_packages = [
    "fastapi", "uvicorn", "opencv-python", "torch", 
    "ultralytics", "numpy", "pillow", "psutil"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        deps["dependencies"][package] = "installed"
    except ImportError:
        deps["dependencies"][package] = "missing"
```

**Стало**:
```python
required_packages = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"), 
    ("opencv-python", "cv2"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("psutil", "psutil")
]

for package_name, import_name in required_packages:
    try:
        __import__(import_name)
        deps["dependencies"][package_name] = "installed"
    except ImportError:
        deps["dependencies"][package_name] = "missing"
```

## Результаты исправлений

### До исправлений:
- ❌ CSP блокировал Chart.js и HLS.js
- ❌ Ошибка "Chart is not defined"
- ❌ Неправильное определение opencv-python и pillow как отсутствующих
- ❌ Графики не отображались

### После исправлений:
- ✅ Chart.js и HLS.js загружаются из CDN
- ✅ Графики инициализируются без ошибок
- ✅ Правильное определение установленных зависимостей
- ✅ Graceful fallback при проблемах с загрузкой

## Тестирование

### Автоматическое тестирование
Создана страница `/static/test-fixes.html` для проверки исправлений:

1. **Тест Chart.js**: Проверяет загрузку и создание графиков
2. **Тест CSP**: Проверяет возможность загрузки внешних скриптов
3. **Тест диагностики**: Проверяет работу API диагностики

### Ручное тестирование
1. Откройте главную страницу приложения
2. Проверьте отсутствие ошибок в консоли браузера
3. Убедитесь, что графики отображаются корректно
4. Запустите диагностику через кнопку 🔧

## Безопасность

### CSP компромиссы
Разрешение `'unsafe-eval'` и внешних CDN снижает безопасность, но необходимо для работы Chart.js.

**Рекомендации для продакшена**:
1. Используйте локальные копии Chart.js и HLS.js
2. Настройте более строгий CSP с конкретными доменами
3. Реализуйте Subresource Integrity (SRI) для CDN ресурсов

### Пример строгого CSP для продакшена:
```python
csp = """
    default-src 'self'; 
    script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js; 
    style-src 'self' 'unsafe-inline'; 
    img-src 'self' data: blob:; 
    media-src 'self' blob:; 
    connect-src 'self' ws: wss:;
"""
```

## Мониторинг

### Логирование загрузки ресурсов
```javascript
script.onload = function() {
    console.log('Chart.js загружен успешно');
};
script.onerror = function() {
    console.warn('Не удалось загрузить Chart.js из CDN');
    // Отправить метрику о проблеме загрузки
};
```

### Метрики для отслеживания
- Успешность загрузки Chart.js
- Время загрузки внешних ресурсов
- Частота ошибок CSP
- Использование fallback механизмов

## Дальнейшие улучшения

1. **Локальные копии библиотек**: Разместить Chart.js и HLS.js локально
2. **Service Worker**: Кэширование внешних ресурсов
3. **Lazy loading**: Загрузка Chart.js только при необходимости
4. **Альтернативные библиотеки**: Рассмотреть более легкие альтернативы Chart.js