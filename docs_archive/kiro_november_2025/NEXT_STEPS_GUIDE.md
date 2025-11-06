# Руководство по следующим шагам

**Дата:** 6 ноября 2025  
**Статус:** 📋 Инструкции для продолжения работы

---

## 🎯 ТЕКУЩЕЕ СОСТОЯНИЕ

✅ **Завершено:**
- API интегрирован с БД (7 endpoints)
- Все критические баги исправлены
- Сервер запущен и работает
- Endpoints протестированы (без данных)

⏳ **Требуется:**
- Создать тестовые акты в БД
- Протестировать endpoints с реальными данными
- Обновить мобильный дашборд

---

## 📝 ШАГ 1: СОЗДАНИЕ ТЕСТОВЫХ АКТОВ

### Вариант А: Через console_app.py (рекомендуется)

```bash
# Запустить обработку видео с сохранением в БД
python console_app.py --video "uploads/2.mp4"

# Или в тестовом режиме (с автоматической сверкой)
python console_app.py --video "uploads/2.mp4" --test-mode
```

**Что произойдет:**
1. Видео будет обработано
2. Акты взвешивания будут обнаружены
3. Данные сохранятся в Supabase БД
4. Можно будет протестировать API endpoints

### Вариант Б: Через SQL (быстрый тест)

```sql
-- Подключиться к Supabase
-- http://localhost:54321

-- Вставить тестовый акт
INSERT INTO weighing_acts (
    started_at, 
    ended_at, 
    left_count, 
    right_count, 
    peak_count,
    total_weight,
    avg_weight
) VALUES (
    '2025-11-06 10:00:00',
    '2025-11-06 10:15:00',
    25,
    23,
    14,
    1850.5,
    132.2
);
```

---

## 📝 ШАГ 2: ТЕСТИРОВАНИЕ API С ДАННЫМИ

### 2.1 Проверить список актов

```powershell
# Получить акты за сегодня
Invoke-RestMethod -Uri "http://localhost:8000/api/weighing/acts" | ConvertTo-Json -Depth 5

# Ожидаемый результат:
# {
#   "acts": [
#     {
#       "id": 1,
#       "started_at": "2025-11-06T10:00:00",
#       "ended_at": "2025-11-06T10:15:00",
#       "left_count": 25,
#       "right_count": 23,
#       ...
#     }
#   ],
#   "total": 1
# }
```

### 2.2 Проверить статистику

```powershell
# Получить статистику
Invoke-RestMethod -Uri "http://localhost:8000/api/weighing/stats" | ConvertTo-Json -Depth 5

# Ожидаемый результат:
# {
#   "total_acts": 1,
#   "total_crossings": 48,
#   "total_weight": 1850.5,
#   "avg_weight": 38.6
# }
```

### 2.3 Проверить детали акта

```powershell
# Получить детали акта (замените 1 на реальный ID)
Invoke-RestMethod -Uri "http://localhost:8000/api/weighing/acts/1" | ConvertTo-Json -Depth 5

# Ожидаемый результат:
# {
#   "id": 1,
#   "started_at": "...",
#   "crossings": [...]
# }
```

### 2.4 Проверить последний акт

```powershell
# Получить последний акт
Invoke-RestMethod -Uri "http://localhost:8000/api/metrics/latest-act" | ConvertTo-Json -Depth 5

# Ожидаемый результат:
# {
#   "act": {
#     "id": 1,
#     ...
#   }
# }
```

---

## 📝 ШАГ 3: ТЕСТИРОВАНИЕ ЭКСПОРТА В EXCEL

### 3.1 Экспорт актов

```powershell
# Экспортировать акты за период
Invoke-WebRequest `
    -Uri "http://localhost:8000/api/export/excel?start_date=2025-11-01&end_date=2025-11-06" `
    -Method POST `
    -OutFile "test_export.xlsx"

# Проверить файл
ls test_export.xlsx
```

**Что проверить:**
- ✅ Файл создан
- ✅ Размер > 0
- ✅ Открывается в Excel
- ✅ Данные корректны
- ✅ Форматирование правильное

### 3.2 Проверить файл в uploads/

```powershell
# Посмотреть созданные файлы
ls uploads/export_*.xlsx
```

---

## 📝 ШАГ 4: ТЕСТИРОВАНИЕ СВЕРКИ

### 4.1 Подготовить тестовый Excel

Создать файл `manual_records.xlsx` с колонками:
- Дата
- Время начала
- Время окончания
- Количество (слева)
- Количество (справа)

### 4.2 Загрузить и сравнить

```powershell
# Сверка с ручными записями
$file = Get-Item "manual_records.xlsx"
$form = @{
    file = $file
    tolerance_minutes = 5
}

Invoke-RestMethod `
    -Uri "http://localhost:8000/api/export/compare" `
    -Method POST `
    -Form $form | ConvertTo-Json -Depth 5
```

**Ожидаемый результат:**
```json
{
  "status": "success",
  "metrics": {...},
  "summary": {
    "matches": 5,
    "discrepancies": 2,
    "missing_in_auto": 0,
    "missing_in_manual": 1
  },
  "report_file": "comparison_20251106_123456.xlsx"
}
```

### 4.3 Скачать отчет

```powershell
# Скачать отчет о сверке
Invoke-WebRequest `
    -Uri "http://localhost:8000/api/export/download/comparison_20251106_123456.xlsx" `
    -OutFile "comparison_report.xlsx"
```

---

## 📝 ШАГ 5: ОБНОВЛЕНИЕ МОБИЛЬНОГО ДАШБОРДА

### 5.1 Найти файл дашборда

```bash
# Мобильный дашборд
static/mobile-dashboard.html
```

### 5.2 Обновить JavaScript

**Добавить функции для новых endpoints:**

```javascript
// Получить акты за сегодня
async function loadActs() {
    const response = await fetch('/api/weighing/acts');
    const data = await response.json();
    displayActs(data.acts);
}

// Получить статистику
async function loadStats() {
    const response = await fetch('/api/weighing/stats');
    const data = await response.json();
    displayStats(data);
}

// Экспорт в Excel
async function exportToExcel() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    const url = `/api/export/excel?start_date=${startDate}&end_date=${endDate}`;
    window.location.href = url;
}
```

### 5.3 Добавить UI элементы

```html
<!-- Кнопка экспорта -->
<button onclick="exportToExcel()">📊 Экспорт в Excel</button>

<!-- Список актов -->
<div id="acts-list"></div>

<!-- Статистика -->
<div id="stats-panel"></div>
```

---

## 📝 ШАГ 6: ОПЦИОНАЛЬНЫЕ ЗАДАЧИ

### 6.1 Фаза 5: Организация scripts/

```bash
# Создать структуру
mkdir scripts/setup scripts/tests scripts/utils scripts/training scripts/deprecated

# Распределить скрипты
move scripts/check_cuda.py scripts/setup/
move scripts/test_*.py scripts/tests/
# и т.д.
```

### 6.2 Фаза 6: Переименование процессоров

```python
# core/processor.py
class YOLODetectionProcessor:  # было: UnifiedVideoProcessor
    """Базовый процессор для YOLO детекции"""
    pass

# pig_tracking/video_processor.py
class PigTrackingPipeline:  # было: IntegratedVideoProcessor
    """Полный пайплайн: детекция + трекинг + подсчет"""
    pass
```

---

## 🔍 ПРОВЕРОЧНЫЙ ЧЕКЛИСТ

### API Endpoints
- [ ] GET /api/weighing/acts - работает с данными
- [ ] GET /api/weighing/stats - показывает статистику
- [ ] GET /api/weighing/acts/{id} - возвращает детали
- [ ] POST /api/export/excel - создает файл
- [ ] POST /api/export/compare - выполняет сверку
- [ ] GET /api/export/download/{filename} - скачивает файл
- [ ] GET /api/metrics/latest-act - возвращает последний акт

### Файлы
- [ ] Экспорт создается в uploads/
- [ ] Файлы открываются в Excel
- [ ] Данные корректны
- [ ] Форматирование правильное

### Мобильный дашборд
- [ ] Подключен к новым endpoints
- [ ] Отображает список актов
- [ ] Показывает статистику
- [ ] Кнопка экспорта работает

---

## 📚 ПОЛЕЗНЫЕ КОМАНДЫ

### Запуск сервисов

```bash
# Docker (Supabase)
docker-compose up -d

# API сервер
python main.py api

# Консольное приложение
python console_app.py --video "uploads/2.mp4"
```

### Проверка состояния

```bash
# Проверить Docker
docker ps

# Проверить API
curl http://localhost:8000/api/health

# Проверить БД
psql -h localhost -p 5432 -U postgres -d postgres
```

### Логи

```bash
# Логи Docker
docker-compose logs -f

# Логи API (в консоли где запущен)
# Или в файле logs/api.log
```

---

## 🎯 КРИТЕРИИ УСПЕХА

### Минимальные требования
- ✅ API возвращает акты из БД
- ✅ Экспорт в Excel работает
- ✅ Файлы создаются в uploads/

### Полная готовность
- ✅ Все endpoints протестированы с данными
- ✅ Мобильный дашборд обновлен
- ✅ Сверка с Excel работает
- ✅ Документация актуальна

---

## 📞 ПОМОЩЬ

### Документация
- API: http://localhost:8000/docs
- Проект: .kiro/COMPLETE_SESSION_SUMMARY.md

### Проблемы
- Сервер не запускается → проверить Docker
- Нет данных → запустить console_app.py
- Ошибки API → проверить логи

---

**Статус:** 📋 Готово к выполнению  
**Приоритет:** Высокий (Шаги 1-3), Средний (Шаги 4-5), Низкий (Шаг 6)  
**Время:** ~3-4 часа для полного тестирования
