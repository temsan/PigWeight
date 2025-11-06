# Новые API Endpoints - Интеграция с БД

**Дата:** 6 ноября 2025  
**Статус:** ✅ Реализовано  
**Цель:** Интеграция API с DatabaseManager для работы с актами взвешивания

---

## 📋 ДОБАВЛЕННЫЕ ENDPOINTS

### 1. Weighing Acts API (`/api/weighing`)

#### GET `/api/weighing/acts`
Получить список актов взвешивания за период

**Query Parameters:**
- `start_date` (optional): начальная дата (YYYY-MM-DD), по умолчанию сегодня
- `end_date` (optional): конечная дата (YYYY-MM-DD), по умолчанию сегодня
- `limit` (optional): максимальное количество актов (1-1000), по умолчанию 100

**Response:**
```json
{
  "acts": [
    {
      "id": 1,
      "started_at": "2025-11-06T12:30:00",
      "ended_at": "2025-11-06T12:45:30",
      "duration_sec": 930,
      "left_count": 25,
      "right_count": 23,
      "peak_count": 14,
      "total_weight": 1850.5,
      "avg_weight": 132.2
    }
  ],
  "total": 1,
  "start_date": "2025-11-06",
  "end_date": "2025-11-06"
}
```

---

#### GET `/api/weighing/stats`
Получить статистику по актам за период

**Query Parameters:**
- `start_date` (optional): начальная дата (YYYY-MM-DD)
- `end_date` (optional): конечная дата (YYYY-MM-DD)

**Response:**
```json
{
  "period": {
    "start": "2025-11-06",
    "end": "2025-11-06"
  },
  "total_acts": 5,
  "total_pigs": 240,
  "total_weight": 9252.5,
  "avg_weight": 38.6,
  "avg_duration_sec": 850,
  "by_day": [
    {
      "date": "2025-11-06",
      "acts": 5,
      "pigs": 240,
      "weight": 9252.5
    }
  ]
}
```

---

#### GET `/api/weighing/acts/{act_id}`
Получить детальную информацию об акте

**Path Parameters:**
- `act_id`: ID акта

**Response:**
```json
{
  "id": 1,
  "started_at": "2025-11-06T12:30:00",
  "ended_at": "2025-11-06T12:45:30",
  "duration_sec": 930,
  "left_count": 25,
  "right_count": 23,
  "peak_count": 14,
  "total_weight": 1850.5,
  "avg_weight": 132.2,
  "crossings": [
    {
      "id": 1,
      "timestamp": "2025-11-06T12:30:05",
      "direction": "left",
      "pig_id": 1,
      "x": 0.25,
      "y": 0.45
    }
  ]
}
```

---

### 2. Export API (`/api/export`)

#### POST `/api/export/excel`
Экспорт актов в Excel

**Query Parameters:**
- `start_date` (required): начальная дата (YYYY-MM-DD)
- `end_date` (required): конечная дата (YYYY-MM-DD)
- `output_path` (optional): путь для сохранения файла

**Response:** Excel файл для скачивания

---

#### POST `/api/export/compare`
Сверка с ручными записями из Excel

**Query Parameters:**
- `manual_excel_path` (required): путь к файлу с ручными записями
- `start_date` (required): начальная дата (YYYY-MM-DD)
- `end_date` (required): конечная дата (YYYY-MM-DD)
- `output_path` (optional): путь для сохранения отчета

**Response:** Excel файл с результатами сверки (цветовое выделение)

---

### 3. Metrics API - Дополнения (`/api/metrics`)

#### GET `/api/metrics/latest-act`
Получить последний завершенный акт из БД

**Response:**
```json
{
  "act": {
    "id": 1,
    "started_at": "2025-11-06T12:30:00",
    "ended_at": "2025-11-06T12:45:30",
    "duration_sec": 930,
    "left_count": 25,
    "right_count": 23,
    "peak_count": 14,
    "total_weight": 1850.5,
    "avg_weight": 132.2
  },
  "timestamp": "2025-11-06T13:00:00"
}
```

---

## 🔧 ТЕХНИЧЕСКИЕ ДЕТАЛИ

### Файлы

**Новые endpoints:**
- `api/endpoints/weighing.py` - CRUD для актов взвешивания
- `api/endpoints/export.py` - экспорт и сверка Excel

**Обновленные файлы:**
- `api/app.py` - подключены новые роутеры
- `api/dependencies.py` - добавлена функция `get_database_manager()`
- `api/endpoints/metrics.py` - добавлен endpoint `/api/metrics/latest-act`

### Зависимости

**DatabaseManager:**
```python
from api.dependencies import get_database_manager

db = get_database_manager()
acts = db.get_acts_by_period(start_date, end_date)
```

**ExcelExporter:**
```python
from pig_tracking.excel_exporter import ExcelExporter

exporter = ExcelExporter(db)
exporter.export_to_excel(start_date, end_date, output_path)
```

**ExcelComparator:**
```python
from pig_tracking.excel_comparator import ExcelComparator

comparator = ExcelComparator(db)
comparator.compare_and_generate_report(manual_excel_path, start_date, end_date, output_path)
```

---

## ✅ ВЫПОЛНЕННЫЕ ТРЕБОВАНИЯ

- ✅ Интеграция API с DatabaseManager (Фаза 4, Задача 4.1)
- ✅ CRUD endpoints для актов взвешивания
- ✅ Статистика по актам за период
- ✅ Экспорт в Excel через API
- ✅ Сверка с ручными записями через API
- ✅ Получение последнего акта из БД

---

## 🚀 ИСПОЛЬЗОВАНИЕ

### Запуск API сервера

```bash
# Убедитесь, что .env настроен
# SUPABASE_URL=http://localhost:54321
# SUPABASE_KEY=your-key

# Запустите сервер
python -m api.app
```

### Примеры запросов

**Получить акты за сегодня:**
```bash
curl "http://localhost:8000/api/weighing/acts"
```

**Получить статистику за период:**
```bash
curl "http://localhost:8000/api/weighing/stats?start_date=2025-11-01&end_date=2025-11-06"
```

**Экспорт в Excel:**
```bash
curl -X POST "http://localhost:8000/api/export/excel?start_date=2025-11-01&end_date=2025-11-06" \
  --output weighing_acts.xlsx
```

**Последний акт:**
```bash
curl "http://localhost:8000/api/metrics/latest-act"
```

---

## 📝 СЛЕДУЮЩИЕ ШАГИ

1. ✅ Интеграция API с БД - ЗАВЕРШЕНО
2. ⏳ Обновить мобильный дашборд для использования новых endpoints
3. ⏳ Добавить WebSocket уведомления о новых актах
4. ⏳ Тестирование всех endpoints

---

**Статус:** ✅ Фаза 4 (Интеграция API с БД) завершена  
**Прогресс:** 4/6 фаз (67%)
