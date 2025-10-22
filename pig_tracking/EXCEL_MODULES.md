# Модули работы с Excel

## 📦 Созданные модули (Задачи 9-11)

### 1. ExcelAnalyzer - Анализ Excel файлов
**Файл:** `excel_analyzer.py`

**Функции:**
- Парсинг Excel файлов (openpyxl или pandas)
- Определение структуры столбцов
- Извлечение данных: секция, дата, вес, количество
- Анализ схемы шаблона
- Сохранение схемы в JSON

**Пример использования:**
```python
from pig_tracking import ExcelAnalyzer

# Создание анализатора
analyzer = ExcelAnalyzer("reference.xlsx")

# Загрузка и анализ
analyzer.load()
schema = analyzer.analyze_schema()
data = analyzer.parse_data()

# Сводка
summary = analyzer.get_summary()
print(f"Записей: {summary['total_records']}")

# Сохранение схемы
analyzer.save_schema("schema.json")
```

---

### 2. ExcelExporter - Экспорт в Excel
**Файл:** `excel_exporter.py`

**Функции:**
- Группировка актов по дате
- Суммирование показателей за день
- Создание Excel файла с форматированием
- Применение стилей (заголовки, цвета, границы)
- Автоширина столбцов

**Пример использования:**
```python
from pig_tracking import ExcelExporter

# Создание экспортера
exporter = ExcelExporter()

# Группировка по датам
grouped = exporter.group_acts_by_date(acts)
summaries = exporter.summarize_by_date(grouped)

# Экспорт
exporter.export_to_excel(
    acts=acts,
    output_path="export.xlsx",
    group_by_date=True
)
```

---

### 3. ExcelComparator - Сверка с ручными записями
**Файл:** `excel_comparator.py`

**Функции:**
- Сопоставление актов по времени (±5 минут)
- Сравнение показателей (количество, вес)
- Вычисление метрик: Recall, Precision, F1, MAE, MAPE
- Генерация отчета с 4 листами:
  - Совпадения (зеленый)
  - Расхождения (желтый/красный)
  - Пропущенные (серый)
  - Метрики

**Пример использования:**
```python
from pig_tracking import ExcelComparator

# Создание компаратора
comparator = ExcelComparator(
    time_tolerance_minutes=5.0,
    count_tolerance_percent=10.0
)

# Сопоставление
results = comparator.match_acts_by_time(
    auto_acts=auto_acts,  # из БД
    manual_acts=manual_acts  # из Excel
)

# Метрики
metrics = comparator.calculate_metrics()
print(f"Recall: {metrics['recall']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")

# Отчет
comparator.generate_report("comparison_report.xlsx")
```

---

## 🔄 Полный пример: Сверка с Excel

```python
from pig_tracking import ExcelAnalyzer, ExcelComparator
from pig_tracking.database import DatabaseManager

# 1. Загрузка ручных записей из Excel
analyzer = ExcelAnalyzer("manual_records.xlsx")
analyzer.load()
manual_acts = analyzer.parse_data()

# 2. Получение автоматических актов из БД
db = DatabaseManager()
auto_acts = db.get_acts_by_period(
    date_from="2025-01-01",
    date_to="2025-01-31"
)

# 3. Сверка
comparator = ExcelComparator()
results = comparator.match_acts_by_time(auto_acts, manual_acts)

# 4. Метрики
metrics = comparator.calculate_metrics()
print(f"""
Результаты сверки:
- Точных совпадений: {metrics['exact_matches']}
- Близких совпадений: {metrics['close_matches']}
- Расхождений: {metrics['mismatches']}
- Recall: {metrics['recall']:.2%}
- Precision: {metrics['precision']:.2%}
""")

# 5. Генерация отчета
comparator.generate_report("comparison_report.xlsx")
```

---

## 📊 Формат данных

### Входные данные (автоматические акты):
```python
{
    'id': 1,
    'started_at': '2025-01-15T10:30:00',
    'ended_at': '2025-01-15T10:30:25',
    'left_count': 15,
    'right_count': 14,
    'peak_count': 8,
    'duration': 25.3,
    'seen_total': 20
}
```

### Входные данные (ручные записи):
```python
{
    'date': '2025-01-15T10:30:00',
    'section': 'A',
    'left_count': 15,
    'right_count': 14,
    'total_weight': 1500.0,
    'avg_weight': 75.0
}
```

### Результат сравнения:
```python
ComparisonResult(
    auto_act={...},
    manual_act={...},
    match_type='exact',  # 'exact', 'close', 'mismatch', 'missing'
    time_diff_minutes=2.5,
    count_diff_percent=3.2,
    weight_diff_percent=1.5
)
```

---

## 📈 Метрики точности

| Метрика | Описание | Формула |
|---------|----------|---------|
| **Recall** | Полнота (сколько ручных актов найдено) | TP / (TP + FN) |
| **Precision** | Точность (сколько авто актов верны) | TP / (TP + FP) |
| **F1-Score** | Гармоническое среднее | 2 * (P * R) / (P + R) |
| **MAE** | Средняя абсолютная ошибка | Σ\|auto - manual\| / n |
| **MAPE** | Средняя процентная ошибка | MAE в процентах |
| **Correlation** | Корреляция | 1 - (MAE / 100) |

---

## 🎨 Цветовое кодирование в отчете

| Цвет | Значение | Условие |
|------|----------|---------|
| 🟢 Зеленый | Совпадение | Разница ≤ 10% |
| 🟡 Желтый | Небольшое расхождение | 10% < Разница ≤ 20% |
| 🔴 Красный | Большое расхождение | Разница > 20% |
| ⚪ Серый | Пропущенные | Нет пары |

---

## 🔧 Зависимости

```bash
pip install openpyxl pandas
```

**Опционально:**
- `openpyxl` - для работы с Excel (рекомендуется)
- `pandas` - альтернатива для парсинга

---

## ✅ Выполненные требования

### Задача 9: ExcelAnalyzer
- ✅ 9.1 Парсинг Excel файла
- ✅ 9.2 Анализ схемы шаблона

### Задача 10: ExcelExporter
- ✅ 10.1 Группировка актов по дате
- ✅ 10.2 Запись в Excel с форматированием

### Задача 11: ExcelComparator
- ✅ 11.1 Сопоставление по времени
- ✅ 11.2 Сравнение показателей
- ✅ 11.3 Вычисление метрик
- ✅ 11.4 Генерация отчета

---

## 🎯 Следующие шаги

### Задача 12: Тестовый режим консольного приложения
- [ ] 12.1 Парсинг параметров `--mode test`
- [ ] 12.2 Автоматическая сверка

**Пример команды:**
```bash
python console_app.py --mode test \
    --video test_video.mp4 \
    --excel-reference manual_records.xlsx \
    --output test_results/
```

---

**Создано:** 2025-10-18  
**Статус:** ✅ Задачи 9-11 выполнены  
**Следующее:** Задача 12 (тестовый режим)
