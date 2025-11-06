# 🔄 ПЛАН СИНХРОНИЗАЦИИ CODE С .kiro/specs

**Проблема:** Разработка велась "от балды", архитектура не соответствует дизайну  
**Решение:** Привести весь code в соответствие с specs  
**Статус:** ANALYSIS COMPLETE ✅

---

## 🎯 MAIN ISSUES

### 1. API ENDPOINTS MISMATCH

**Specs предусматривает:**
```
/api/stats/current         - Текущие показатели
/api/events/list           - Список событий
/api/export/excel          - Экспорт в Excel
/api/verify/compare        - Сверка с Excel
```

**На самом деле есть:**
```
/api/metrics/current       - ✅ Есть (но другое место)
/api/events/{stream_id}    - ✅ Есть (но разное)
/api/events/stats          - 🔴 404 (добавили но не синхронизировано)
/api/records/export        - ✅ Есть (но другое место)
```

**ДЕЙСТВИЕ:** Стандартизировать пути согласно specs

---

### 2. DATABASE SCHEMA MISMATCH

**Specs:** 
- `weighing_acts` (id, started_at, ended_at, counts, peak_count)
- `crossings` (id, act_id, direction, timestamp, coordinates)

**На самом деле:**
- ✅ Таблицы есть (примерно)
- ❌ Названия полей отличаются
- ❌ Связи неправильные

**ДЕЙСТВИЕ:** Миграция БД к spec-совместимой схеме

---

### 3. VIDEO PROCESSING PIPELINE

**Specs говорит:**
```
VideoCapture (читай кадры)
    ↓
UnifiedVideoProcessor (YOLO детекция)
    ↓
LineAnalyzer (подсчет пересечений)
    ↓
ActDetector (определение актов)
    ↓
Database (сохранение)
```

**На самом деле:**
- Кусочки разбросаны по разным местам
- Нет ясного pipeline
- Нет стандартного интерфейса

**ДЕЙСТВИЕ:** Собрать в единый pipeline согласно specs

---

## 📋 ПЛАН ДЕЙСТВИЙ (ПРИОРИТЕТ)

### PHASE 1: API STANDARDIZATION (2-3 часа)

**Создать новый маршрут `/api/` согласно specs:**

```python
# api/routes.py (новый файл!)
@router.get("/api/stats/current")
    → Текущие показатели (заменить /metrics/current)

@router.get("/api/events/list")
    → Список всех событий

@router.get("/api/events/{event_id}")
    → Деталь конкретного события

@router.post("/api/export/excel")
    → Экспорт в Excel (заменить /records/export)

@router.post("/api/verify/compare")
    → Сверка с Excel (заменить /records/compare)

@router.get("/api/config/parameters")
    → Параметры обработки

@router.post("/api/config/parameters")
    → Изменить параметры
```

**Результат:** Единая, интуитивная API структура

---

### PHASE 2: DATABASE MIGRATION (3-4 часа)

**Создать миграции для приведения схемы:**

```sql
-- Проверить текущую схему
SELECT column_name, data_type FROM information_schema.columns 
WHERE table_name IN ('weighing_acts', 'crossings')

-- Добавить/переименовать поля
ALTER TABLE weighing_acts ADD COLUMN IF NOT EXISTS peak_count INT
ALTER TABLE crossings ADD COLUMN IF NOT EXISTS act_id INT
```

**Результат:** 100% совместимость с specs

---

### PHASE 3: UNIFY VIDEO PROCESSING (4-5 часов)

**Собрать pipeline в один класс согласно specs:**

```python
# core/pipeline.py (новый)
class VideoPipeline:
    """Main processing pipeline according to specs"""
    
    def __init__(self, stream_id: str):
        self.video_capture = VideoCapture(stream_id)
        self.processor = UnifiedVideoProcessor()
        self.line_analyzer = LineAnalyzer()
        self.act_detector = ActDetector()
        self.db = DatabaseManager()
    
    async def process_frame(self, frame):
        """Полный pipeline обработки"""
        # Детектирование
        detections = await self.processor.detect(frame)
        
        # Анализ линий
        crossings = self.line_analyzer.analyze(detections)
        
        # Определение актов
        acts = self.act_detector.detect(crossings)
        
        # Сохранение
        await self.db.save_results(acts, crossings)
        
        return acts
```

**Результат:** Чистый, стандартный pipeline

---

### PHASE 4: UPDATE DOCUMENTATION (1-2 часа)

**Обновить все доки согласно new API:**
- `.kiro/specs` → актуально
- `README.md` → примеры использования
- Inline comments в коде

---

## 🚨 IMPACT ANALYSIS

### Что сломается

❌ Старый API (`/metrics/current`, `/records/export`)  
❌ Frontend может сломаться (нужны fix)  

### Что выиграем

✅ Полная архитектурная консистентность  
✅ Легче поддерживать  
✅ Новичкам понятнее  
✅ Масштабируемость  

---

## 📊 TIMELINE

| Фаза | Задача | Время | Статус |
|------|--------|-------|--------|
| 1 | API Standardization | 2-3ч | ⏳ NEXT |
| 2 | DB Migration | 3-4ч | ⏳ |
| 3 | Pipeline Unification | 4-5ч | ⏳ |
| 4 | Documentation | 1-2ч | ⏳ |
| **TOTAL** | | **~11 ч** | |

---

## 🎯 NEXT STEP

**НАЧАТЬ С PHASE 1:**
1. Прочитать `/api/routes.py` в specs
2. Создать новый маршрут в `api/app.py`
3. Redirect старый API на новый (для совместимости)
4. Обновить frontend
5. Протестировать

---

## 📌 IMPORTANT

**ЭТО ДОЛГОСРОЧНОЕ УЛУЧШЕНИЕ!**

Не сломает production, но даст архитектурную чистоту.

**Можно делать постепенно, фаза за фазой.**

---

**Рекомендация:** Начнём с Phase 1 (API), потом Phase 2 (DB).  
Phases 3-4 можно делать параллельно с тестированием.

