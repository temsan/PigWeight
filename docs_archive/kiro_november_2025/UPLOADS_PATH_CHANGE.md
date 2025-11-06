# Изменение пути экспорта на uploads/

**Дата:** 6 ноября 2025  
**Статус:** ✅ ВЫПОЛНЕНО

---

## 🎯 ИЗМЕНЕНИЕ

Изменен путь сохранения экспортированных файлов с `tempfile.gettempdir()` на `uploads/`

---

## 📝 ПРИЧИНА

- Папка `uploads/` уже используется в проекте
- Более логичная структура
- Легче найти экспортированные файлы
- Не зависит от системной временной папки

---

## 🔧 ЧТО ИЗМЕНЕНО

### Файл: `api/endpoints/export.py`

**До:**
```python
temp_dir = Path(tempfile.gettempdir())
output_path = temp_dir / f"export_{timestamp}.xlsx"
```

**После:**
```python
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
output_path = uploads_dir / f"export_{timestamp}.xlsx"
```

---

## 📂 СТРУКТУРА ПАПКИ uploads/

```
uploads/
├── export_20251106_171500.xlsx          # Экспорт актов
├── upload_20251106_171530_manual.xlsx   # Загруженный файл для сверки
├── comparison_20251106_171545.xlsx      # Отчет о сверке
└── ...
```

---

## 🚀 ENDPOINTS

### POST /api/export/excel
- Сохраняет файл в `uploads/export_{timestamp}.xlsx`
- Возвращает файл для скачивания

### POST /api/export/compare
- Сохраняет загруженный файл в `uploads/upload_{timestamp}_{filename}`
- Создает отчет в `uploads/comparison_{timestamp}.xlsx`
- Возвращает метрики и ссылку на скачивание

### GET /api/export/download/{filename}
- Скачивает файл из `uploads/{filename}`

---

## ✅ ПРЕИМУЩЕСТВА

1. **Централизованное хранение** - все файлы в одном месте
2. **Легкий доступ** - можно просмотреть через файловый менеджер
3. **Не зависит от системы** - не используется временная папка ОС
4. **Согласованность** - используется та же папка, что и для других загрузок

---

## 📝 ПРИМЕЧАНИЯ

- Папка `uploads/` создается автоматически при первом экспорте
- Старые файлы не удаляются автоматически (можно добавить cleanup)
- Рекомендуется добавить `uploads/*.xlsx` в `.gitignore`

---

**Статус:** ✅ Изменение применено  
**Файлы:** api/endpoints/export.py  
**Тестирование:** Требуется
