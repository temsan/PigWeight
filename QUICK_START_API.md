# 🚀 Быстрый старт API - ВКЛАДКА 2

## Запуск за 3 шага

### 1️⃣ Запустить API сервер

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

**Ожидаемый вывод:**
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 2️⃣ Проверить работу

Открыть в браузере:
- **Swagger UI:** http://localhost:8080/docs
- **Monitor:** http://localhost:8080/monitor.html
- **Dashboard:** http://localhost:8080/dashboard

### 3️⃣ Запустить тесты

```bash
python test_api_full.py
```

---

## 📊 Что проверяется

✅ Health Check  
✅ Swagger UI  
✅ API камер  
✅ Журнал актов  
✅ Статистика  
✅ Записи  
✅ Dashboard  
✅ Monitoring  

---

## 🎯 Результат

После запуска тестов вы увидите:

```
🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!

📊 Следующие шаги:
  1. Откройте Swagger UI: http://localhost:8080/docs
  2. Откройте Dashboard: http://localhost:8080/dashboard
  3. Откройте Monitoring: http://localhost:8080/monitoring
```

---

## 🔗 Координация с ВКЛАДКОЙ 1

**Сейчас:** API работает, но данных нет  
**После ВКЛАДКИ 1:** API вернет реальные данные из БД

**Порядок действий:**
1. ✅ ВКЛАДКА 2: Запустить API (вы здесь)
2. ⏳ ВКЛАДКА 1: Обработать видео
3. 🎉 Проверить данные в API

---

## 📱 Веб-интерфейс

Новый интерфейс мониторинга:
```
http://localhost:8080/monitor.html
```

**Функции:**
- 🟢 Статус сервера
- 📊 Статистика в реальном времени
- 📝 Список актов
- ↔️ Пересечения линий
- 🔄 Автообновление каждые 5 сек

---

## ⚡ Быстрые команды

```bash
# Запуск API
python -m uvicorn api.app:app --port 8080 --reload

# Тесты
python test_api_full.py

# Проверка health
curl http://localhost:8080/health

# Статистика
curl http://localhost:8080/api/weighing/stats
```

---

**Статус:** ✅ Готово к работе  
**Время:** ~2 минуты на запуск
