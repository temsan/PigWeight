# Задания для параллельной работы

## 🎯 ВКЛАДКА 1: Тестирование обработки видео

### Задача: Запустить обработку тестового видео и проверить результаты

**Приоритет:** ВЫСОКИЙ  
**Время:** 10-15 минут

#### Шаги:

1. **Запустить консольное приложение:**
```bash
python console_app.py
```

2. **Выбрать тестовое видео:**
   - В меню выбрать `uploads/test_video.mp4`
   - Запустить обработку

3. **Проверить результаты:**
   - Дождаться завершения обработки
   - Проверить логи на наличие ошибок
   - Записать статистику:
     - Количество обработанных кадров
     - Количество обнаруженных объектов
     - Количество треков
     - FPS обработки

4. **Проверить данные в БД:**
```bash
python -c "from pig_tracking.database import DatabaseManager; db = DatabaseManager(); print(f'Акты: {db.get_stats()}')"
```

#### Ожидаемый результат:
- Видео обработано без ошибок
- В БД записаны данные о пересечениях линий
- Возможно созданы акты взвешивания (если объекты пересекли линии)

#### Файлы для проверки:
- `logs/console.log` - логи обработки
- База данных Supabase - таблицы `weighing_acts` и `crossings`

---

## 🎯 ВКЛАДКА 2: Запуск и тестирование API

### Задача: Запустить API сервер и протестировать эндпоинты

**Приоритет:** ВЫСОКИЙ  
**Время:** 10-15 минут

#### Шаги:

1. **Запустить API сервер:**
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

2. **Проверить доступность:**
   - Открыть в браузере: http://localhost:8080/docs
   - Проверить Swagger UI

3. **Протестировать эндпоинты:**

**Health check:**
```bash
curl http://localhost:8080/health
```

**Получить статистику:**
```bash
curl http://localhost:8080/api/statistics
```

**Получить акты взвешивания:**
```bash
curl http://localhost:8080/api/weighing-acts
```

**Получить пересечения линий:**
```bash
curl http://localhost:8080/api/line-crossings
```

4. **Создать тестовый скрипт:**

Создать файл `test_api_endpoints.py`:
```python
import requests
import json

BASE_URL = "http://localhost:8080"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health: {response.status_code} - {response.json()}")

def test_statistics():
    response = requests.get(f"{BASE_URL}/api/statistics")
    print(f"Statistics: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))

def test_weighing_acts():
    response = requests.get(f"{BASE_URL}/api/weighing-acts")
    print(f"Weighing Acts: {response.status_code}")
    if response.status_code == 200:
        acts = response.json()
        print(f"  Найдено актов: {len(acts)}")

def test_crossings():
    response = requests.get(f"{BASE_URL}/api/line-crossings?limit=10")
    print(f"Line Crossings: {response.status_code}")
    if response.status_code == 200:
        crossings = response.json()
        print(f"  Найдено пересечений: {len(crossings)}")

if __name__ == '__main__':
    print("Тестирование API эндпоинтов\n")
    test_health()
    test_statistics()
    test_weighing_acts()
    test_crossings()
```

Запустить:
```bash
python test_api_endpoints.py
```

#### Ожидаемый результат:
- API сервер запущен и доступен
- Все эндпоинты возвращают корректные ответы
- Swagger UI работает

---

## 🎯 ДОПОЛНИТЕЛЬНЫЕ ЗАДАЧИ (если есть время)

### Задача 3: Создать простой веб-интерфейс для мониторинга

**Файл:** `static/monitor.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>Pig Tracking Monitor</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        .stat { padding: 10px; margin: 10px; border: 1px solid #ccc; }
        .refresh { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Система отслеживания свиней - Мониторинг</h1>
    
    <button class="refresh" onclick="loadStats()">Обновить</button>
    
    <div id="stats"></div>
    <div id="acts"></div>
    <div id="crossings"></div>
    
    <script>
        async function loadStats() {
            const response = await fetch('/api/statistics');
            const data = await response.json();
            document.getElementById('stats').innerHTML = 
                '<h2>Статистика</h2><pre>' + JSON.stringify(data, null, 2) + '</pre>';
        }
        
        async function loadActs() {
            const response = await fetch('/api/weighing-acts');
            const data = await response.json();
            document.getElementById('acts').innerHTML = 
                '<h2>Акты взвешивания (' + data.length + ')</h2><pre>' + 
                JSON.stringify(data.slice(0, 5), null, 2) + '</pre>';
        }
        
        async function loadCrossings() {
            const response = await fetch('/api/line-crossings?limit=10');
            const data = await response.json();
            document.getElementById('crossings').innerHTML = 
                '<h2>Последние пересечения (' + data.length + ')</h2><pre>' + 
                JSON.stringify(data, null, 2) + '</pre>';
        }
        
        loadStats();
        loadActs();
        loadCrossings();
        
        // Автообновление каждые 5 секунд
        setInterval(() => {
            loadStats();
            loadActs();
            loadCrossings();
        }, 5000);
    </script>
</body>
</html>
```

Открыть: http://localhost:8080/monitor.html

---

### Задача 4: Обновить статус задач в спецификации

**Файл:** `.kiro/specs/pig-tracking-system/tasks.md`

Обновить статус задачи 8:
```markdown
## 8. Тестирование MVP с базой данных
**Статус:** ✅ done
**Приоритет:** Высокий

### Выполнено:
- [x] Система готова (9/9 проверок)
- [x] Тестовое видео создано
- [x] База данных подключена
- [x] JWT ключи исправлены
- [x] Обработка видео протестирована
- [x] API протестирован

### Результаты:
- Видео обработано: X кадров
- Обнаружено объектов: Y
- Записано в БД: Z актов, W пересечений
- API работает корректно
```

---

### Задача 5: Создать финальный отчет

**Файл:** `MVP_FINAL_REPORT.md`

Структура:
```markdown
# Финальный отчет MVP - Система отслеживания свиней

## Резюме
- Статус: ✅ Готово / ⚠️ Требует доработки
- Дата: 2025-10-18

## Тестирование обработки видео
- Видео: test_video.mp4
- Результаты: ...

## Тестирование API
- Эндпоинты: ...
- Производительность: ...

## База данных
- Актов: N
- Пересечений: M

## Проблемы и решения
1. JWT ключи - ✅ исправлено
2. ...

## Следующие шаги
1. Тестирование на реальном видео
2. Оптимизация производительности
3. ...
```

---

## 📊 Чек-лист выполнения

### Вкладка 1:
- [ ] Запущена обработка видео
- [ ] Видео обработано без ошибок
- [ ] Проверены логи
- [ ] Проверена БД
- [ ] Записана статистика

### Вкладка 2:
- [ ] Запущен API сервер
- [ ] Проверен Swagger UI
- [ ] Протестированы все эндпоинты
- [ ] Создан тестовый скрипт
- [ ] Все тесты прошли

### Дополнительно:
- [ ] Создан веб-интерфейс мониторинга
- [ ] Обновлен статус задач
- [ ] Создан финальный отчет

---

## 🚀 Быстрый старт

**Вкладка 1:**
```bash
python console_app.py
# Выбрать uploads/test_video.mp4
```

**Вкладка 2:**
```bash
python -m uvicorn api.app:app --port 8080 --reload
# Открыть http://localhost:8080/docs
```

---

## 📞 Координация

После выполнения задач:
1. Сообщить о результатах
2. Поделиться статистикой
3. Обсудить найденные проблемы
4. Спланировать следующие шаги

---

**Создано:** 2025-10-18 17:15  
**Статус:** Готово к выполнению
