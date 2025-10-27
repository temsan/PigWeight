# 🧪 Руководство по тестированию системы распознавания

**Дата:** 27 октября 2025  
**Статус:** Готово к тестированию ✅

---

## 📋 Чек-лист перед тестированием

- [x] Интеграция с БД исправлена (27.10.2025)
- [x] Конвертация данных настроена
- [x] Обработка ошибок добавлена
- [ ] Supabase запущен
- [ ] Видео файл подготовлен
- [ ] База данных протестирована

---

## 🚀 Шаг 1: Настройка окружения

### 1.1. Проверьте наличие файла `.env`

Если файл `.env` отсутствует, создайте его на основе `config.env.example`:

```bash
cp config.env.example .env
```

### 1.2. Добавьте настройки Supabase в `.env`

Добавьте следующие строки в конец файла `.env`:

```env
# Supabase Configuration
SUPABASE_URL=http://localhost:54321
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJzZXJ2aWNlX3JvbGUiLAogICAgImlzcyI6ICJzdXBhYmFzZS1kZW1vIiwKICAgICJpYXQiOiAxNjQxNzY5MjAwLAogICAgImV4cCI6IDE3OTk1MzU2MDAKfQ.DaYlNEoUrrEn2Ig7tqibS-PHK5vgusbcbo7X36XVt4Q
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJzZXJ2aWNlX3JvbGUiLAogICAgImlzcyI6ICJzdXBhYmFzZS1kZW1vIiwKICAgICJpYXQiOiAxNjQxNzY5MjAwLAogICAgImV4cCI6IDE3OTk1MzU2MDAKfQ.DaYlNEoUrrEn2Ig7tqibS-PHK5vgusbcbo7X36XVt4Q
```

**Примечание:** Это демо-ключи для локального Supabase. В продакшене используйте настоящие ключи!

---

## 🐳 Шаг 2: Запуск Supabase

### 2.1. Запустите Docker Compose

```bash
docker-compose up -d
```

### 2.2. Проверьте статус контейнеров

```bash
docker-compose ps
```

Вы должны увидеть контейнеры:
- `db` (PostgreSQL)
- `kong` (API Gateway)
- `auth` (GoTrue)
- `rest` (PostgREST)

### 2.3. Проверьте доступность API

```bash
curl http://localhost:54321/rest/v1/
```

Должен вернуться JSON с информацией о API.

### 2.4. Проверьте миграции

Миграции применяются автоматически при первом запуске БД из файла:
```
supabase/migrations/001_initial_schema.sql
```

Проверьте, что таблицы созданы:

```bash
# Подключитесь к БД
docker exec -it pigweight-db-1 psql -U postgres -d postgres

# Проверьте таблицы
\dt

# Выход
\q
```

Вы должны увидеть таблицы:
- `weighing_acts`
- `crossings`
- `excel_schemas`

---

## 🎬 Шаг 3: Тестирование обработки видео

### 3.1. Подготовьте тестовое видео

Убедитесь, что в папке `uploads/` есть хотя бы один видеофайл:

```bash
ls uploads/
```

Доступные видео:
- `0825.mp4`
- `2.mp4`
- `Preview+Archive.50 2025-08-02 08_58_32-2025-08-02 10_01_44_(1)_(1)_(1).mkv`

### 3.2. Запустите обработку видео

**Интерактивный режим:**
```bash
python console_app.py
```

Выберите видео из списка.

**Режим с указанием файла:**
```bash
python console_app.py --video uploads/0825.mp4
```

### 3.3. Проверьте вывод

Вы должны увидеть:

1. **Инициализация:**
   ```
   ✓ Процессор готов
   ⏳ Начинаем обработку кадров...
   ```

2. **Прогресс обработки:**
   ```
   [████████████████░░░░░░░░░░░░░░] 54.2% | 1234/2278 кадров | 18.3 FPS | ETA: 57s
   ```

3. **Результаты:**
   ```
   ✅ Обработка завершена!
   
   📊 Результаты:
      • Обработано кадров: 2278
      • Обнаружено актов взвешивания: 3
      • Общее количество проходов: 47
      • Проходы слева: 24
      • Проходы справа: 23
      • Пиковое количество одновременно: 12
   ```

4. **Сохранение:**
   ```
   💾 Результаты сохранены в JSON: results/0825_20251027_163045_results.json
   
   💾 Сохранение результатов в базу данных...
   ✅ Сохранено 3 из 3 актов в базу данных
   ```

---

## 🔍 Шаг 4: Проверка данных в БД

### 4.1. Проверьте акты взвешивания

```sql
docker exec -it pigweight-db-1 psql -U postgres -d postgres -c "SELECT * FROM weighing_acts;"
```

Вы должны увидеть записи с полями:
- `id`
- `started_at`, `ended_at`
- `left_count`, `right_count`, `peak_count`
- `stream_id`, `video_file`

### 4.2. Проверьте пересечения

```sql
docker exec -it pigweight-db-1 psql -U postgres -d postgres -c "SELECT * FROM crossings LIMIT 10;"
```

Вы должны увидеть записи с полями:
- `id`, `act_id`
- `pig_id`
- `direction` ('left' или 'right')
- `crossed_at`
- `line_x`, `line_y`

### 4.3. Подсчитайте статистику

```sql
docker exec -it pigweight-db-1 psql -U postgres -d postgres -c "
SELECT 
    COUNT(*) as total_acts,
    SUM(left_count) as total_left,
    SUM(right_count) as total_right,
    MAX(peak_count) as max_peak
FROM weighing_acts;
"
```

---

## ✅ Чек-лист проверки

После выполнения всех шагов убедитесь:

- [ ] Supabase запущен и доступен
- [ ] База данных создана с таблицами
- [ ] Видео успешно обработано
- [ ] Акты взвешивания обнаружены
- [ ] Данные сохранены в JSON файл
- [ ] Данные сохранены в БД
- [ ] В логах нет ошибок
- [ ] Можно прочитать данные из БД через SQL

---

## 🐛 Troubleshooting

### Проблема: База данных недоступна

**Симптомы:**
```
⚠️ База данных недоступна, результаты сохранены только в JSON
```

**Решение:**
1. Проверьте, что Supabase запущен: `docker-compose ps`
2. Проверьте логи: `docker-compose logs db`
3. Проверьте настройки в `.env`
4. Перезапустите Supabase: `docker-compose restart`

### Проблема: Ошибка CHECK constraint

**Симптомы:**
```
ERROR: new row for relation "crossings" violates check constraint
```

**Решение:**
Убедитесь, что используется последняя версия `console_app.py` (27.10.2025), где `direction` сохраняется как 'left' или 'right'.

### Проблема: Ошибка конвертации timestamp

**Симптомы:**
```
TypeError: an integer is required (got type float)
```

**Решение:**
Убедитесь, что используется последняя версия `console_app.py` с функцией `datetime.fromtimestamp()`.

---

## 📊 Ожидаемые результаты

После успешного тестирования:

1. **JSON файл** (`results/*.json`) содержит:
   - Метаданные видео
   - Список актов с временными метками
   - Статистику пересечений
   - Детали каждого акта

2. **База данных** содержит:
   - Записи в таблице `weighing_acts`
   - Записи в таблице `crossings`
   - Связи между актами и пересечениями

3. **Логи** показывают:
   - Успешную инициализацию
   - Прогресс обработки
   - Успешное сохранение в БД

---

## 🎯 Следующие шаги

После успешного тестирования:

1. ✅ Отметить задачу тестирования как выполненную
2. 📝 Обновить документацию с результатами
3. 🚀 Подготовить систему к продакшену
4. 📈 Провести нагрузочное тестирование (опционально)
5. 🌐 Разработать веб-интерфейс (опционально)

---

**Последнее обновление:** 27.10.2025  
**Автор:** Kiro AI Assistant

