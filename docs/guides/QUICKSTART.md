# 🚀 Быстрый старт за 5 минут

## Шаг 1: Подготовка (1 мин)

```bash
# Скопировать настройки
cp .env.example .env

# Установить зависимости
pip install -r requirements-pig-tracking.txt
```

## Шаг 2: Запуск Supabase (2 мин)

```bash
# Запустить базу данных
docker-compose up -d

# Подождать 10 секунд пока запустится
```

## Шаг 3: Проверка системы (1 мин)

```bash
# Проверить что все готово
python check_system.py
```

Должно быть: `✅ 9/9 проверок пройдено`

## Шаг 4: Запуск (1 мин)

```bash
# Положить видео в папку uploads/
# Запустить приложение
python console_app.py
```

Выбрать видео из списка и дождаться обработки.

## ✅ Готово!

Результаты сохранены в базу данных Supabase.

Просмотр результатов:
- Через Supabase Studio: http://localhost:8000
- Через Python: `python test_database.py`

---

## 🐛 Если что-то не работает

```bash
# Перезапустить Supabase
docker-compose down -v
docker-compose up -d

# Проверить систему
python check_system.py
```

## 📚 Подробная документация

См. `README_PIG_TRACKING.md`