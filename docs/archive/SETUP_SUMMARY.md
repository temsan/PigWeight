# PigWeight Setup Summary

## Ключевые результаты
- Активирована базовая конфигурация из `.env` / `.env.example`.
- FastAPI-приложение и обработчик потоков запускаются через `python main.py`.
- Проверка среды автоматизирована скриптом `validate_setup.py`.

## Основные команды
```bash
# Первичная установка зависимостей
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Быстрая проверка конфигурации
python validate_setup.py

# Развёртывание API и стриминга
python main.py
```

## Что проверяет `validate_setup.py`
- Наличие ключевых файлов (`requirements.txt`, `main.py`, `api/app.py`, `core/processor.py`).
- Заполненность базовых параметров в `.env` или `.env.example`.
- Версию Python (3.10+) и доступность модулей `fastapi`, `uvicorn`, `torch`.

## Минимальные требования
- Python 3.10 или новее.
- CUDA / GPU по желанию (настройки через `DEVICE` и `USE_HALF`).
- Модели распознавания расположены в `models/`.

## Следующие шаги
1. Отредактировать `.env` под своё оборудование (URL камер, параметры очереди).
2. Запустить `pytest` и по необходимости `python scripts/run_all_tests.py`.
3. Настроить систему логирования (`logs/`) и мониторинг, если требуется.

🎯 После этих шагов система готова к прогону потоков и последующей автоматизации.
