# Scripts - Скрипты проекта

## 📁 Структура

### `/setup` - Настройка и проверка
- `check_system.py` - проверка готовности системы (9 проверок)
- `create_test_video.py` - создание тестового видео
- `generate_jwt_keys.py` - генерация JWT ключей для Supabase
- `fix_supabase_connection.py` - диагностика и исправление подключения к БД

### `/tests` - Тестирование
- `test_mvp.py` - полное тестирование MVP
- `test_api_endpoints.py` - тестирование API эндпоинтов
- `test_database.py` - тестирование базы данных
- `test_integration.py` - интеграционные тесты
- `test_inference.py` - тестирование инференса модели
- `test_video_processing.py` - тестирование обработки видео
- `validate_setup.py` - валидация настройки окружения

### `/utils` - Утилиты
- `cleanup_venv.bat` - очистка виртуального окружения (Windows)
- `start_server_venv.bat` - запуск сервера в venv (Windows)
- `start_server.bat` - запуск сервера (Windows)

---

## 🔧 Основные скрипты в корне `/scripts`

### Анализ и производительность:
- `analyze_measurements.py` - анализ измерений
- `analyze_pipeline.py` - анализ pipeline
- `deep_perf_analysis.py` - глубокий анализ производительности
- `perf_test.py` - тест производительности
- `performance_validation.py` - валидация производительности
- `load_testing.py` - нагрузочное тестирование
- `stream_performance_test.py` - тест производительности стрима

### CUDA и GPU:
- `check_cuda.py` - проверка CUDA
- `debug_cuda.py` - отладка CUDA
- `gpu_memory_check.py` - проверка памяти GPU

### Модели:
- `convert_model_to_onnx.py` - конвертация модели в ONNX
- `convert_to_onnx.py` - конвертация в ONNX
- `train_pig_yolo11_seg.py` - обучение модели YOLO11
- `train_pig_yolo11_seg_colab.py` - обучение в Google Colab
- `finetune_pig_yolo11_seg_colab.py` - дообучение в Colab

### Тестирование стримов:
- `test_video_stream.py` - тест видео стрима
- `test_mjpeg_stream.py` - тест MJPEG стрима
- `test_webrtc.py` - тест WebRTC
- `test_webrtc_debug.py` - отладка WebRTC
- `webrtc_streamer.py` - WebRTC стример

### Утилиты:
- `clean_uploads.py` / `.bat` - очистка загрузок
- `cleanup_old_records.py` - очистка старых записей
- `cleanup_venv.py` - очистка виртуального окружения
- `quick_status_check.py` - быстрая проверка статуса
- `run_all_tests.py` - запуск всех тестов
- `test_imports.py` - тест импортов
- `test_local_video.py` - тест локального видео
- `patch_codex.py` - патч для codex

---

## 🚀 Быстрые команды

### Проверка системы:
```bash
python scripts/setup/check_system.py
```

### Создание тестового видео:
```bash
python scripts/setup/create_test_video.py
```

### Тестирование MVP:
```bash
python scripts/tests/test_mvp.py
```

### Тестирование API:
```bash
python scripts/tests/test_api_endpoints.py
```

### Проверка CUDA:
```bash
python scripts/check_cuda.py
```

### Анализ производительности:
```bash
python scripts/deep_perf_analysis.py
```

---

## 📝 Примечания

- Скрипты в `/setup` используются для первоначальной настройки
- Скрипты в `/tests` используются для тестирования
- Скрипты в `/utils` - вспомогательные утилиты
- Остальные скрипты в корне `/scripts` - специализированные инструменты

---

**Обновлено:** 2025-10-18
