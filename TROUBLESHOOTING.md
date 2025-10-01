# 🔧 Устранение неполадок PigWeight

## Проблемы и решения после обновления (01.10.2025)

### ✅ Что исправлено

1. **Всплывающие счетчики пересечения линий** 
   - Восстановлен обработчик `window.__popups` в `static/index.html`
   - События `crossings` теперь корректно обрабатываются из WebSocket

2. **Пастельные маски**
   - Код отрисовки работает (проверен в `index.html`, строки 1427-1488)
   - Использует `globalAlpha: 0.60` для полупрозрачности

3. **API журнала**
   - Исправлен путь: `/api/journal/records` → `/api/records`
   - Добавлен расчет статистики на клиенте
   - Сверка данных использует `/api/verification/compare`

4. **Очищены дубликаты**
   - Удалены: `.env.optimized`, `main_optimized.py`, `api/simple_endpoints.py`
   - Очищен кеш Python (`__pycache__`)
   - Убраны ссылки на несуществующие файлы

### ❌ Почему маски и линии не видны

**Основная причина:** Видеопоток не активен или av_worker не может открыть источник.

#### Проверка состояния системы

```bash
# 1. Проверить статус сервера
curl http://localhost:8000/api/system/status

# Ответ должен содержать:
# - "status": "online"
# - "available_videos": [список файлов]
# - "active_streams": [активные потоки]
```

#### Если маски не появляются:

1. **Проверьте, есть ли активный поток**
   ```javascript
   // В консоли браузера:
   fetch('/api/system/status').then(r => r.json()).then(console.log)
   ```
   
   Если `active_streams: []` — поток не запущен.

2. **Запустите тестовый файл**
   ```bash
   python scripts/test_local_video.py
   ```
   
   Это покажет доступные файлы и статус системы.

3. **Проверьте логи**
   ```powershell
   Get-Content logs/app.log -Tail 100 | Select-String -Pattern "masks|LEFT|RIGHT|ENTER"
   ```
   
   Ищите строки:
   - `✅ Получено масок: N`
   - `LEFT ENTER: track X, left_in=Y`
   - `RIGHT ENTER: track X, right_in=Y`

### 🔍 Диагностика av_worker timeout

Если в логе ошибки типа `av_worker timeout on open_file/open_rtsp`:

1. **Проверьте доступность видео**
   ```python
   import cv2
   cap = cv2.VideoCapture('путь/к/файлу.mp4')
   print(f"Открыт: {cap.isOpened()}")
   print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
   ```

2. **Перезапустите av_worker**
   - Остановите сервер (Ctrl+C)
   - Очистите кеш: `Remove-Item -Recurse -Force __pycache__`
   - Запустите снова: `python main.py`

3. **Проверьте кодеки**
   ```bash
   python -c "import av; print(av.codecs_available)"
   ```

### 📊 Проверка журналов и актов

Журналы теперь доступны через:
- **Dashboard**: http://localhost:8000/dashboard
- **API**: http://localhost:8000/api/records

#### Если журнал пустой:

1. **Запустите поток и дождитесь его завершения**
   - Акты создаются только после остановки потока
   - Проверьте папку `records/` - там должны быть файлы

2. **Проверьте формат файлов**
   ```bash
   ls records/act_*.json | head -5
   ```

3. **Очистка старых актов** (если хаос в records/)
   ```powershell
   # Переместить старые акты в архив
   New-Item -ItemType Directory -Force -Path records/archive
   Move-Item records/act_*_202509*.* records/archive/
   ```

### 🚀 Быстрый старт для проверки

```bash
# 1. Очистить кеш
Remove-Item -Recurse -Force api/__pycache__, api/endpoints/__pycache__

# 2. Запустить сервер
python main.py

# 3. Открыть в браузере
# http://localhost:8000

# 4. Загрузить тестовый файл через UI
# Кнопка "Открыть видеофайл"

# 5. Проверить консоль браузера (F12)
# Должны появиться сообщения о масках и пересечениях
```

### 📝 Что проверить в UI

- [ ] Видео загружается и воспроизводится
- [ ] Seekbar показывает длительность файла
- [ ] Перемотка работает (клик на seekbar)
- [ ] Пастельные маски появляются на свиньях
- [ ] Вертикальные линии можно двигать
- [ ] При пересечении линий появляются "+1/-1"
- [ ] Счетчики "Слева (вход)" и "Справа (вход)" обновляются
- [ ] Журнал загружается (вкладка "Журнал")
- [ ] Dashboard показывает акты (http://localhost:8000/dashboard)

### 🐛 Известные ограничения

1. **Ручное сохранение актов** - временно отключено
   - Используйте автоматическое сохранение при остановке потока
   
2. **Экспорт журнала** - перенаправляет на Dashboard
   - Dashboard имеет встроенную функцию экспорта

3. **WebSocket может отключаться**
   - Проверьте, что используете MJPEG транспорт
   - WebRTC еще в разработке

### 📞 Дополнительная помощь

Если проблема не решена:

1. Соберите диагностическую информацию:
   ```bash
   # Статус системы
   curl http://localhost:8000/api/system/status > system_status.json
   
   # Последние 200 строк лога
   Get-Content logs/app.log -Tail 200 > recent_logs.txt
   
   # Список актов
   ls records/act_*.json | Measure-Object | Select-Object Count
   ```

2. Проверьте версии зависимостей:
   ```bash
   python -c "import torch, ultralytics, av; print(f'Torch: {torch.__version__}, Ultralytics: {ultralytics.__version__}, PyAV: {av.__version__}')"
   ```

3. Откройте issue с этими данными в репозитории.

---

**Обновлено:** 01.10.2025  
**Ветка:** optimize  
**Коммиты:** 545432a, 350e9c4

