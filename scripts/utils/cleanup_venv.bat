@echo off
REM Скрипт для удаления автоматически создаваемых виртуальных окружений

echo 🧹 Очистка автоматически созданных папок виртуальных окружений...

if exist .venv_temp (
    echo Удаляем .venv_temp...
    rmdir /s /q .venv_temp
    echo ✅ .venv_temp удалена
) else (
    echo ✅ .venv_temp не найдена
)

if exist .venv_auto (
    echo Удаляем .venv_auto...
    rmdir /s /q .venv_auto
    echo ✅ .venv_auto удалена
) else (
    echo ✅ .venv_auto не найдена
)

if exist venv_temp (
    echo Удаляем venv_temp...
    rmdir /s /q venv_temp
    echo ✅ venv_temp удалена
) else (
    echo ✅ venv_temp не найдена
)

echo 🎯 Очистка завершена
echo.
echo 💡 Для предотвращения создания используйте:
echo    - Основное окружение: .venv
echo    - Деактивация автоматических окружений в IDE
pause