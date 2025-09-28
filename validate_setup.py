#!/usr/bin/env python3
"""Быстрая проверка базовой конфигурации PigWeight."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List

REQUIRED_FILES = [
    'requirements.txt',
    'main.py',
    'core/processor.py',
    'api/app.py',
    'config.env.example',
]


def check_files() -> List[str]:
    """Возвращает список отсутствующих обязательных файлов."""
    missing = [name for name in REQUIRED_FILES if not Path(name).exists()]
    return missing


def check_env_file() -> List[str]:
    """Проверяет наличие ключевых параметров в .env или .env.example."""
    env_path = Path('.env') if Path('.env').exists() else Path('.env.example')
    if not env_path.exists():
        return ['⚠️ Файл .env или .env.example не найден']

    content = env_path.read_text(encoding='utf-8')
    notes: List[str] = []
    required_keys = ['MODEL_PATH', 'DEVICE', 'IMG_SIZE', 'HOST', 'PORT']
    for key in required_keys:
        if f'{key}=' not in content:
            notes.append(f'⚠️ В конфигурации отсутствует ключ {key}')

    if not notes:
        notes.append('✅ Конфигурационный файл содержит базовые параметры')
    else:
        notes.insert(0, f'⚠️ Проверен файл {env_path.name}, найдены пропуски:')
    return notes


def check_python() -> List[str]:
    """Оценивает версию Python и наличие ключевых модулей."""
    messages: List[str] = []
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    if sys.version_info >= (3, 10):
        messages.append(f'✅ Python {version} подходит (требуется 3.10 или новее)')
    else:
        messages.append(f'⚠️ Требуется Python 3.10+, обнаружена версия {version}')

    for module in ['fastapi', 'uvicorn', 'torch']:
        try:
            importlib.import_module(module)
            messages.append(f'✅ Модуль {module} установлен')
        except ImportError:
            messages.append(f'⚠️ Модуль {module} не найден')
    return messages


def main() -> int:
    """Точка входа скрипта."""
    print('🔍 Проверка окружения PigWeight')
    print('=' * 48)

    print('\n📦 Обязательные файлы:')
    missing = check_files()
    if not missing:
        print('   ✅ Все обязательные файлы найдены')
    else:
        for name in missing:
            print(f'   ⚠️ Отсутствует {name}')

    print('\n🛠️ Конфигурация (.env):')
    for note in check_env_file():
        print(f'   {note}')

    print('\n🐍 Среда Python:')
    for note in check_python():
        print(f'   {note}')

    print('\nГотово. Устраните предупреждения перед запуском `python main.py`.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
