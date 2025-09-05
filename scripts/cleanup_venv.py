#!/usr/bin/env python3
"""
Скрипт для очистки автоматически создаваемых виртуальных окружений
"""

import os
import shutil
from pathlib import Path

def cleanup_venv_dirs():
    """Удаление автоматически создаваемых папок виртуальных окружений"""
    
    base_dir = Path(__file__).parent.parent
    venv_patterns = ['.venv_temp', '.venv_auto', 'venv_temp', '.env_temp']
    
    removed_count = 0
    
    print("🧹 Очистка автоматически созданных папок виртуальных окружений...")
    
    for pattern in venv_patterns:
        venv_path = base_dir / pattern
        if venv_path.exists():
            try:
                if venv_path.is_dir():
                    shutil.rmtree(venv_path)
                    print(f"✅ Удалена папка: {pattern}")
                    removed_count += 1
                else:
                    venv_path.unlink()
                    print(f"✅ Удален файл: {pattern}")
                    removed_count += 1
            except Exception as e:
                print(f"❌ Не удалось удалить {pattern}: {e}")
        else:
            print(f"✅ {pattern} не найдена")
    
    if removed_count == 0:
        print("🎯 Нет папок для удаления")
    else:
        print(f"🎯 Удалено папок: {removed_count}")
    
    print()
    print("💡 Рекомендации:")
    print("   - Используйте основное окружение: .venv")
    print("   - Отключите автосоздание в настройках IDE")
    print("   - Проверьте, что .gitignore содержит /.venv*")

if __name__ == "__main__":
    cleanup_venv_dirs()