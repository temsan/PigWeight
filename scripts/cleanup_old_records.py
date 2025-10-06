#!/usr/bin/env python3
"""
Скрипт для очистки старых записей в папке records/
Удаляет файлы старше указанного количества дней
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# Конфигурация
RECORDS_DIR = Path("records")
DAYS_TO_KEEP = 30  # Количество дней для хранения записей
DRY_RUN = False  # Если True, только показывает что будет удалено

def get_file_age_days(file_path):
    """Получить возраст файла в днях"""
    file_time = os.path.getmtime(file_path)
    current_time = time.time()
    age_seconds = current_time - file_time
    return age_seconds / (24 * 3600)  # Конвертируем в дни

def cleanup_old_records():
    """Очистка старых записей"""
    if not RECORDS_DIR.exists():
        print(f"Папка {RECORDS_DIR} не найдена")
        return
    
    print(f"Очистка записей старше {DAYS_TO_KEEP} дней в {RECORDS_DIR}")
    print(f"Текущая дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_size_before = 0
    total_size_after = 0
    files_to_delete = []
    
    # Собираем все файлы для анализа
    for file_path in RECORDS_DIR.rglob("*"):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            total_size_before += file_size
            
            age_days = get_file_age_days(file_path)
            
            if age_days > DAYS_TO_KEEP:
                files_to_delete.append((file_path, file_size, age_days))
            else:
                total_size_after += file_size
    
    print(f"\nСтатистика:")
    print(f"   Всего файлов: {len(list(RECORDS_DIR.rglob('*')))}")
    print(f"   Файлов к удалению: {len(files_to_delete)}")
    print(f"   Размер до очистки: {total_size_before / (1024*1024):.1f} MB")
    print(f"   Размер после очистки: {total_size_after / (1024*1024):.1f} MB")
    print(f"   Экономия места: {(total_size_before - total_size_after) / (1024*1024):.1f} MB")
    
    if not files_to_delete:
        print("Нет файлов для удаления")
        return
    
    print(f"\nФайлы для удаления (старше {DAYS_TO_KEEP} дней):")
    
    # Группируем по типам файлов
    by_type = {}
    for file_path, size, age in files_to_delete:
        ext = file_path.suffix
        if ext not in by_type:
            by_type[ext] = []
        by_type[ext].append((file_path, size, age))
    
    for ext, files in by_type.items():
        print(f"\n   {ext or 'без расширения'}: {len(files)} файлов")
        total_size = sum(size for _, size, _ in files)
        print(f"   Размер: {total_size / (1024*1024):.1f} MB")
        
        # Показываем несколько примеров
        for i, (file_path, size, age) in enumerate(files[:5]):
            print(f"     - {file_path.name} ({age:.1f} дней, {size/1024:.1f} KB)")
        
        if len(files) > 5:
            print(f"     ... и еще {len(files) - 5} файлов")
    
    if DRY_RUN:
        print(f"\nРЕЖИМ ПРОСМОТРА (DRY_RUN=True)")
        print("   Для реального удаления установите DRY_RUN=False")
    else:
        print(f"\nУДАЛЕНИЕ ФАЙЛОВ...")
        deleted_count = 0
        deleted_size = 0
        
        for file_path, size, age in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
                deleted_size += size
                print(f"   Удален: {file_path.name}")
            except Exception as e:
                print(f"   Ошибка удаления {file_path.name}: {e}")
        
        print(f"\nУдалено {deleted_count} файлов, освобождено {deleted_size / (1024*1024):.1f} MB")

def main():
    """Главная функция"""
    print("Скрипт очистки старых записей PigWeight")
    print("=" * 50)
    
    cleanup_old_records()
    
    print("\n" + "=" * 50)
    print("Рекомендации:")
    print("   - Запускайте этот скрипт еженедельно")
    print("   - Настройте DAYS_TO_KEEP под ваши потребности")
    print("   - Сначала запустите с DRY_RUN=True для проверки")

if __name__ == "__main__":
    main()
