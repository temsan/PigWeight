#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для очистки директории uploads, где хранятся файлы с оригинальными именами для кеширования.
"""

import os
import shutil
from pathlib import Path
import logging
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('clean_uploads')


def clean_uploads(uploads_dir='uploads', keep_dir=True):
    """
    Очистка директории uploads.
    
    Args:
        uploads_dir (str): Путь к директории uploads
        keep_dir (bool): Сохранить директорию после очистки
    """
    uploads_path = Path(uploads_dir)
    
    if not uploads_path.exists():
        logger.info(f"Директория {uploads_dir} не существует. Создаем...")
        uploads_path.mkdir(parents=True, exist_ok=True)
        return
    
    if not uploads_path.is_dir():
        logger.error(f"{uploads_dir} не является директорией!")
        return
    
    try:
        if keep_dir:
            # Удаляем все файлы в директории
            file_count = 0
            for item in uploads_path.iterdir():
                if item.is_file():
                    item.unlink()
                    file_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    file_count += 1
            logger.info(f"Удалено {file_count} файлов/директорий из {uploads_dir}")
        else:
            # Удаляем директорию полностью и создаем заново
            shutil.rmtree(uploads_path)
            uploads_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Директория {uploads_dir} полностью очищена и создана заново")
    except Exception as e:
        logger.error(f"Ошибка при очистке директории {uploads_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Очистка директории uploads')
    parser.add_argument('--dir', default='uploads', help='Путь к директории uploads')
    parser.add_argument('--remove-dir', action='store_true', help='Удалить и пересоздать директорию')
    
    args = parser.parse_args()
    
    logger.info(f"Начинаем очистку директории {args.dir}")
    clean_uploads(args.dir, not args.remove_dir)
    logger.info("Очистка завершена")


if __name__ == '__main__':
    main()