"""
Пример использования IntegratedVideoProcessor для обработки видео.
"""

import asyncio
import logging
from pathlib import Path

from pig_tracking import IntegratedVideoProcessor
from core.config import setup_logging, CONFIG

# Настройка логирования
logger = setup_logging(debug=True)


async def process_video_example(video_path: str):
    """Пример обработки видеофайла"""
    
    # Создаем процессор с параметрами из конфига
    processor = IntegratedVideoProcessor(
        stream_id="example_video",
        conf_threshold=CONFIG.CONF_THRESHOLD,
        img_size=CONFIG.IMG_SIZE,
        line_left_x=CONFIG.LINE_LEFT_X,
        line_right_x=CONFIG.LINE_RIGHT_X,
        min_pigs_for_act=3,
        max_interval_sec=30.0
    )
    
    # Инициализируем процессор
    await processor.initialize()
    
    # Callback для отображения прогресса
    def progress_callback(current, total):
        progress = (current / total) * 100
        print(f"Прогресс: {current}/{total} ({progress:.1f}%)")
    
    # Обрабатываем видео
    summary = await processor.process_video_file(
        video_path,
        progress_callback=progress_callback
    )
    
    # Выводим итоговую статистику
    print("\n" + "="*60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*60)
    print(f"Видео: {summary['video_path']}")
    print(f"Обработано кадров: {summary['frames_processed']}/{summary['total_frames']}")
    print(f"Время обработки: {summary['processing_time']:.1f}s")
    print(f"Средний FPS: {summary['avg_fps']:.1f}")
    print(f"Среднее время на кадр: {summary['avg_frame_time']*1000:.1f}ms")
    print()
    
    # Статистика пересечений
    crossing_stats = summary['crossing_stats']
    print("ПЕРЕСЕЧЕНИЯ ЛИНИЙ:")
    print(f"  Вход слева: {crossing_stats['left_in']}")
    print(f"  Вход справа: {crossing_stats['right_in']}")
    print(f"  Всего пересечений: {crossing_stats['total_crossings']}")
    print()
    
    # Статистика актов
    act_stats = summary['act_stats']
    print("АКТЫ ВЗВЕШИВАНИЯ:")
    print(f"  Завершенных актов: {act_stats['completed_acts_count']}")
    
    if 'completed_acts' in act_stats:
        print("\nДетали актов:")
        for act in act_stats['completed_acts']:
            print(f"  Акт #{act['act_id']}:")
            print(f"    Начало: {act['started_at_iso']}")
            print(f"    Окончание: {act['ended_at_iso']}")
            print(f"    Длительность: {act['duration']:.1f}s")
            print(f"    Вход слева: {act['left_count']}")
            print(f"    Вход справа: {act['right_count']}")
            print(f"    Пиковое количество: {act['peak_count']}")
            print(f"    Всего уникальных: {act['seen_total']}")
            print()
    
    print("="*60)


async def main():
    """Главная функция"""
    # Путь к видеофайлу (можно передать как аргумент командной строки)
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # По умолчанию ищем видео в папке uploads
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            videos = list(uploads_dir.glob("*.mp4"))
            if videos:
                video_path = str(videos[0])
                print(f"Используем видео: {video_path}")
            else:
                print("Видеофайлы не найдены в папке uploads/")
                print("Использование: python example_usage.py <путь_к_видео>")
                return
        else:
            print("Папка uploads/ не найдена")
            print("Использование: python example_usage.py <путь_к_видео>")
            return
    
    # Проверяем существование файла
    if not Path(video_path).exists():
        print(f"Ошибка: файл не найден: {video_path}")
        return
    
    # Обрабатываем видео
    await process_video_example(video_path)


if __name__ == "__main__":
    asyncio.run(main())
