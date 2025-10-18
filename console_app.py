#!/usr/bin/env python3
"""
Консольное приложение для системы отслеживания свиней
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

# Импорты из существующей системы
from core.config import get_config
from pig_tracking.database import DatabaseManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoSelector:
    """Класс для выбора видеофайлов"""
    
    def __init__(self, uploads_dir: str = "uploads"):
        self.uploads_dir = Path(uploads_dir)
        if not self.uploads_dir.exists():
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    def get_video_files(self) -> List[Path]:
        """Получает список видеофайлов из папки uploads"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        video_files = []
        for file_path in self.uploads_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        return sorted(video_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def get_file_info(self, file_path: Path) -> dict:
        """Получает информацию о видеофайле"""
        try:
            import cv2
            
            # Размер файла
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024
            
            if size_gb >= 1:
                size_str = f"{size_gb:.1f} GB"
            else:
                size_str = f"{size_mb:.0f} MB"
            
            # Длительность видео
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                
                if fps > 0 and frame_count > 0:
                    duration_sec = frame_count / fps
                    hours = int(duration_sec // 3600)
                    minutes = int((duration_sec % 3600) // 60)
                    seconds = int(duration_sec % 60)
                    
                    if hours > 0:
                        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        duration_str = f"{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = "неизвестно"
                
                cap.release()
            else:
                duration_str = "ошибка чтения"
            
            return {
                'size': size_str,
                'duration': duration_str,
                'path': file_path
            }
            
        except Exception as e:
            logger.warning(f"Ошибка получения информации о файле {file_path}: {e}")
            return {
                'size': 'неизвестно',
                'duration': 'неизвестно',
                'path': file_path
            }
    
    def select_video_interactive(self) -> Optional[Path]:
        """Интерактивный выбор видеофайла"""
        video_files = self.get_video_files()
        
        if not video_files:
            print(f"❌ Видеофайлы не найдены в папке {self.uploads_dir}")
            print(f"   Поместите видеофайлы в папку {self.uploads_dir.absolute()}")
            return None
        
        print(f"\n📁 Доступные видео в папке {self.uploads_dir}:")
        print("=" * 80)
        
        for i, video_file in enumerate(video_files, 1):
            info = self.get_file_info(video_file)
            print(f"{i:2d}. {video_file.name}")
            print(f"    Размер: {info['size']}, Длительность: {info['duration']}")
        
        print("=" * 80)
        
        while True:
            try:
                choice = input(f"\nВыберите номер видео (1-{len(video_files)}) или 'q' для выхода: ").strip()
                
                if choice.lower() == 'q':
                    print("Выход...")
                    return None
                
                index = int(choice) - 1
                if 0 <= index < len(video_files):
                    selected_file = video_files[index]
                    print(f"✅ Выбран файл: {selected_file.name}")
                    return selected_file
                else:
                    print(f"❌ Неверный номер. Введите число от 1 до {len(video_files)}")
                    
            except ValueError:
                print("❌ Введите число или 'q' для выхода")
            except KeyboardInterrupt:
                print("\n\nВыход...")
                return None

class PigTrackingApp:
    """Основное приложение для отслеживания свиней"""
    
    def __init__(self):
        self.config = get_config()
        self.db = None
        self.video_selector = VideoSelector()
    
    def initialize_database(self):
        """Инициализация подключения к базе данных"""
        try:
            self.db = DatabaseManager()
            logger.info("✅ Подключение к базе данных успешно")
            
            # Показываем статистику
            stats = self.db.get_stats()
            logger.info(f"📊 В базе: {stats['total_acts']} актов, {stats['total_crossings']} проходов")
            
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к базе данных: {e}")
            logger.warning("   Результаты будут сохранены только в JSON файл")
            logger.warning("   Для сохранения в базу запустите: docker-compose up -d")
            self.db = None
    
    async def process_video(self, video_path: Path):
        """Обрабатывает видеофайл"""
        logger.info(f"🎬 Начинаем обработку видео: {video_path.name}")
        
        try:
            # Импортируем IntegratedVideoProcessor
            from pig_tracking.video_processor import IntegratedVideoProcessor
            
            print(f"\n🚀 Обработка видео: {video_path.name}")
            print("=" * 60)
            
            # Создаем процессор
            processor = IntegratedVideoProcessor(
                stream_id=video_path.stem,
                conf_threshold=self.config.CONF_THRESHOLD,
                img_size=self.config.IMG_SIZE
            )
            
            print("⏳ Инициализация процессора...")
            await processor.initialize()
            
            print("⏳ Начинаем обработку кадров...")
            
            # Обрабатываем видео
            summary = await processor.process_video_file(str(video_path))
            
            print("\n✅ Обработка завершена!")
            print("\n📊 Результаты:")
            print(f"   • Обработано кадров: {summary['frames_processed']}")
            print(f"   • Обнаружено актов взвешивания: {summary['act_stats']['completed_acts_count']}")
            print(f"   • Общее количество проходов: {summary['crossing_stats']['total_crossings']}")
            print(f"   • Проходы слева: {summary['crossing_stats']['left_crossings']}")
            print(f"   • Проходы справа: {summary['crossing_stats']['right_crossings']}")
            print(f"   • Пиковое количество одновременно: {summary['act_stats']['peak_concurrent']}")
            
            # Сохраняем результаты
            if summary['act_stats']['completed_acts_count'] > 0:
                # Сохранение в JSON (всегда)
                import json
                results_dir = Path('results')
                results_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                json_path = results_dir / f"{video_path.stem}_{timestamp}_results.json"
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"\n💾 Результаты сохранены в JSON: {json_path}")
                
                # Сохранение в базу данных (если доступна)
                if self.db:
                    print("\n💾 Сохранение результатов в базу данных...")
                    
                    for act in summary['act_stats']['completed_acts']:
                        # Конвертируем в формат для базы данных
                        from pig_tracking.database import WeighingAct, CrossingEvent
                        
                        db_act = WeighingAct(
                            started_at=act['started_at'],
                            ended_at=act['ended_at'],
                            duration_sec=act['duration_sec'],
                            left_count=act['left_count'],
                            right_count=act['right_count'],
                            peak_count=act['peak_count'],
                            total_weight=None,
                            avg_weight=None,
                            stream_id=video_path.stem,
                            video_file=video_path.name
                        )
                        
                        # Добавляем проходы
                        for crossing in act.get('crossings', []):
                            db_crossing = CrossingEvent(
                                pig_id=crossing['pig_id'],
                                direction=crossing['direction'],
                                timestamp=crossing['timestamp'],
                                line_x=crossing['line_x'],
                                line_y=crossing['line_y'],
                                weight_estimate=None,
                                stream_id=video_path.stem
                            )
                            db_act.crossings.append(db_crossing)
                        
                        # Сохраняем в базу
                        act_id = self.db.save_weighing_act(db_act)
                        logger.info(f"✅ Акт {act_id} сохранен в базу")
                    
                    print(f"✅ Сохранено {summary['act_stats']['completed_acts_count']} актов в базу данных")
                else:
                    print("⚠️ База данных недоступна, результаты сохранены только в JSON")
            else:
                print("\n⚠️ Акты взвешивания не обнаружены")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки видео: {e}", exc_info=True)
            raise
    
    async def run_async(self, args):
        """Асинхронный метод запуска приложения"""
        try:
            # Инициализация базы данных
            self.initialize_database()
            
            # Определяем видеофайл для обработки
            video_path = None
            
            if args.video:
                # Видео указано в аргументах
                video_path = Path(args.video)
                if not video_path.exists():
                    logger.error(f"❌ Видеофайл не найден: {video_path}")
                    return False
            else:
                # Интерактивный выбор видео
                video_path = self.video_selector.select_video_interactive()
                if not video_path:
                    return False
            
            # Обработка видео
            await self.process_video(video_path)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ Обработка прервана пользователем")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения: {e}")
            return False
    
    def run(self, args):
        """Основной метод запуска приложения (синхронная обертка)"""
        import asyncio
        return asyncio.run(self.run_async(args))

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Система автоматического отслеживания свиней',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python console_app.py                           # Интерактивный выбор видео
  python console_app.py --video uploads/test.mp4 # Обработка конкретного файла
  
Перед запуском убедитесь что Supabase запущен:
  docker-compose up -d
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Путь к видеофайлу для обработки'
    )
    
    parser.add_argument(
        '--mode',
        choices=['process', 'test'],
        default='process',
        help='Режим работы (process - обычная обработка, test - тестовый режим)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Включить отладочный режим'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🐛 Отладочный режим включен")
    
    # Загрузка переменных окружения
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.debug("📄 Переменные окружения загружены из .env")
    except ImportError:
        logger.warning("⚠️ python-dotenv не установлен, переменные окружения не загружены")
    
    # Проверка переменных окружения
    if not os.getenv('SUPABASE_KEY'):
        logger.error("❌ SUPABASE_KEY не найден в переменных окружения")
        logger.error("   Скопируйте .env.example в .env и настройте параметры")
        return 1
    
    # Запуск приложения
    app = PigTrackingApp()
    success = app.run(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())