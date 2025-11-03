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
import asyncio
import json

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

# Импорты из существующей системы
from core.config import get_config
from pig_tracking.database import DatabaseManager, WeighingAct, CrossingEvent

# Rich для красивого TUI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.syntax import Syntax
    from rich.align import Align
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    from rich.spinner import Spinner
    from rich.status import Status
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False

# Questionary для интерактивных меню со стрелками
try:
    import questionary
    HAVE_QUESTIONARY = True
except ImportError:
    HAVE_QUESTIONARY = False


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальная консоль Rich
console = Console() if HAVE_RICH else None


class RichFormatter:
    """Форматер для красивого вывода Rich"""
    
    @staticmethod
    def print_header(title: str, subtitle: str = ""):
        """Печатает красивый заголовок"""
        if not HAVE_RICH:
            print(f"\n{'='*80}\n{title}\n{'='*80}\n")
            return
        
        content = f"[bold magenta]{title}[/bold magenta]"
        if subtitle:
            content += f"\n[dim]{subtitle}[/dim]"
        
        panel = Panel(
            content,
            border_style="magenta",
            padding=(1, 2),
            expand=False
        )
        console.print(panel)
    
    @staticmethod
    def print_success(message: str):
        """Печатает успешное сообщение"""
        if not HAVE_RICH:
            print(f"✓ {message}")
            return
        console.print(f"[green]✓[/green] [bold green]{message}[/bold green]")
    
    @staticmethod
    def print_error(message: str):
        """Печатает ошибку"""
        if not HAVE_RICH:
            print(f"✗ {message}")
            return
        console.print(f"[red]✗[/red] [bold red]{message}[/bold red]")
    
    @staticmethod
    def print_warning(message: str):
        """Печатает предупреждение"""
        if not HAVE_RICH:
            print(f"⚠ {message}")
            return
        console.print(f"[yellow]⚠[/yellow] [bold yellow]{message}[/bold yellow]")
    
    @staticmethod
    def print_info(message: str):
        """Печатает информацию"""
        if not HAVE_RICH:
            print(f"ℹ {message}")
            return
        console.print(f"[cyan]ℹ[/cyan] [bold cyan]{message}[/bold cyan]")
    
    @staticmethod
    def print_table(title: str, columns: List[str], rows: List[List[str]], styles: List[str] = None):
        """Печатает красивую таблицу"""
        if not HAVE_RICH:
            # Simple table without Rich
            col_widths = [max(len(str(col)), max(len(str(row[i])) for row in rows)) for i, col in enumerate(columns)]
            header = " | ".join(f"{col:<{width}}" for col, width in zip(columns, col_widths))
            print(f"\n{title}")
            print("=" * len(header))
            print(header)
            print("-" * len(header))
            for row in rows:
                print(" | ".join(f"{str(val):<{width}}" for val, width in zip(row, col_widths)))
            return
        
        table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")
        
        if styles is None:
            styles = ["cyan"] * len(columns)
        
        for col, style in zip(columns, styles):
            table.add_column(col, style=style)
        
        for row in rows:
            table.add_row(*[str(val) for val in row])
        
        console.print(table)
    
    @staticmethod
    def create_progress_bar(title: str = "Обработка"):
        """Создает progress bar с красивым форматированием"""
        if not HAVE_RICH:
            return None
        
        return Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False
        )
    
    @staticmethod
    def print_acts_metrics(acts_data: List[dict]):
        """Печатает красивую таблицу метрик актов взвешивания"""
        if not HAVE_RICH:
            # Simple output without Rich
            print("\n📊 МЕТРИКИ АКТОВ ВЗВЕШИВАНИЯ:")
            print("=" * 120)
            for i, act in enumerate(acts_data, 1):
                print(f"\nАкт #{i}:")
                print(f"  Время начала:     {act.get('started_at_iso', '-')}")
                print(f"  Время окончания:  {act.get('ended_at_iso', '-')}")
                print(f"  Длительность:     {act.get('duration', 0):.1f} сек")
                print(f"  Зашло слева:      {act.get('left_count', 0)} шт")
                print(f"  Вышло справа:     {act.get('right_count', 0)} шт")
                print(f"  Пиковое кол-во:   {act.get('peak_count', 0)} шт")
                print(f"  Всего свиней:     {act.get('seen_total', 0)} шт")
                print(f"  Всего вес:        {act.get('total_weight', 0):.1f} кг" if act.get('total_weight') else "  Всего вес:        не определён")
                print(f"  Средний вес:      {act.get('avg_weight', 0):.1f} кг" if act.get('avg_weight') else "  Средний вес:      не определён")
            return
        
        # Rich таблица с красивым форматированием
        table = Table(title="📊 Метрики актов взвешивания", box=box.ROUNDED, show_header=True)
        
        table.add_column("Акт", style="magenta", width=6)
        table.add_column("Время начала", style="cyan", width=20)
        table.add_column("Длит-ть (сек)", style="blue", width=12)
        table.add_column("Слева↙️", style="yellow", width=8)
        table.add_column("Справа↗️", style="yellow", width=8)
        table.add_column("Пик кол-во", style="green", width=10)
        table.add_column("Всего шт", style="white", width=8)
        table.add_column("Вес (кг)", style="cyan", width=10)
        table.add_column("Ср.вес (кг)", style="cyan", width=10)
        
        for i, act in enumerate(acts_data, 1):
            duration = act.get('duration', 0)
            total_weight = act.get('total_weight')
            avg_weight = act.get('avg_weight')
            
            weight_str = f"{total_weight:.1f}" if total_weight else "—"
            avg_str = f"{avg_weight:.1f}" if avg_weight else "—"
            
            table.add_row(
                str(i),
                act.get('started_at_iso', '—')[:19],
                f"{duration:.1f}",
                str(act.get('left_count', 0)),
                str(act.get('right_count', 0)),
                str(act.get('peak_count', 0)),
                str(act.get('seen_total', 0)),
                weight_str,
                avg_str
            )
        
        console.print()
        console.print(table)
    
    @staticmethod
    def print_summary_stats(summary: dict):
        """Печатает итоговую статистику обработки"""
        if not HAVE_RICH:
            print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
            print("=" * 60)
            print(f"Обработано кадров:        {summary.get('frames_processed', 0)}")
            print(f"Обнаружено актов:         {summary.get('act_stats', {}).get('completed_acts_count', 0)}")
            print(f"Всего проходов:           {summary.get('crossing_stats', {}).get('total_crossings', 0)}")
            print(f"Проходы слева:            {summary.get('crossing_stats', {}).get('left_crossings', 0)}")
            print(f"Проходы справа:           {summary.get('crossing_stats', {}).get('right_crossings', 0)}")
            print(f"Пиковое кол-во:           {summary.get('act_stats', {}).get('peak_concurrent', 0)}")
            return
        
        # Rich таблица итогов
        act_stats = summary.get('act_stats', {})
        crossing_stats = summary.get('crossing_stats', {})
        
        table = Table(title="📊 Итоговая статистика", box=box.DOUBLE, show_header=False)
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", style="green", justify="right")
        
        table.add_row("🎬 Обработано кадров", str(summary.get('frames_processed', 0)))
        table.add_row("🐷 Обнаружено актов взвешивания", str(act_stats.get('completed_acts_count', 0)))
        table.add_row("📍 Всего пересечений линий", str(crossing_stats.get('total_crossings', 0)))
        table.add_row("↙️ Пересечений слева", str(crossing_stats.get('left_crossings', 0)))
        table.add_row("↗️ Пересечений справа", str(crossing_stats.get('right_crossings', 0)))
        table.add_row("📈 Пиковое количество одновременно", str(act_stats.get('peak_concurrent', 0)))
        
        console.print()
        console.print(table)


class VideoSelector:
    """Класс для выбора видеофайлов и камер"""
    
    def __init__(self, uploads_dir: str = "uploads"):
        self.uploads_dir = Path(uploads_dir)
        if not self.uploads_dir.exists():
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.cameras = self._load_cameras()
    
    def _load_cameras(self) -> dict:
        """Загружает список камер из .env"""
        cameras = {}
        
        # Ищем переменные CAM_CH*
        for key, value in os.environ.items():
            if key.startswith('CAM_CH') and value:
                cam_id = key.replace('CAM_CH', '')
                cameras[f"cam{cam_id}"] = {
                    'name': f"Камера {cam_id}",
                    'url': value
                }
        
        # Fallback на CAM_URL или CAM_DEFAULT
        if not cameras:
            cam_url = os.getenv('CAM_URL') or os.getenv('CAM_DEFAULT')
            if cam_url:
                cameras['cam101'] = {
                    'name': 'Камера 101',
                    'url': cam_url
                }
        
        return cameras
    
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
    
    def select_source_interactive(self) -> Optional[dict]:
        """Интерактивный выбор источника (видео или камера) с красивым TUI и навигацией стрелками"""
        # Пытаемся использовать questionary для красивого меню со стрелками
        if HAVE_QUESTIONARY:
            return self._select_source_questionary()
        elif HAVE_RICH:
            return self._select_source_rich()
        else:
            return self._select_source_simple()
    
    def _select_source_questionary(self) -> Optional[dict]:
        """Интерактивное меню со стрелками (questionary)"""
        video_files = self.get_video_files()
        
        sources = []
        choices = []
        
        # Добавляем камеры
        if self.cameras:
            for cam_id, cam_info in self.cameras.items():
                sources.append({
                    'type': 'camera',
                    'id': cam_id,
                    'name': cam_info['name'],
                    'url': cam_info['url']
                })
                choices.append(f"🎥 {cam_info['name']} (RTSP поток)")
        
        # Добавляем видеофайлы
        for video_file in video_files:
            info = self.get_file_info(video_file)
            sources.append({
                'type': 'file',
                'path': video_file
            })
            choices.append(f"📁 {video_file.name} ({info['size']} • {info['duration']})")
        
        if not sources:
            if HAVE_RICH:
                console.print(Panel(
                    "[red]❌ Нет доступных источников[/red]\n\n"
                    f"Поместите видеофайлы в папку [cyan]{self.uploads_dir.absolute()}[/cyan]\n"
                    "или настройте камеры в .env (CAM_CH101, CAM_CH102, ...)",
                    title="Ошибка",
                    border_style="red"
                ))
            else:
                print(f"❌ Нет доступных источников")
            return None
        
        # Красивый заголовок
        if HAVE_RICH:
            console.print()
            console.print(Panel(
                "[bold magenta]🐷 PigWeight - Выбор источника[/bold magenta]\n"
                "[dim]Используйте стрелки ↑↓ для навигации, Enter для выбора[/dim]",
                border_style="magenta"
            ))
        
        # Интерактивное меню со стрелками
        try:
            answer = questionary.select(
                "Выберите источник для обработки:",
                choices=choices,
                pointer="→ ",
                use_shortcuts=True,
                use_arrow_keys=True,
                use_pointers=True
            ).ask()
            
            if answer is None:
                if HAVE_RICH:
                    console.print("[yellow]Выход...[/yellow]")
                else:
                    print("Выход...")
                return None
            
            # Находим индекс выбранного варианта
            index = choices.index(answer)
            source = sources[index]
            
            # Выводим подтверждение
            if HAVE_RICH:
                if source['type'] == 'camera':
                    console.print(f"[green]✅ Выбрана камера:[/green] [bold]{source['name']}[/bold]")
                else:
                    console.print(f"[green]✅ Выбран файл:[/green] [bold]{source['path'].name}[/bold]")
            else:
                if source['type'] == 'camera':
                    print(f"✅ Выбрана камера: {source['name']}")
                else:
                    print(f"✅ Выбран файл: {source['path'].name}")
            
            return source
            
        except (KeyboardInterrupt, EOFError):
            if HAVE_RICH:
                console.print("\n[yellow]Выход...[/yellow]")
            else:
                print("\nВыход...")
            return None
    
    def _select_source_rich(self) -> Optional[dict]:
        """Интерактивный выбор источника с Rich (fallback если нет questionary)"""
        video_files = self.get_video_files()
        
        # Создаем таблицу источников
        table = Table(title="🐷 PigWeight - Выбор источника", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("№", style="cyan", width=4)
        table.add_column("Тип", style="green", width=10)
        table.add_column("Название", style="white")
        table.add_column("Детали", style="yellow")
        
        sources = []
        
        # Добавляем камеры
        if self.cameras:
            for cam_id, cam_info in self.cameras.items():
                sources.append({
                    'type': 'camera',
                    'id': cam_id,
                    'name': cam_info['name'],
                    'url': cam_info['url']
                })
                table.add_row(
                    str(len(sources)),
                    "🎥 Камера",
                    cam_info['name'],
                    "RTSP поток"
                )
        
        # Добавляем видеофайлы
        for video_file in video_files:
            info = self.get_file_info(video_file)
            sources.append({
                'type': 'file',
                'path': video_file
            })
            table.add_row(
                str(len(sources)),
                "📁 Файл",
                video_file.name,
                f"{info['size']} • {info['duration']}"
            )
        
        if not sources:
            console.print(Panel(
                "[red]❌ Нет доступных источников[/red]\n\n"
                f"Поместите видеофайлы в папку [cyan]{self.uploads_dir.absolute()}[/cyan]\n"
                "или настройте камеры в .env (CAM_CH101, CAM_CH102, ...)",
                title="Ошибка",
                border_style="red"
            ))
            return None
        
        console.print(table)
        console.print()
        
        # Выбор источника
        while True:
            try:
                choice = Prompt.ask(
                    f"[bold cyan]Выберите источник[/bold cyan] [dim](1-{len(sources)} или q для выхода)[/dim]",
                    default="1"
                )
                
                if choice.lower() == 'q':
                    console.print("[yellow]Выход...[/yellow]")
                    return None
                
                index = int(choice) - 1
                if 0 <= index < len(sources):
                    source = sources[index]
                    
                    if source['type'] == 'camera':
                        console.print(f"[green]✅ Выбрана камера:[/green] [bold]{source['name']}[/bold]")
                    else:
                        console.print(f"[green]✅ Выбран файл:[/green] [bold]{source['path'].name}[/bold]")
                    
                    return source
                else:
                    console.print(f"[red]❌ Неверный номер. Введите число от 1 до {len(sources)}[/red]")
                    
            except ValueError:
                console.print("[red]❌ Введите число или 'q' для выхода[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Выход...[/yellow]")
                return None
    
    def _select_source_simple(self) -> Optional[dict]:
        """Простой выбор источника без Rich (fallback)"""
        video_files = self.get_video_files()
        
        print(f"\n📹 Выберите источник для обработки:")
        print("=" * 80)
        
        sources = []
        
        # Показываем камеры
        if self.cameras:
            print("\n🎥 Камеры:")
            for cam_id, cam_info in self.cameras.items():
                sources.append({
                    'type': 'camera',
                    'id': cam_id,
                    'name': cam_info['name'],
                    'url': cam_info['url']
                })
                print(f"{len(sources):2d}. {cam_info['name']} (RTSP)")
            print()
        
        # Показываем видеофайлы
        if video_files:
            print("📁 Видеофайлы:")
            for video_file in video_files:
                info = self.get_file_info(video_file)
                sources.append({
                    'type': 'file',
                    'path': video_file
                })
                print(f"{len(sources):2d}. {video_file.name}")
                print(f"    Размер: {info['size']}, Длительность: {info['duration']}")
        
        if not sources:
            print(f"❌ Нет доступных источников")
            return None
        
        print("=" * 80)
        
        while True:
            try:
                choice = input(f"\nВыберите источник (1-{len(sources)}) или 'q' для выхода: ").strip()
                
                if choice.lower() == 'q':
                    print("Выход...")
                    return None
                
                index = int(choice) - 1
                if 0 <= index < len(sources):
                    return sources[index]
                else:
                    print(f"❌ Неверный номер. Введите число от 1 до {len(sources)}")
                    
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
        """Инициализация подключения к базе данных (опционально)"""
        try:
            self.db = DatabaseManager()
            logger.info("✅ Подключение к базе данных успешно")
            
            # Показываем статистику
            stats = self.db.get_stats()
            logger.info(f"📊 В базе: {stats['total_acts']} актов, {stats['total_crossings']} проходов")
            
        except Exception as e:
            logger.info(f"ℹ️ База данных недоступна (это нормально)")
            logger.info("   Результаты будут сохранены в JSON файл")
            logger.info("   Для сохранения в БД: запустите docker-compose up -d")
            self.db = None
    
    async def process_video(self, video_path: Path):
        """Обрабатывает видеофайл"""
        logger.info(f"🎬 Начинаем обработку видео: {video_path.name}")
        
        try:
            # Импортируем IntegratedVideoProcessor
            from pig_tracking.video_processor import IntegratedVideoProcessor
            
            if HAVE_RICH:
                console.print()
                console.print(Panel(
                    f"[bold cyan]Обработка видео:[/bold cyan] [white]{video_path.name}[/white]",
                    title="🎬 PigWeight",
                    border_style="cyan",
                    box=box.DOUBLE
                ))
            else:
                print(f"\n🚀 Обработка видео: {video_path.name}")
                print("=" * 60)
            
            # Создаем процессор
            processor = IntegratedVideoProcessor(
                stream_id=video_path.stem,
                conf_threshold=self.config.CONF_THRESHOLD,
                img_size=self.config.IMG_SIZE
            )
            
            if HAVE_RICH:
                console.print("[yellow]⏳ Инициализация процессора...[/yellow]")
            else:
                print("⏳ Инициализация процессора...")
            
            await processor.initialize()
            
            if HAVE_RICH:
                console.print("[green]✓[/green] Процессор готов")
                console.print("[yellow]⏳ Начинаем обработку кадров...[/yellow]")
            else:
                print("⏳ Начинаем обработку кадров...")
            
            # Обрабатываем видео
            summary = await processor.process_video_file(str(video_path))
            
            if HAVE_RICH:
                console.print()
                RichFormatter.print_success("Обработка завершена!")
                RichFormatter.print_summary_stats(summary)
            else:
                print("\n✅ Обработка завершена!")
                print("\n📊 Результаты:")
                print(f"   • Обработано кадров: {summary['frames_processed']}")
                print(f"   • Обнаружено актов взвешивания: {summary['act_stats']['completed_acts_count']}")
                print(f"   • Общее количество проходов: {summary['crossing_stats']['total_crossings']}")
                print(f"   • Проходы слева: {summary['crossing_stats']['left_crossings']}")
                print(f"   • Проходы справа: {summary['crossing_stats']['right_crossings']}")
                print(f"   • Пиковое количество одновременно: {summary['act_stats']['peak_concurrent']}")
            
            # Выводим детальные метрики актов если есть
            if summary['act_stats']['completed_acts_count'] > 0:
                acts_data = summary['act_stats'].get('completed_acts', [])
                if acts_data and HAVE_RICH:
                    RichFormatter.print_acts_metrics(acts_data)
            
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
                
                if HAVE_RICH:
                    RichFormatter.print_success(f"Результаты сохранены в JSON: {json_path}")
                else:
                    print(f"\n💾 Результаты сохранены в JSON: {json_path}")
                
                # Сохранение в базу данных (если доступна)
                if self.db:
                    if HAVE_RICH:
                        RichFormatter.print_info("Сохранение результатов в базу данных...")
                    else:
                        print("\n💾 Сохранение результатов в базу данных...")
                    
                    try:
                        saved_count = 0
                        for act in summary['act_stats']['completed_acts']:
                            try:
                                # Конвертируем timestamp (float) в datetime
                                started_at = datetime.fromtimestamp(act['started_at'])
                                ended_at = datetime.fromtimestamp(act['ended_at']) if act.get('ended_at') else datetime.now()
                                
                                db_act = WeighingAct(
                                    started_at=started_at,
                                    ended_at=ended_at,
                                    duration_sec=act.get('duration_sec', act.get('duration', 0.0)),
                                    left_count=act.get('left_count', 0),
                                    right_count=act.get('right_count', 0),
                                    peak_count=act.get('peak_count', 0),
                                    total_weight=act.get('total_weight'),
                                    avg_weight=act.get('avg_weight'),
                                    stream_id=video_path.stem,
                                    video_file=video_path.name
                                )
                                
                                # Сохраняем в базу
                                act_id = self.db.save_weighing_act(db_act)
                                saved_count += 1
                                if HAVE_RICH:
                                    RichFormatter.print_success(f"Акт #{act['act_id']} сохранен в БД (ID: {act_id})")
                                else:
                                    logger.info(f"✅ Акт #{act['act_id']} сохранен в БД с ID {act_id}")
                                
                            except Exception as e:
                                RichFormatter.print_error(f"Ошибка сохранения акта #{act.get('act_id', '?')}: {e}")
                                continue
                        
                        if saved_count > 0:
                            if HAVE_RICH:
                                RichFormatter.print_success(f"Сохранено {saved_count} из {summary['act_stats']['completed_acts_count']} актов в БД")
                            else:
                                print(f"✅ Сохранено {saved_count} из {summary['act_stats']['completed_acts_count']} актов в базу данных")
                        else:
                            if HAVE_RICH:
                                RichFormatter.print_warning("Не удалось сохранить акты в БД")
                            else:
                                print(f"⚠️ Не удалось сохранить акты в базу данных")
                    
                    except Exception as e:
                        RichFormatter.print_error(f"Ошибка сохранения в БД: {e}")
                        print("   Результаты сохранены только в JSON")
                else:
                    if HAVE_RICH:
                        RichFormatter.print_warning("База данных недоступна, результаты сохранены только в JSON")
                    else:
                        print("⚠️ База данных недоступна, результаты сохранены только в JSON")
            else:
                if HAVE_RICH:
                    RichFormatter.print_warning("Акты взвешивания не обнаружены")
                else:
                    print("\n⚠️ Акты взвешивания не обнаружены")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки видео: {e}", exc_info=True)
            raise
    
    async def run_test_mode(self, args):
        """Тестовый режим с автоматической сверкой"""
        print("\n🧪 ТЕСТОВЫЙ РЕЖИМ")
        print("=" * 60)
        
        # Проверяем параметры
        if not args.video:
            logger.error("❌ Для тестового режима требуется --video")
            return False
        
        if not args.excel_reference:
            logger.error("❌ Для тестового режима требуется --excel-reference")
            return False
        
        video_path = Path(args.video)
        excel_path = Path(args.excel_reference)
        
        if not video_path.exists():
            logger.error(f"❌ Видеофайл не найден: {video_path}")
            return False
        
        if not excel_path.exists():
            logger.error(f"❌ Excel файл не найден: {excel_path}")
            return False
        
        # Определяем папку для результатов
        output_dir = Path(args.output) if args.output else Path('test_results')
        output_dir.mkdir(exist_ok=True)
        
        print(f"📹 Видео: {video_path.name}")
        print(f"📊 Эталон: {excel_path.name}")
        print(f"📁 Результаты: {output_dir}")
        print()
        
        # 1. Обработка видео
        print("🎬 Шаг 1: Обработка видео...")
        summary = await self.process_video(video_path)
        
        if summary['act_stats']['completed_acts_count'] == 0:
            logger.warning("⚠️ Акты не обнаружены, сверка невозможна")
            return False
        
        # 2. Сверка с Excel
        print("\n📊 Шаг 2: Сверка с эталонными данными...")
        
        try:
            from pig_tracking.excel_analyzer import ExcelAnalyzer
            from pig_tracking.excel_comparator import ExcelComparator
            
            # Читаем эталонные данные
            analyzer = ExcelAnalyzer(str(excel_path))
            analyzer.load()
            manual_acts = analyzer.parse_data()
            
            print(f"   Эталонных записей: {len(manual_acts)}")
            print(f"   Автоматических актов: {summary['act_stats']['completed_acts_count']}")
            
            # Сверка
            comparator = ExcelComparator(
                time_tolerance_minutes=5.0,
                count_tolerance_percent=10.0
            )
            
            # Конвертируем акты в нужный формат
            auto_acts = summary['act_stats']['completed_acts']
            
            # Сопоставление
            results = comparator.match_acts_by_time(auto_acts, manual_acts)
            
            # Метрики
            metrics = comparator.calculate_metrics()
            
            # Генерация отчета
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = output_dir / f"comparison_report_{timestamp}.xlsx"
            comparator.generate_report(str(report_path))
            
            # Вывод результатов
            print("\n✅ Сверка завершена!")
            print("\n📈 Метрики точности:")
            print(f"   • Recall (полнота): {metrics['recall']:.1%}")
            print(f"   • Precision (точность): {metrics['precision']:.1%}")
            print(f"   • F1-Score: {metrics['f1_score']:.1%}")
            print(f"   • MAE (средняя ошибка): {metrics['mae']:.1f}")
            print(f"   • MAPE (ошибка в %): {metrics['mape']:.1f}%")
            print(f"   • Корреляция: {metrics['correlation']:.1%}")
            
            print(f"\n📄 Отчет сохранен: {report_path}")
            
            # Сохраняем метрики в JSON
            import json
            metrics_path = output_dir / f"metrics_{timestamp}.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'video': str(video_path),
                    'excel_reference': str(excel_path),
                    'timestamp': timestamp,
                    'metrics': metrics,
                    'summary': {
                        'auto_acts': len(auto_acts),
                        'manual_acts': len(manual_acts),
                        'matched': metrics.get('matched_count', 0),
                        'missed': metrics.get('missed_count', 0)
                    }
                }, f, ensure_ascii=False, indent=2)
            
            print(f"📊 Метрики сохранены: {metrics_path}")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ Не установлены модули Excel: {e}")
            logger.error("   Установите: pip install openpyxl pandas")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка сверки: {e}", exc_info=True)
            return False
    
    async def run_async(self, args):
        """Асинхронный метод запуска приложения"""
        try:
            # Инициализация базы данных
            self.initialize_database()
            
            # Проверяем режим работы
            if args.mode == 'test':
                return await self.run_test_mode(args)
            
            # Обычный режим обработки
            source = None
            
            if args.video:
                # Видео указано в аргументах
                video_path = Path(args.video)
                if not video_path.exists():
                    logger.error(f"❌ Видеофайл не найден: {video_path}")
                    return False
                source = {'type': 'file', 'path': video_path}
            else:
                # Интерактивный выбор источника
                source = self.video_selector.select_source_interactive()
                if not source:
                    return False
            
            # Обработка в зависимости от типа источника
            if source['type'] == 'file':
                await self.process_video(source['path'])
            elif source['type'] == 'camera':
                logger.info(f"🎥 Обработка с камеры: {source['name']}")
                logger.info(f"   URL: {source['url']}")
                logger.info("⚠️ Обработка RTSP потоков пока не реализована в консольном режиме")
                logger.info("   Используйте веб-интерфейс: http://localhost:8000")
                return False
            
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
    # Красивый заголовок
    if HAVE_RICH:
        console.clear()
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🐷 PigWeight v3.0[/bold cyan]\n"
            "[white]Система автоматического отслеживания и взвешивания свиней[/white]",
            border_style="cyan",
            box=box.DOUBLE
        ))
        console.print()
    
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
        help='Режим работы (process - обычная обработка, test - тестовый режим с автоматической сверкой)'
    )
    
    parser.add_argument(
        '--excel-reference',
        type=str,
        help='Путь к Excel файлу с эталонными данными (для тестового режима)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Папка для сохранения результатов тестирования (по умолчанию: test_results)'
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
    
    # Проверка переменных окружения (опционально для БД)
    if not os.getenv('SUPABASE_KEY'):
        logger.info("ℹ️ SUPABASE_KEY не найден - работаем без БД (только JSON)")
        logger.info("   Для сохранения в БД: скопируйте .env.example в .env")
    
    # Запуск приложения
    app = PigTrackingApp()
    success = app.run(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())