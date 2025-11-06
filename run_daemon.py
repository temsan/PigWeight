#!/usr/bin/env python3
"""
Запуск PigWeight в режиме демона (background service)
Для непрерывного мониторинга камер на сервере
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import time
import signal
import json

# Добавляем корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Rich для красивого вывода
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False

console = Console() if HAVE_RICH else None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('logs/daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DaemonConfig:
    """Конфигурация для демона"""
    
    def __init__(self):
        self.cameras = {
            'cam101': {
                'rtsp': os.getenv('RTSP_URL_CAM101', 'rtsp://localhost:8554/cam101'),
                'mode': 'monitor',
                'confidence': 0.30,
                'min_pigs': 3,
                'max_interval': 30.0,
                'continuous': True
            },
            'cam102': {
                'rtsp': os.getenv('RTSP_URL_CAM102', 'rtsp://localhost:8554/cam102'),
                'mode': 'monitor',
                'confidence': 0.30,
                'min_pigs': 3,
                'max_interval': 30.0,
                'continuous': True
            }
        }
    
    def to_dict(self):
        """Преобразовать в словарь"""
        return self.cameras
    
    def save(self, filename='daemon_config.json'):
        """Сохранить конфигурацию в JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Конфигурация сохранена в {filename}")


class PigWeightDaemon:
    """Демон для непрерывного мониторинга камер"""
    
    def __init__(self, camera_id: str, config: dict):
        self.camera_id = camera_id
        self.config = config
        self.process = None
        self.running = False
        self.restart_count = 0
        self.max_restarts = 10
        self.restart_delay = 30  # секунд
    
    def start(self):
        """Запустить демон"""
        if self.running:
            logger.warning(f"Демон {self.camera_id} уже запущен")
            return
        
        logger.info(f"🚀 Запуск демона для {self.camera_id}...")
        
        cmd = [
            sys.executable,
            'console_app.py',
            '--mode', self.config.get('mode', 'monitor'),
            '--rtsp', self.config.get('rtsp'),
            '--confidence', str(self.config.get('confidence', 0.30)),
            '--min-pigs', str(self.config.get('min_pigs', 3)),
            '--max-interval', str(self.config.get('max_interval', 30)),
        ]
        
        if self.config.get('continuous'):
            cmd.append('--continuous')
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.running = True
            self.restart_count = 0
            logger.info(f"✅ Демон {self.camera_id} запущен (PID: {self.process.pid})")
        except Exception as e:
            logger.error(f"❌ Ошибка при запуске демона {self.camera_id}: {e}")
            self.running = False
    
    def stop(self):
        """Остановить демон"""
        if not self.running or not self.process:
            return
        
        logger.info(f"🛑 Остановка демона для {self.camera_id}...")
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
            logger.info(f"✅ Демон {self.camera_id} остановлен")
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Принудительное завершение демона {self.camera_id}")
            self.process.kill()
        finally:
            self.running = False
    
    def is_running(self) -> bool:
        """Проверить, работает ли демон"""
        if not self.process:
            return False
        
        return self.process.poll() is None
    
    def auto_restart(self):
        """Автоматический перезапуск при падении"""
        if self.is_running():
            return
        
        if self.restart_count >= self.max_restarts:
            logger.error(f"❌ Демон {self.camera_id} достиг максимума перезапусков ({self.max_restarts})")
            return
        
        logger.warning(f"⚠️ Демон {self.camera_id} упал, перезапуск через {self.restart_delay}с...")
        self.restart_count += 1
        time.sleep(self.restart_delay)
        self.start()


class DaemonManager:
    """Менеджер демонов"""
    
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.daemons = {}
        self.running = True
        
        # Обработчики сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов (Ctrl+C, etc.)"""
        logger.info("\n⏹️ Получен сигнал завершения...")
        self.stop_all()
        sys.exit(0)
    
    def start_all(self):
        """Запустить все демоны"""
        if HAVE_RICH:
            console.print(Panel(
                "[bold cyan]🐷 PigWeight Daemon Manager[/bold cyan]\n"
                "[white]Запуск мониторинга камер[/white]",
                border_style="cyan",
                box=box.DOUBLE
            ))
        
        logger.info("=" * 70)
        logger.info("🚀 Запуск всех демонов...")
        logger.info("=" * 70)
        
        for camera_id, cfg in self.config.to_dict().items():
            daemon = PigWeightDaemon(camera_id, cfg)
            daemon.start()
            self.daemons[camera_id] = daemon
            time.sleep(2)  # Небольшая задержка между запусками
        
        # Показываем статус
        self._print_status()
    
    def stop_all(self):
        """Остановить все демоны"""
        logger.info("\n" + "=" * 70)
        logger.info("🛑 Остановка всех демонов...")
        logger.info("=" * 70)
        
        for camera_id, daemon in self.daemons.items():
            daemon.stop()
        
        logger.info("✅ Все демоны остановлены")
    
    def _print_status(self):
        """Вывести статус демонов"""
        if HAVE_RICH:
            table = Table(box=box.ROUNDED, title="📊 Статус демонов")
            table.add_column("Камера", style="cyan")
            table.add_column("Статус", style="green")
            table.add_column("PID", style="yellow")
            table.add_column("Перезапусков", style="magenta")
            
            for camera_id, daemon in self.daemons.items():
                status = "🟢 Активен" if daemon.is_running() else "🔴 Остановлен"
                pid = str(daemon.process.pid) if daemon.process else "-"
                restarts = str(daemon.restart_count)
                table.add_row(camera_id, status, pid, restarts)
            
            console.print(table)
        else:
            print("\n" + "=" * 70)
            print("СТАТУС ДЕМОНОВ")
            print("=" * 70)
            for camera_id, daemon in self.daemons.items():
                status = "✓ Активен" if daemon.is_running() else "✗ Остановлен"
                pid = daemon.process.pid if daemon.process else "-"
                print(f"  {camera_id}: {status} (PID: {pid}, Перезапусков: {daemon.restart_count})")
    
    def monitor(self):
        """Основной цикл мониторинга"""
        logger.info("\n✅ Мониторинг запущен. Нажмите Ctrl+C для выхода")
        
        check_interval = 30  # Проверка каждые 30 секунд
        last_status_print = 0
        status_print_interval = 300  # Вывод статуса каждые 5 минут
        
        try:
            while self.running:
                current_time = time.time()
                
                # Проверка здоровья демонов
                for camera_id, daemon in self.daemons.items():
                    if not daemon.is_running():
                        daemon.auto_restart()
                
                # Периодический вывод статуса
                if current_time - last_status_print >= status_print_interval:
                    logger.info("\n📊 Периодический отчёт о статусе:")
                    self._print_status()
                    last_status_print = current_time
                
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='PigWeight Daemon Manager - запуск демонов мониторинга камер',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Запустить все демоны
  python run_daemon.py --start

  # Остановить все демоны
  python run_daemon.py --stop

  # Показать статус
  python run_daemon.py --status

  # Сохранить конфигурацию
  python run_daemon.py --save-config daemon_config.json

  # Загрузить конфигурацию
  python run_daemon.py --load-config daemon_config.json --start
        """
    )
    
    parser.add_argument('--start', action='store_true', help='Запустить демоны')
    parser.add_argument('--stop', action='store_true', help='Остановить демоны')
    parser.add_argument('--status', action='store_true', help='Показать статус')
    parser.add_argument('--save-config', type=str, help='Сохранить конфигурацию в файл')
    parser.add_argument('--load-config', type=str, help='Загрузить конфигурацию из файла')
    parser.add_argument('--monitor', action='store_true', help='Запустить в режиме мониторинга')
    
    args = parser.parse_args()
    
    # Создаём конфигурацию
    config = DaemonConfig()
    
    # Обработка команд
    if args.save_config:
        config.save(args.save_config)
        print(f"✅ Конфигурация сохранена в {args.save_config}")
        return 0
    
    if args.start:
        manager = DaemonManager(config)
        manager.start_all()
        
        if args.monitor:
            manager.monitor()
        else:
            print("\n✅ Демоны запущены в фоне")
            print("Используйте 'python run_daemon.py --status' для проверки статуса")
        
        return 0
    
    if args.stop:
        manager = DaemonManager(config)
        # Пытаемся подключиться к существующим демонам и остановить их
        print("❌ Остановка всех демонов не реализована")
        print("Используйте killall python или Ctrl+C в процессе мониторинга")
        return 0
    
    if args.status:
        print("\n📊 Для проверки статуса запустите с флагом --monitor")
        return 0
    
    # По умолчанию - интерактивный режим
    if not any([args.start, args.stop, args.status, args.save_config]):
        print("🐷 PigWeight Daemon Manager")
        print("=" * 70)
        print("\nДоступные команды:")
        print("  --start              Запустить демоны")
        print("  --monitor            Запустить демоны с мониторингом")
        print("  --status             Показать статус")
        print("  --save-config FILE   Сохранить конфигурацию")
        print("  --load-config FILE   Загрузить конфигурацию")
        print("\nПримеры:")
        print("  python run_daemon.py --start --monitor")
        print("  python run_daemon.py --save-config my_daemon.json")
        return 0


if __name__ == "__main__":
    sys.exit(main())

