"""
Оптимизированная версия PigWeight с интеграцией всех новых компонентов
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional

# Ранняя загрузка конфигурации
try:
    from dotenv import load_dotenv
    # Загружаем оптимизированную конфигурацию
    env_file = os.getenv('CONFIG_FILE', '.env.optimized')
    if Path(env_file).exists():
        load_dotenv(env_file)
        print(f"[OK] Загружена конфигурация: {env_file}")
    else:
        load_dotenv()  # Fallback на дефолтный .env
except Exception as e:
    print(f"[WARNING] Ошибка загрузки .env: {e}")

# Импорт упрощенной системы логирования
from core.config import setup_logging
logger = setup_logging(debug=os.getenv("DEBUG", "false").lower() == "true")

# Импорт оптимизированных компонентов
try:
    from core.optimized_config import get_config, set_config, apply_performance_profile, OptimizedConfig
    from core.performance_monitor import PerformanceMonitor, MonitorConfig
    from core.adaptive_quality_controller import AdaptiveQualityController
    # from core.async_rtsp_decoder import AsyncRTSPDecoder # DEPRECATED
    from core.priority_frame_queue import PriorityFrameQueue
    from core.dynamic_batcher import DynamicBatcher
    from core.h264_direct_track import H264DirectTrack, H264StreamAdapter
    
    OPTIMIZED_COMPONENTS_AVAILABLE = True
    logger.info("[OK] Все оптимизированные компоненты успешно импортированы")
except ImportError as e:
    logger.error(f"Не удалось импортировать оптимизированные компоненты: {e}")
    logger.info("Попытка установки недостающих зависимостей...")
    OPTIMIZED_COMPONENTS_AVAILABLE = False
    
    # Определяем заглушки для случая отсутствия компонентов
    class OptimizedConfig:
        def __init__(self):
            self.cuda_enabled = False
            
    def get_config():
        return OptimizedConfig()
        
    def apply_performance_profile(profile):
        logger.warning(f"Профиль {profile} не может быть применен - компоненты недоступны")

class OptimizedPigWeightServer:
    """Оптимизированный сервер PigWeight с новыми компонентами"""
    
    def __init__(self, config):
        self.config = config
        
        # Компоненты системы
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.quality_controller: Optional[AdaptiveQualityController] = None
        self.frame_queue: Optional[PriorityFrameQueue] = None
        self.batcher: Optional[DynamicBatcher] = None
        
        # Состояние сервера
        self._running = False
        self._startup_tasks = []
        
        # FastAPI app (будет инициализирован позже)
        self._app = None
        
    async def initialize(self):
        """Инициализация всех компонентов"""
        logger.info("[START] Инициализация оптимизированного сервера PigWeight...")
        
        try:
            # 1. Проверка системных требований
            await self._check_system_requirements()
            
            # 2. Создание директорий
            self._ensure_directories()
            
            # 3. Инициализация мониторинга производительности
            await self._init_performance_monitor()
            
            # 4. Инициализация контроллера качества
            await self._init_quality_controller()
            
            # 5. Инициализация очереди кадров
            await self._init_frame_queue()
            
            # 6. Инициализация батчера
            await self._init_batcher()
            
            # 7. Инициализация FastAPI приложения
            await self._init_fastapi_app()
            
            logger.info("[OK] Все компоненты успешно инициализированы")
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка инициализации: {e}")
            raise
            
    async def start(self):
        """Запуск сервера"""
        if self._running:
            logger.warning("Сервер уже запущен")
            return
            
        logger.info("[START] Запуск оптимизированного сервера...")
        
        try:
            # Запуск всех компонентов
            await self._start_components()
            
            # Запуск FastAPI сервера
            await self._start_fastapi_server()
            
            self._running = True
            logger.info("[READY] Сервер успешно запущен и готов к работе")
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска сервера: {e}")
            await self.stop()
            raise
            
    async def stop(self):
        """Остановка сервера"""
        if not self._running:
            return
            
        logger.info("[STOP] Остановка сервера...")
        
        try:
            # Остановка компонентов в обратном порядке
            if self.batcher:
                await self.batcher.stop()
                
            if self.frame_queue:
                await self.frame_queue.stop()
                
            if self.quality_controller:
                await self.quality_controller.stop()
                
            if self.performance_monitor:
                await self.performance_monitor.stop()
                
            self._running = False
            logger.info("✅ Сервер остановлен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при остановке: {e}")
            
    async def _check_system_requirements(self):
        """Проверка системных требований"""
        logger.info("[CHECK] Проверка системных требований...")
        
        # Проверка CUDA если включен
        if self.config.cuda_enabled:
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    device_name = torch.cuda.get_device_name(self.config.cuda_device)
                    memory = torch.cuda.get_device_properties(self.config.cuda_device).total_memory / (1024**3)
                    logger.info(f"[CUDA] CUDA готов: {device_name} ({memory:.1f} GB)")
                else:
                    logger.warning("[WARNING] CUDA недоступен, переключаемся на CPU")
                    self.config.cuda_enabled = False
                    # Автоматическое переключение на ONNX Runtime при отсутствии GPU
                    if hasattr(self.config, 'auto_fallback_to_onnx') and self.config.auto_fallback_to_onnx:
                        await self._setup_onnx_runtime()
            except ImportError:
                logger.warning("[WARNING] PyTorch не установлен, отключаем CUDA")
                self.config.cuda_enabled = False
                # Автоматическое переключение на ONNX Runtime
                if hasattr(self.config, 'auto_fallback_to_onnx') and self.config.auto_fallback_to_onnx:
                    await self._setup_onnx_runtime()
        
        # Проверка ONNX Runtime если настроен
        if hasattr(self.config, 'use_onnx_runtime') and self.config.use_onnx_runtime:
            await self._setup_onnx_runtime()
                
        # Проверка доступности GPU мониторинга
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                logger.info(f"[GPU] Мониторинг GPU доступен: {len(gpus)} устройств")
        except ImportError:
            logger.warning("[WARNING] GPUtil недоступен, мониторинг GPU отключен")
            
        # Проверка WebRTC компонентов
        try:
            import aiortc
            logger.info("[WEBRTC] aiortc доступен для WebRTC")
        except ImportError:
            logger.warning("[WARNING] aiortc недоступен, WebRTC отключен")
    
    async def _setup_onnx_runtime(self):
        """Настройка ONNX Runtime для CPU оптимизации"""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            # Настройка провайдеров
            providers = []
            if 'CUDAExecutionProvider' in available_providers and self.config.cuda_enabled:
                providers.append('CUDAExecutionProvider')
                logger.info("[ONNX] ONNX Runtime с CUDA провайдером готов")
            
            if 'CPUExecutionProvider' in available_providers:
                providers.append('CPUExecutionProvider')
                logger.info("[ONNX] ONNX Runtime с CPU провайдером готов")
            
            if not providers:
                logger.error("[ONNX] Нет доступных провайдеров ONNX Runtime")
                return False
            
            # Обновляем конфигурацию
            if hasattr(self.config, 'onnx_providers'):
                self.config.onnx_providers = providers
                self.config.use_onnx_runtime = True
                logger.info(f"[ONNX] Активные провайдеры: {providers}")
            
            return True
            
        except ImportError:
            logger.error("[ERROR] ONNX Runtime не установлен")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Ошибка настройки ONNX Runtime: {e}")
            return False
            logger.warning("[WARNING] aiortc недоступен, WebRTC функции ограничены")
            
    def _ensure_directories(self):
        """Создание необходимых директорий"""
        directories = [
            'logs',
            'models', 
            'uploads',
            'stream',
            'static',
            'core'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    async def _init_performance_monitor(self):
        """Инициализация мониторинга производительности"""
        logger.info("[MONITOR] Инициализация PerformanceMonitor...")
        
        self.performance_monitor = PerformanceMonitor(self.config.monitor_config)
        await self.performance_monitor.start()
        
        # Регистрируем callback для обновления метрик
        self.update_performance_metrics = self.performance_monitor.update_external_metrics
        
    async def _init_quality_controller(self):
        """Инициализация контроллера качества"""
        logger.info("[QUALITY] Инициализация AdaptiveQualityController...")
        
        def quality_change_callback(new_settings):
            logger.info(f"[QUALITY] Изменение качества: {new_settings.level.name}")
            # Здесь можно добавить логику применения новых настроек
            
        self.quality_controller = AdaptiveQualityController(
            self.config.quality_config,
            quality_change_callback
        )
        await self.quality_controller.start()
        
    async def _init_frame_queue(self):
        """Инициализация приоритетной очереди кадров"""
        logger.info("[QUEUE] Инициализация PriorityFrameQueue...")
        
        self.frame_queue = PriorityFrameQueue(self.config.queue_config)
        await self.frame_queue.start()
        
    async def _init_batcher(self):
        """Инициализация динамического батчера"""
        logger.info("[BATCHER] Инициализация DynamicBatcher...")
        
        async def process_batch(frames):
            # Здесь будет логика обработки батча кадров
            # Подключается к существующей inference системе
            logger.debug(f"Обработка батча из {len(frames)} кадров")
            
        self.batcher = DynamicBatcher(self.config.batcher_config, process_batch)
        await self.batcher.start()
        
    async def _init_fastapi_app(self):
        """Инициализация FastAPI приложения"""
        logger.info("[FASTAPI] Инициализация FastAPI приложения...")
        
        # Импорт и настройка приложения
        from api.app import app as base_app
        from api.optimized_endpoints import router as optimized_router
        
        # Добавление оптимизированных endpoints
        base_app.include_router(optimized_router, prefix="/api/v2")
        
        # Добавление middleware для мониторинга
        await self._setup_monitoring_middleware(base_app)
        
        self._app = base_app
        
    async def _setup_monitoring_middleware(self, app):
        """Настройка middleware для мониторинга"""
        from fastapi import Request, Response
        import time
        
        @app.middleware("http")
        async def performance_monitoring_middleware(request: Request, call_next):
            start_time = time.time()
            
            response: Response = await call_next(request)
            
            process_time = time.time() - start_time
            
            # Обновление метрик производительности
            if self.performance_monitor:
                metrics = {
                    'request_latency_ms': process_time * 1000,
                    'requests_per_second': 1.0 / process_time if process_time > 0 else 0,
                    'status_code': response.status_code
                }
                self.performance_monitor.update_external_metrics(metrics)
                
            return response
            
    async def _start_components(self):
        """Запуск всех компонентов"""
        logger.info("[START] Запуск компонентов...")
        
        # Компоненты уже запущены в init методах
        # Здесь можем добавить дополнительную логику если нужно
        
    async def _start_fastapi_server(self):
        """Запуск FastAPI сервера"""
        import uvicorn
        
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        
        logger.info(f"[SERVER] Запуск сервера на {host}:{port}")
        
        # Запуск в отдельной задаче для возможности остановки
        config = uvicorn.Config(
            self._app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    def get_stats(self) -> dict:
        """Получение статистики всех компонентов"""
        stats = {
            'server_running': self._running,
            'config': self.config.to_dict(),
            'performance_monitor': None,
            'quality_controller': None, 
            'frame_queue': None,
            'batcher': None
        }
        
        if self.performance_monitor:
            stats['performance_monitor'] = self.performance_monitor.get_stats()
            
        if self.quality_controller:
            stats['quality_controller'] = self.quality_controller.get_stats()
            
        if self.frame_queue:
            stats['frame_queue'] = self.frame_queue.get_stats()
            
        if self.batcher:
            stats['batcher'] = self.batcher.get_stats()
            
        return stats

async def main_async():
    """Асинхронная главная функция"""
    
    parser = argparse.ArgumentParser(description='PigWeight Optimized Server')
    parser.add_argument('--config', default='.env.optimized', help='Файл конфигурации')
    parser.add_argument('--profile', choices=['ULTRA_PERFORMANCE', 'BALANCED', 'POWER_SAVING', 'MINIMAL_RESOURCE'], 
                       help='Профиль производительности')
    parser.add_argument('--install', action='store_true', help='Установить зависимости')
    parser.add_argument('--validate-config', action='store_true', help='Проверить конфигурацию')
    
    args = parser.parse_args()
    
    if args.install:
        await install_optimized_dependencies()
        return
        
    # Применение профиля производительности
    if args.profile:
        logger.info(f"[PROFILE] Применение профиля: {args.profile}")
        apply_performance_profile(args.profile)
    else:
        # Применяем BALANCED профиль по умолчанию
        logger.info("[PROFILE] Применение профиля по умолчанию: BALANCED")
        apply_performance_profile('BALANCED')
        
    # Загрузка конфигурации
    config = OptimizedConfig.from_env(args.config)
    
    if args.validate_config:
        if config.validate():
            logger.info("[OK] Конфигурация валидна")
        else:
            logger.error("[ERROR] Конфигурация содержит ошибки")
            return
        
    # Создание и запуск сервера
    server = OptimizedPigWeightServer(config)
    
    try:
        await server.initialize()
        await server.start()
    except KeyboardInterrupt:
        logger.info("[STOP] Получен сигнал остановки...")
    except Exception as e:
        logger.error(f"[ERROR] Критическая ошибка: {e}")
        raise
    finally:
        await server.stop()

def main():
    """Синхронная обертка для асинхронного main"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("[BYE] Сервер остановлен пользователем")
    except Exception as e:
        logger.error(f"[ERROR] Ошибка запуска: {e}")
        sys.exit(1)

async def install_optimized_dependencies():
    """Установка оптимизированных зависимостей"""
    logger.info("[INSTALL] Установка оптимизированных зависимостей...")
    
    import subprocess
    import sys
    
    optimized_packages = [
        "torch>=2.1.0",
        "torchvision>=0.16.0", 
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=11.0",
        "aiortc>=1.5.0",
        "av>=10.0.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0"
    ]
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade"
        ] + optimized_packages)
        
        logger.info("[OK] Оптимизированные зависимости установлены")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Ошибка установки: {e}")
        raise

if __name__ == "__main__":
    if not OPTIMIZED_COMPONENTS_AVAILABLE:
        logger.error("[ERROR] Оптимизированные компоненты недоступны. Установите зависимости:")
        logger.error("   pip install ultralytics torch torchvision opencv-python fastapi uvicorn")
        logger.error("   Или запустите: python main_optimized.py --install")
        sys.exit(1)
    else:
        main()