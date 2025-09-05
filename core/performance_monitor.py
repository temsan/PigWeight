"""
PerformanceMonitor - Системы мониторинга производительности
WebSocket broadcast метрик, история для анализа трендов
"""

import asyncio
import logging
import time
import json
import psutil
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import statistics
import threading
from datetime import datetime, timedelta

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    timestamp: float = field(default_factory=time.time)
    
    # Системные ресурсы
    cpu_usage: float = 0.0
    memory_usage: float = 0.0  # %
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    
    # GPU метрики (если доступно)
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0  # %
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature: float = 0.0
    
    # Сеть
    network_sent_mb: float = 0.0
    network_received_mb: float = 0.0
    
    # Производительность приложения
    current_fps: float = 0.0
    target_fps: float = 30.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Concurrent streams
    active_streams: int = 0
    total_connections: int = 0
    
    # Ошибки и качество
    error_rate: float = 0.0  # errors per minute
    frame_drop_rate: float = 0.0  # %
    quality_level: str = "MEDIUM"
    
    # Throughput
    frames_processed: int = 0
    bytes_processed_mb: float = 0.0
    inference_throughput: float = 0.0  # inferences per second

@dataclass
class MonitorConfig:
    """Конфигурация мониторинга"""
    # Интервалы сбора данных
    metrics_interval: float = 1.0  # секунды
    websocket_broadcast_interval: float = 2.0  # секунды
    
    # История метрик
    max_history_minutes: int = 60  # хранить историю на час
    aggregation_intervals: List[int] = field(default_factory=lambda: [60, 300, 900])  # 1min, 5min, 15min
    
    # WebSocket сервер
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    max_websocket_clients: int = 10
    
    # Алерты
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 90.0,
        'memory_usage': 95.0,
        'gpu_usage': 95.0,
        'latency_ms': 200.0,
        'error_rate': 10.0,
        'fps_drop': 0.5  # FPS < target * 0.5
    })

class MetricsAggregator:
    """Агрегатор метрик для различных временных интервалов"""
    
    def __init__(self, intervals: List[int]):
        self.intervals = intervals
        self.buckets: Dict[int, deque] = {
            interval: deque(maxlen=interval) for interval in intervals
        }
        
    def add_metric(self, metrics: PerformanceMetrics):
        """Добавление метрики во все интервалы"""
        for interval in self.intervals:
            self.buckets[interval].append(metrics)
            
    def get_aggregated(self, interval: int) -> Optional[Dict[str, Any]]:
        """Получение агрегированных метрик для интервала"""
        if interval not in self.buckets or not self.buckets[interval]:
            return None
            
        metrics_list = list(self.buckets[interval])
        
        try:
            return {
                'interval_seconds': interval,
                'count': len(metrics_list),
                'timespan': {
                    'start': metrics_list[0].timestamp,
                    'end': metrics_list[-1].timestamp
                },
                'cpu_usage': {
                    'avg': statistics.mean(m.cpu_usage for m in metrics_list),
                    'max': max(m.cpu_usage for m in metrics_list),
                    'min': min(m.cpu_usage for m in metrics_list)
                },
                'memory_usage': {
                    'avg': statistics.mean(m.memory_usage for m in metrics_list),
                    'max': max(m.memory_usage for m in metrics_list),
                    'min': min(m.memory_usage for m in metrics_list)
                },
                'gpu_usage': {
                    'avg': statistics.mean(m.gpu_usage for m in metrics_list),
                    'max': max(m.gpu_usage for m in metrics_list),
                    'min': min(m.gpu_usage for m in metrics_list)
                } if any(m.gpu_usage > 0 for m in metrics_list) else None,
                'fps': {
                    'avg': statistics.mean(m.current_fps for m in metrics_list if m.current_fps > 0),
                    'max': max((m.current_fps for m in metrics_list if m.current_fps > 0), default=0),
                    'min': min((m.current_fps for m in metrics_list if m.current_fps > 0), default=0)
                },
                'latency_ms': {
                    'avg': statistics.mean(m.avg_latency_ms for m in metrics_list if m.avg_latency_ms > 0),
                    'p95': statistics.mean(m.p95_latency_ms for m in metrics_list if m.p95_latency_ms > 0),
                    'p99': statistics.mean(m.p99_latency_ms for m in metrics_list if m.p99_latency_ms > 0)
                },
                'throughput': {
                    'avg_inference_fps': statistics.mean(m.inference_throughput for m in metrics_list if m.inference_throughput > 0),
                    'total_frames': sum(m.frames_processed for m in metrics_list),
                    'total_mb': sum(m.bytes_processed_mb for m in metrics_list)
                }
            }
        except Exception as e:
            logger.error(f"Ошибка агрегации метрик: {e}")
            return None

class AlertManager:
    """Менеджер алертов для критических состояний"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.active_alerts: Set[str] = set()
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_cooldowns: Dict[str, float] = {}
        self.cooldown_period = 60.0  # 1 минута между алертами одного типа
        
    def check_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Проверка алертов и возврат списка новых"""
        if not self.config.enable_alerts:
            return []
            
        new_alerts = []
        current_time = time.time()
        
        # Проверка каждого порога
        checks = [
            ('cpu_usage', metrics.cpu_usage, 'CPU usage'),
            ('memory_usage', metrics.memory_usage, 'Memory usage'), 
            ('gpu_usage', metrics.gpu_usage, 'GPU usage'),
            ('latency_ms', metrics.avg_latency_ms, 'Average latency'),
            ('error_rate', metrics.error_rate, 'Error rate')
        ]
        
        # FPS специальная проверка
        if metrics.current_fps > 0 and metrics.target_fps > 0:
            fps_ratio = metrics.current_fps / metrics.target_fps
            if fps_ratio < self.config.alert_thresholds.get('fps_drop', 0.5):
                checks.append(('fps_drop', fps_ratio, 'FPS drop'))
                
        for alert_type, value, description in checks:
            threshold = self.config.alert_thresholds.get(alert_type)
            if threshold is None:
                continue
                
            # Проверка превышения порога
            is_critical = (
                (alert_type != 'fps_drop' and value > threshold) or
                (alert_type == 'fps_drop' and value < threshold)
            )
            
            # Проверка cooldown
            last_alert_time = self.alert_cooldowns.get(alert_type, 0)
            if is_critical and (current_time - last_alert_time) > self.cooldown_period:
                alert = {
                    'type': alert_type,
                    'description': description,
                    'value': value,
                    'threshold': threshold,
                    'severity': self._calculate_severity(alert_type, value, threshold),
                    'timestamp': current_time,
                    'message': f"{description}: {value:.2f} (threshold: {threshold:.2f})"
                }
                
                new_alerts.append(alert)
                self.active_alerts.add(alert_type)
                self.alert_cooldowns[alert_type] = current_time
                self.alert_history.append(alert)
                
                # Ограничиваем историю
                if len(self.alert_history) > 100:
                    self.alert_history.pop(0)
                    
            elif not is_critical and alert_type in self.active_alerts:
                # Алерт разрешился
                self.active_alerts.discard(alert_type)
                
        return new_alerts
        
    def _calculate_severity(self, alert_type: str, value: float, threshold: float) -> str:
        """Расчет серьезности алерта"""
        if alert_type == 'fps_drop':
            ratio = value / threshold if threshold > 0 else 0
            if ratio < 0.3:
                return 'CRITICAL'
            elif ratio < 0.6:
                return 'HIGH'
            else:
                return 'MEDIUM'
        else:
            ratio = value / threshold if threshold > 0 else 0
            if ratio > 1.5:
                return 'CRITICAL'
            elif ratio > 1.2:
                return 'HIGH'
            else:
                return 'MEDIUM'

class WebSocketBroadcaster:
    """WebSocket broadcaster для передачи метрик клиентам"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self._running = False
        
    async def start(self):
        """Запуск WebSocket сервера"""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets не установлен, WebSocket broadcasting недоступен")
            return
            
        try:
            self.server = await serve(
                self._handle_client,
                self.config.websocket_host,
                self.config.websocket_port,
                max_size=2**16,
                max_queue=32
            )
            self._running = True
            logger.info(f"WebSocket сервер запущен на {self.config.websocket_host}:{self.config.websocket_port}")
        except Exception as e:
            logger.error(f"Ошибка запуска WebSocket сервера: {e}")
            
    async def stop(self):
        """Остановка сервера"""
        self._running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Закрыть все соединения
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        self.clients.clear()
        
    async def _handle_client(self, websocket, path):
        """Обработка WebSocket клиента"""
        if len(self.clients) >= self.config.max_websocket_clients:
            await websocket.close(4008, "Too many connections")
            return
            
        self.clients.add(websocket)
        logger.info(f"WebSocket клиент подключен: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        except Exception as e:
            logger.debug(f"WebSocket соединение закрыто: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"WebSocket клиент отключен: {websocket.remote_address}")
            
    async def broadcast(self, data: Dict[str, Any]):
        """Трансляция данных всем клиентам"""
        if not self.clients:
            return
            
        message = json.dumps(data, default=str)
        disconnected = []
        
        for client in self.clients.copy():
            try:
                await client.send(message)
            except Exception as e:
                logger.debug(f"Ошибка отправки WebSocket сообщения: {e}")
                disconnected.append(client)
                
        # Удаляем отключенных клиентов
        for client in disconnected:
            self.clients.discard(client)

class PerformanceMonitor:
    """
    Системный монитор производительности с возможностями:
    - Системные метрики (CPU, память, GPU)
    - FPS и end-to-end латентность  
    - WebSocket broadcast метрик
    - История для анализа трендов
    - Алерты при критических состояниях
    """
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        
        # Компоненты
        self.aggregator = MetricsAggregator(config.aggregation_intervals)
        self.alert_manager = AlertManager(config)
        self.websocket_broadcaster = WebSocketBroadcaster(config)
        
        # История метрик
        self.metrics_history: deque[PerformanceMetrics] = deque(
            maxlen=int(config.max_history_minutes * 60 / config.metrics_interval)
        )
        
        # Состояние
        self._running = False
        self._metrics_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        
        # Внешние источники метрик
        self._external_metrics: Dict[str, Any] = {}
        self._metrics_lock = threading.RLock()
        
        # Базовые системные метрики
        self._last_network_stats = psutil.net_io_counters()
        self._process_start_time = time.time()
        
        logger.info("PerformanceMonitor инициализирован")
        
    async def start(self):
        """Запуск мониторинга"""
        if self._running:
            return
            
        self._running = True
        
        # Запуск компонентов
        await self.websocket_broadcaster.start()
        
        # Запуск задач
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        
        logger.info("PerformanceMonitor запущен")
        
    async def stop(self):
        """Остановка мониторинга"""
        if not self._running:
            return
            
        self._running = False
        
        # Остановка задач
        tasks = [self._metrics_task, self._broadcast_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Остановка компонентов
        await self.websocket_broadcaster.stop()
        
        logger.info("PerformanceMonitor остановлен")
        
    def update_external_metrics(self, metrics: Dict[str, Any]):
        """Обновление внешних метрик от компонентов приложения"""
        with self._metrics_lock:
            self._external_metrics.update(metrics)
            
    async def _metrics_loop(self):
        """Цикл сбора метрик"""
        while self._running:
            try:
                # Сбор метрик
                metrics = await self._collect_metrics()
                
                # Добавление в историю и агрегатор
                self.metrics_history.append(metrics)
                self.aggregator.add_metric(metrics)
                
                # Проверка алертов
                alerts = self.alert_manager.check_alerts(metrics)
                if alerts:
                    logger.warning(f"Новые алерты: {[a['message'] for a in alerts]}")
                    
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Ошибка в metrics_loop: {e}")
                await asyncio.sleep(1.0)
                
    async def _broadcast_loop(self):
        """Цикл трансляции метрик"""
        while self._running:
            try:
                # Подготовка данных для трансляции
                broadcast_data = await self._prepare_broadcast_data()
                
                # Трансляция через WebSocket
                await self.websocket_broadcaster.broadcast(broadcast_data)
                
                await asyncio.sleep(self.config.websocket_broadcast_interval)
                
            except Exception as e:
                logger.error(f"Ошибка в broadcast_loop: {e}")
                await asyncio.sleep(1.0)
                
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Сбор всех метрик"""
        metrics = PerformanceMetrics()
        
        # Системные ресурсы
        try:
            # CPU
            metrics.cpu_usage = psutil.cpu_percent(interval=None)
            
            # Память
            memory = psutil.virtual_memory()
            metrics.memory_usage = memory.percent
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_total_gb = memory.total / (1024**3)
            
            # Сеть
            network_stats = psutil.net_io_counters()
            if self._last_network_stats:
                sent_diff = network_stats.bytes_sent - self._last_network_stats.bytes_sent
                recv_diff = network_stats.bytes_recv - self._last_network_stats.bytes_recv
                metrics.network_sent_mb = sent_diff / (1024**2) / self.config.metrics_interval
                metrics.network_received_mb = recv_diff / (1024**2) / self.config.metrics_interval
            self._last_network_stats = network_stats
            
            # GPU метрики
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        metrics.gpu_usage = gpu.load * 100
                        metrics.gpu_memory_usage = gpu.memoryUtil * 100
                        metrics.gpu_memory_used_mb = gpu.memoryUsed
                        metrics.gpu_memory_total_mb = gpu.memoryTotal
                        metrics.gpu_temperature = gpu.temperature
                except Exception as e:
                    logger.debug(f"Ошибка получения GPU метрик: {e}")
                    
        except Exception as e:
            logger.error(f"Ошибка сбора системных метрик: {e}")
            
        # Внешние метрики
        with self._metrics_lock:
            external = self._external_metrics.copy()
            
        # Производительность приложения
        metrics.current_fps = external.get('fps', 0.0)
        metrics.target_fps = external.get('target_fps', 30.0)
        metrics.avg_latency_ms = external.get('avg_latency_ms', 0.0)
        metrics.p95_latency_ms = external.get('p95_latency_ms', 0.0)
        metrics.p99_latency_ms = external.get('p99_latency_ms', 0.0)
        
        # Потоки и соединения
        metrics.active_streams = external.get('active_streams', 0)
        metrics.total_connections = external.get('total_connections', 0)
        
        # Ошибки и качество
        metrics.error_rate = external.get('error_rate', 0.0)
        metrics.frame_drop_rate = external.get('frame_drop_rate', 0.0)
        metrics.quality_level = external.get('quality_level', 'MEDIUM')
        
        # Throughput
        metrics.frames_processed = external.get('frames_processed', 0)
        metrics.bytes_processed_mb = external.get('bytes_processed_mb', 0.0)
        metrics.inference_throughput = external.get('inference_throughput', 0.0)
        
        return metrics
        
    async def _prepare_broadcast_data(self) -> Dict[str, Any]:
        """Подготовка данных для WebSocket трансляции"""
        data = {
            'type': 'performance_update',
            'timestamp': time.time()
        }
        
        # Последние метрики
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            data['current'] = asdict(latest_metrics)
            
        # Агрегированные данные
        data['aggregated'] = {}
        for interval in self.config.aggregation_intervals:
            aggregated = self.aggregator.get_aggregated(interval)
            if aggregated:
                data['aggregated'][f'{interval}s'] = aggregated
                
        # Активные алерты
        data['alerts'] = {
            'active': list(self.alert_manager.active_alerts),
            'recent': self.alert_manager.alert_history[-5:] if self.alert_manager.alert_history else []
        }
        
        # Статистика WebSocket
        data['websocket_clients'] = len(self.websocket_broadcaster.clients)
        
        return data
        
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Получение последних метрик"""
        return self.metrics_history[-1] if self.metrics_history else None
        
    def get_metrics_history(self, minutes: int = 10) -> List[PerformanceMetrics]:
        """Получение истории метрик за указанный период"""
        if not self.metrics_history:
            return []
            
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
    def get_aggregated_metrics(self, interval: int) -> Optional[Dict[str, Any]]:
        """Получение агрегированных метрик"""
        return self.aggregator.get_aggregated(interval)
        
    def get_alerts(self) -> Dict[str, Any]:
        """Получение информации об алертах"""
        return {
            'active': list(self.alert_manager.active_alerts),
            'history': self.alert_manager.alert_history,
            'thresholds': self.config.alert_thresholds
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Получение общей статистики монитора"""
        uptime = time.time() - self._process_start_time
        
        return {
            'running': self._running,
            'uptime_seconds': uptime,
            'metrics_collected': len(self.metrics_history),
            'websocket_clients': len(self.websocket_broadcaster.clients),
            'active_alerts': len(self.alert_manager.active_alerts),
            'config': {
                'metrics_interval': self.config.metrics_interval,
                'max_history_minutes': self.config.max_history_minutes,
                'websocket_port': self.config.websocket_port,
                'alerts_enabled': self.config.enable_alerts
            }
        }