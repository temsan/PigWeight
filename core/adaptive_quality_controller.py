"""
AdaptiveQualityController - Контроллер адаптивного качества
5 уровней качества с автоматической адаптацией каждые 2 секунды
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque
import statistics

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class QualityLevel(IntEnum):
    """Уровни качества от самого высокого к минимальному"""
    ULTRA = 5      # Максимальное качество
    HIGH = 4       # Высокое качество
    MEDIUM = 3     # Среднее качество
    LOW = 2        # Низкое качество
    MINIMAL = 1    # Минимальное качество

@dataclass
class QualitySettings:
    """Настройки качества для определенного уровня"""
    level: QualityLevel
    # Видео параметры
    resolution_scale: float = 1.0  # Масштаб разрешения (1.0 = оригинал)
    fps_limit: float = 30.0
    jpeg_quality: int = 85
    h264_bitrate: int = 2000000  # bps
    h264_preset: str = "medium"  # ultrafast, superfast, veryfast, faster, fast, medium, slow
    
    # Обработка
    batch_size: int = 8
    inference_threads: int = 1
    use_half_precision: bool = False
    skip_frame_interval: int = 1  # Обработка каждого N-го кадра
    
    # Качество детекции
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100

@dataclass
class SystemMetrics:
    """Системные метрики для принятия решений"""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    current_fps: float = 0.0
    target_fps: float = 30.0
    processing_latency_ms: float = 0.0
    network_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class AdaptationConfig:
    """Конфигурация адаптации"""
    # Пороговые значения для деградации качества
    cpu_threshold_high: float = 80.0      # % CPU для снижения качества
    cpu_threshold_low: float = 60.0       # % CPU для повышения качества
    memory_threshold_high: float = 85.0   # % памяти для снижения качества
    memory_threshold_low: float = 70.0    # % памяти для повышения качества
    gpu_threshold_high: float = 90.0      # % GPU для снижения качества
    gpu_threshold_low: float = 70.0       # % GPU для повышения качества
    
    # FPS и латентность
    fps_drop_threshold: float = 0.8       # Если FPS < target * threshold
    latency_threshold_ms: float = 100.0   # Максимальная допустимая латентность
    
    # Управление адаптацией
    adaptation_interval: float = 2.0      # Интервал проверки в секундах
    cooldown_period: float = 10.0         # Период между изменениями качества
    stability_window: int = 3             # Количество измерений для стабильности
    
    # Веса факторов (сумма должна быть 1.0)
    cpu_weight: float = 0.3
    memory_weight: float = 0.2
    gpu_weight: float = 0.3
    latency_weight: float = 0.2

class QualityPresets:
    """Предустановленные настройки качества"""
    
    PRESETS = {
        QualityLevel.ULTRA: QualitySettings(
            level=QualityLevel.ULTRA,
            resolution_scale=1.0,
            fps_limit=60.0,
            jpeg_quality=95,
            h264_bitrate=4000000,
            h264_preset="slow",
            batch_size=16,
            inference_threads=2,
            use_half_precision=False,
            skip_frame_interval=1,
            confidence_threshold=0.3,
            nms_threshold=0.3,
            max_detections=200
        ),
        
        QualityLevel.HIGH: QualitySettings(
            level=QualityLevel.HIGH,
            resolution_scale=1.0,
            fps_limit=30.0,
            jpeg_quality=85,
            h264_bitrate=2500000,
            h264_preset="medium",
            batch_size=12,
            inference_threads=1,
            use_half_precision=False,
            skip_frame_interval=1,
            confidence_threshold=0.4,
            nms_threshold=0.35,
            max_detections=150
        ),
        
        QualityLevel.MEDIUM: QualitySettings(
            level=QualityLevel.MEDIUM,
            resolution_scale=0.8,
            fps_limit=25.0,
            jpeg_quality=75,
            h264_bitrate=1500000,
            h264_preset="fast",
            batch_size=8,
            inference_threads=1,
            use_half_precision=True,
            skip_frame_interval=1,
            confidence_threshold=0.5,
            nms_threshold=0.4,
            max_detections=100
        ),
        
        QualityLevel.LOW: QualitySettings(
            level=QualityLevel.LOW,
            resolution_scale=0.6,
            fps_limit=20.0,
            jpeg_quality=65,
            h264_bitrate=1000000,
            h264_preset="faster",
            batch_size=4,
            inference_threads=1,
            use_half_precision=True,
            skip_frame_interval=2,  # Каждый второй кадр
            confidence_threshold=0.6,
            nms_threshold=0.45,
            max_detections=50
        ),
        
        QualityLevel.MINIMAL: QualitySettings(
            level=QualityLevel.MINIMAL,
            resolution_scale=0.4,
            fps_limit=15.0,
            jpeg_quality=50,
            h264_bitrate=500000,
            h264_preset="ultrafast",
            batch_size=2,
            inference_threads=1,
            use_half_precision=True,
            skip_frame_interval=3,  # Каждый третий кадр
            confidence_threshold=0.7,
            nms_threshold=0.5,
            max_detections=25
        )
    }

class SystemMonitor:
    """Мониторинг системных ресурсов"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self._last_network_check = 0.0
        self._network_stats = {}
        
    def get_system_metrics(self) -> SystemMetrics:
        """Получение текущих системных метрик"""
        metrics = SystemMetrics()
        
        try:
            # CPU метрики
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
            # Память
            memory = psutil.virtual_memory()
            metrics.memory_usage_percent = memory.percent
            
            # GPU метрики
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Используем первый GPU
                        metrics.gpu_usage_percent = gpu.load * 100
                        metrics.gpu_memory_usage_percent = gpu.memoryUtil * 100
                except Exception as e:
                    logger.debug(f"Ошибка получения GPU метрик: {e}")
                    
        except Exception as e:
            logger.error(f"Ошибка получения системных метрик: {e}")
            
        return metrics

class AdaptiveQualityController:
    """
    Контроллер адаптивного качества с 5 уровнями:
    - Мониторинг FPS, CPU, латентности каждые 2 секунды
    - Автоматическая адаптация с cooldown 10 секунд
    - Весовые коэффициенты для разных факторов
    """
    
    def __init__(self, 
                 config: AdaptationConfig,
                 quality_change_callback: Optional[Callable[[QualitySettings], None]] = None):
        self.config = config
        self.quality_change_callback = quality_change_callback
        
        # Текущее состояние
        self._current_quality = QualityLevel.MEDIUM
        self._current_settings = QualityPresets.PRESETS[self._current_quality]
        
        # Мониторинг
        self._monitor = SystemMonitor()
        self._metrics_history: deque[SystemMetrics] = deque(maxlen=self.config.stability_window * 2)
        
        # Состояние контроллера
        self._running = False
        self._adaptation_task: Optional[asyncio.Task] = None
        self._last_adaptation = 0.0
        
        # Статистика
        self.stats = {
            'total_adaptations': 0,
            'quality_changes': 0,
            'upscales': 0,
            'downscales': 0,
            'stability_score': 0.0,
            'avg_response_time': 0.0
        }
        
        # Внешние метрики (устанавливаются извне)
        self._external_fps = 0.0
        self._external_latency = 0.0
        
        logger.info(f"AdaptiveQualityController создан с уровнем {self._current_quality.name}")
        
    async def start(self):
        """Запуск контроллера"""
        if self._running:
            return
            
        self._running = True
        self._adaptation_task = asyncio.create_task(self._adaptation_loop())
        
        logger.info("AdaptiveQualityController запущен")
        
    async def stop(self):
        """Остановка контроллера"""
        if not self._running:
            return
            
        self._running = False
        
        if self._adaptation_task:
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass
                
        logger.info("AdaptiveQualityController остановлен")
        
    def update_performance_metrics(self, fps: float, latency_ms: float):
        """Обновление метрик производительности от внешних компонентов"""
        self._external_fps = fps
        self._external_latency = latency_ms
        
    def get_current_settings(self) -> QualitySettings:
        """Получение текущих настроек качества"""
        return self._current_settings
        
    def set_quality_level(self, level: QualityLevel, force: bool = False):
        """Ручная установка уровня качества"""
        if not force and not self._can_adapt():
            logger.warning(f"Адаптация заблокирована cooldown периодом")
            return
            
        if level != self._current_quality:
            old_level = self._current_quality
            self._current_quality = level
            self._current_settings = QualityPresets.PRESETS[level]
            self._last_adaptation = time.time()
            
            # Вызов callback
            if self.quality_change_callback:
                self.quality_change_callback(self._current_settings)
                
            # Обновление статистики
            self.stats['quality_changes'] += 1
            if level > old_level:
                self.stats['upscales'] += 1
            else:
                self.stats['downscales'] += 1
                
            logger.info(f"Качество изменено: {old_level.name} -> {level.name}")
            
    async def _adaptation_loop(self):
        """Основной цикл адаптации"""
        while self._running:
            try:
                await asyncio.sleep(self.config.adaptation_interval)
                
                # Сбор метрик
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Адаптация качества
                if len(self._metrics_history) >= self.config.stability_window:
                    await self._adapt_quality()
                    
                self.stats['total_adaptations'] += 1
                
            except Exception as e:
                logger.error(f"Ошибка в adaptation_loop: {e}")
                await asyncio.sleep(1.0)
                
    def _collect_metrics(self) -> SystemMetrics:
        """Сбор системных метрик"""
        metrics = self._monitor.get_system_metrics()
        
        # Добавление внешних метрик
        metrics.current_fps = self._external_fps
        metrics.target_fps = self._current_settings.fps_limit
        metrics.processing_latency_ms = self._external_latency
        
        return metrics
        
    async def _adapt_quality(self):
        """Адаптация уровня качества"""
        if not self._can_adapt():
            return
            
        # Анализ стабильных метрик
        stable_metrics = self._get_stable_metrics()
        
        # Расчет стресс-фактора системы
        stress_score = self._calculate_stress_score(stable_metrics)
        
        # Определение целевого уровня качества
        target_level = self._determine_target_quality(stress_score)
        
        # Применение изменений
        if target_level != self._current_quality:
            self.set_quality_level(target_level)
            
        # Обновление стабильности
        self.stats['stability_score'] = self._calculate_stability_score()
        
    def _can_adapt(self) -> bool:
        """Проверка возможности адаптации (cooldown)"""
        return (time.time() - self._last_adaptation) >= self.config.cooldown_period
        
    def _get_stable_metrics(self) -> SystemMetrics:
        """Получение стабилизированных метрик"""
        if len(self._metrics_history) < self.config.stability_window:
            return self._metrics_history[-1] if self._metrics_history else SystemMetrics()
            
        # Медианные значения для стабильности
        recent_metrics = list(self._metrics_history)[-self.config.stability_window:]
        
        return SystemMetrics(
            cpu_usage_percent=statistics.median(m.cpu_usage_percent for m in recent_metrics),
            memory_usage_percent=statistics.median(m.memory_usage_percent for m in recent_metrics),
            gpu_usage_percent=statistics.median(m.gpu_usage_percent for m in recent_metrics),
            gpu_memory_usage_percent=statistics.median(m.gpu_memory_usage_percent for m in recent_metrics),
            current_fps=statistics.mean(m.current_fps for m in recent_metrics),
            processing_latency_ms=statistics.mean(m.processing_latency_ms for m in recent_metrics),
            timestamp=time.time()
        )
        
    def _calculate_stress_score(self, metrics: SystemMetrics) -> float:
        """Расчет стресс-фактора системы (0.0 = нет нагрузки, 1.0 = максимальная)"""
        scores = []
        
        # CPU стресс
        cpu_stress = max(0, (metrics.cpu_usage_percent - self.config.cpu_threshold_low) / 
                        (self.config.cpu_threshold_high - self.config.cpu_threshold_low))
        scores.append(min(1.0, cpu_stress) * self.config.cpu_weight)
        
        # Memory стресс
        memory_stress = max(0, (metrics.memory_usage_percent - self.config.memory_threshold_low) / 
                           (self.config.memory_threshold_high - self.config.memory_threshold_low))
        scores.append(min(1.0, memory_stress) * self.config.memory_weight)
        
        # GPU стресс
        if metrics.gpu_usage_percent > 0:
            gpu_stress = max(0, (metrics.gpu_usage_percent - self.config.gpu_threshold_low) / 
                            (self.config.gpu_threshold_high - self.config.gpu_threshold_low))
            scores.append(min(1.0, gpu_stress) * self.config.gpu_weight)
        
        # Latency стресс
        latency_stress = max(0, metrics.processing_latency_ms / self.config.latency_threshold_ms)
        scores.append(min(1.0, latency_stress) * self.config.latency_weight)
        
        # FPS стресс (обратная зависимость)
        if metrics.target_fps > 0:
            fps_ratio = metrics.current_fps / metrics.target_fps
            fps_stress = max(0, 1.0 - fps_ratio / self.config.fps_drop_threshold)
            # Добавляем FPS стресс к latency весу
            scores.append(fps_stress * self.config.latency_weight * 0.5)
        
        return sum(scores)
        
    def _determine_target_quality(self, stress_score: float) -> QualityLevel:
        """Определение целевого уровня качества по стресс-фактору"""
        # Пороги для переключения качества
        thresholds = {
            0.2: QualityLevel.ULTRA,    # Низкая нагрузка
            0.4: QualityLevel.HIGH,     # Умеренная нагрузка  
            0.6: QualityLevel.MEDIUM,   # Средняя нагрузка
            0.8: QualityLevel.LOW,      # Высокая нагрузка
            1.0: QualityLevel.MINIMAL   # Критическая нагрузка
        }
        
        # Гистерезис для предотвращения частых переключений
        hysteresis = 0.05
        current_threshold = None
        
        # Найти текущий порог
        for threshold, level in thresholds.items():
            if level == self._current_quality:
                current_threshold = threshold
                break
                
        # Применение гистерезиса
        adjusted_score = stress_score
        if current_threshold:
            if stress_score > current_threshold:
                adjusted_score = stress_score - hysteresis  # Упрощаем повышение качества
            else:
                adjusted_score = stress_score + hysteresis  # Упрощаем снижение качества
                
        # Определение уровня
        for threshold in sorted(thresholds.keys()):
            if adjusted_score <= threshold:
                return thresholds[threshold]
                
        return QualityLevel.MINIMAL
        
    def _calculate_stability_score(self) -> float:
        """Расчет оценки стабильности системы"""
        if len(self._metrics_history) < 2:
            return 1.0
            
        # Анализ вариабельности метрик
        recent_metrics = list(self._metrics_history)[-self.config.stability_window:]
        
        variabilities = []
        
        # CPU вариабельность
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        if len(cpu_values) > 1:
            cpu_var = statistics.stdev(cpu_values) / statistics.mean(cpu_values) if statistics.mean(cpu_values) > 0 else 0
            variabilities.append(cpu_var)
            
        # FPS вариабельность
        fps_values = [m.current_fps for m in recent_metrics if m.current_fps > 0]
        if len(fps_values) > 1:
            fps_var = statistics.stdev(fps_values) / statistics.mean(fps_values) if statistics.mean(fps_values) > 0 else 0
            variabilities.append(fps_var)
            
        # Latency вариабельность
        latency_values = [m.processing_latency_ms for m in recent_metrics if m.processing_latency_ms > 0]
        if len(latency_values) > 1:
            latency_var = statistics.stdev(latency_values) / statistics.mean(latency_values) if statistics.mean(latency_values) > 0 else 0
            variabilities.append(latency_var)
            
        if variabilities:
            avg_variability = statistics.mean(variabilities)
            stability = max(0.0, 1.0 - avg_variability)
            return min(1.0, stability)
            
        return 1.0
        
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики контроллера"""
        current_metrics = self._metrics_history[-1] if self._metrics_history else SystemMetrics()
        
        return {
            'current_quality': self._current_quality.name,
            'current_settings': {
                'resolution_scale': self._current_settings.resolution_scale,
                'fps_limit': self._current_settings.fps_limit,
                'batch_size': self._current_settings.batch_size,
                'jpeg_quality': self._current_settings.jpeg_quality
            },
            'metrics': {
                'cpu_usage': current_metrics.cpu_usage_percent,
                'memory_usage': current_metrics.memory_usage_percent,
                'gpu_usage': current_metrics.gpu_usage_percent,
                'current_fps': current_metrics.current_fps,
                'latency_ms': current_metrics.processing_latency_ms
            },
            'stats': self.stats,
            'can_adapt': self._can_adapt(),
            'time_to_next_adaptation': max(0, self.config.cooldown_period - (time.time() - self._last_adaptation))
        }