"""
Единый видео процессор для системы PigWeight
Объединяет функциональность всех существующих процессоров в одном классе
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class ProcessingOptions:
    """Опции обработки видео"""
    confidence_threshold: float = 0.3
    img_size: int = 640
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 1
    max_cache_size: int = 1000
    enable_tracking: bool = True
    enable_segmentation: bool = False


@dataclass
class FrameResult:
    """Результат обработки кадра"""
    frame_number: int
    timestamp: float
    detections: List[Dict[str, Any]]
    pig_count: int
    processing_time_ms: float
    confidence_scores: List[float]
    masks: Optional[np.ndarray] = None


class UnifiedVideoProcessor:
    """
    Единый процессор для всех типов видео обработки
    Заменяет lightweight_processor, ultra_fast_endpoints, gpu_video_processor
    """
    
    def __init__(self, model_path: str, options: Optional[ProcessingOptions] = None):
        self.model_path = Path(model_path)
        self.options = options or ProcessingOptions()
        
        # Определяем лучшее устройство
        self.device = self._detect_best_device()
        logger.info(f"Используется устройство: {self.device}")
        
        # Загружаем модель
        self.model = None
        self._load_model()
        
        # Кеширование
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Статистика
        self.stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0,
            'pig_detections': 0,
            'cache_hit_rate': 0.0
        }
        
        # Thread pool для CPU-intensive операций
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def _detect_best_device(self) -> str:
        """Автоматическое определение лучшего устройства"""
        if self.options.device != "auto":
            return self.options.device
            
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Обнаружена GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return "cuda"
        else:
            logger.info("GPU не найдена, используется CPU")
            return "cpu"
    
    def _load_model(self):
        """Загрузка модели YOLO"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            
            logger.info(f"Загрузка модели: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            
            # Прогрев модели
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, device=self.device, verbose=False)
            
            logger.info("Модель успешно загружена и прогрета")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    async def process_frame(self, frame: np.ndarray, frame_number: int = 0, 
                          timestamp: float = 0.0) -> FrameResult:
        """Обработка одного кадра"""
        start_time = time.time()
        
        try:
            # Проверяем кеш
            cache_key = f"{frame_number}_{hash(frame.tobytes())}"
            if cache_key in self._cache:
                self._cache_hits += 1
                result = self._cache[cache_key]
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result
            
            self._cache_misses += 1
            
            # Выполняем инференс в отдельном потоке
            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                self._executor, 
                self._run_inference, 
                frame
            )
            
            # Обрабатываем результаты
            pig_count = 0
            confidence_scores = []
            processed_detections = []
            
            for det in detections:
                if det['class'] == 'pig' or det['class_id'] == 0:  # Предполагаем, что свиньи имеют class_id = 0
                    pig_count += 1
                
                confidence_scores.append(det['confidence'])
                processed_detections.append({
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class': det.get('class', 'pig'),
                    'class_id': det.get('class_id', 0)
                })
            
            processing_time = (time.time() - start_time) * 1000
            
            result = FrameResult(
                frame_number=frame_number,
                timestamp=timestamp,
                detections=processed_detections,
                pig_count=pig_count,
                processing_time_ms=processing_time,
                confidence_scores=confidence_scores
            )
            
            # Кешируем результат
            if len(self._cache) < self.options.max_cache_size:
                self._cache[cache_key] = result
            
            # Обновляем статистику
            self._update_stats(processing_time, pig_count)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки кадра: {e}")
            return FrameResult(
                frame_number=frame_number,
                timestamp=timestamp,
                detections=[],
                pig_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                confidence_scores=[]
            )
    
    def _run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Выполнение инференса модели"""
        try:
            results = self.model(
                frame,
                device=self.device,
                conf=self.options.confidence_threshold,
                imgsz=self.options.img_size,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': cls,
                            'class': 'pig' if cls == 0 else f'class_{cls}'
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            return []
    
    async def process_video(self, video_path: Union[str, Path], 
                          progress_callback: Optional[Callable] = None,
                          frame_skip: int = 1) -> Dict[str, Any]:
        """Обработка всего видео"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
        
        logger.info(f"Начинаем обработку видео: {video_path}")
        start_time = time.time()
        
        # Открываем видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Не удается открыть видео: {video_path}")
        
        # Получаем информацию о видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Видео: {total_frames} кадров, {fps:.2f} FPS, {duration:.2f}с")
        
        results = []
        frame_number = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Пропускаем кадры если нужно
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                timestamp = frame_number / fps if fps > 0 else 0
                
                # Обрабатываем кадр
                result = await self.process_frame(frame, frame_number, timestamp)
                results.append(result)
                
                processed_frames += 1
                
                # Вызываем callback прогресса
                if progress_callback:
                    progress = (frame_number + 1) / total_frames
                    await progress_callback(progress, result)
                
                frame_number += 1
                
                # Логируем прогресс каждые 100 кадров
                if processed_frames % 100 == 0:
                    progress = (frame_number / total_frames) * 100
                    logger.info(f"Обработано {processed_frames} кадров ({progress:.1f}%)")
        
        finally:
            cap.release()
        
        processing_time = time.time() - start_time
        
        # Собираем статистику
        total_pigs = sum(r.pig_count for r in results)
        avg_pigs_per_frame = total_pigs / len(results) if results else 0
        max_pigs = max((r.pig_count for r in results), default=0)
        
        summary = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'duration': duration,
            'processing_time': processing_time,
            'processing_fps': processed_frames / processing_time if processing_time > 0 else 0,
            'total_pigs_detected': total_pigs,
            'average_pigs_per_frame': avg_pigs_per_frame,
            'max_pigs_in_frame': max_pigs,
            'results': results
        }
        
        logger.info(f"Обработка завершена за {processing_time:.2f}с")
        logger.info(f"Обнаружено свиней: {total_pigs} (среднее: {avg_pigs_per_frame:.1f}/кадр)")
        
        return summary
    
    def _update_stats(self, processing_time: float, pig_count: int):
        """Обновление статистики"""
        self.stats['frames_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['pig_detections'] += pig_count
        
        if self.stats['frames_processed'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['frames_processed']
            self.stats['average_fps'] = 1000 / avg_time if avg_time > 0 else 0
        
        total_cache_requests = self._cache_hits + self._cache_misses
        if total_cache_requests > 0:
            self.stats['cache_hit_rate'] = self._cache_hits / total_cache_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        return {
            **self.stats,
            'device': self.device,
            'model_path': str(self.model_path),
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        }
    
    def clear_cache(self):
        """Очистка кеша"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Кеш очищен")
    
    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


# Глобальный экземпляр процессора
_processor_instance: Optional[UnifiedVideoProcessor] = None


def get_processor(model_path: str = None, options: Optional[ProcessingOptions] = None) -> UnifiedVideoProcessor:
    """Получение глобального экземпляра процессора (Singleton)"""
    global _processor_instance
    
    if _processor_instance is None:
        if model_path is None:
            raise ValueError("model_path должен быть указан при первом вызове")
        _processor_instance = UnifiedVideoProcessor(model_path, options)
    
    return _processor_instance


def reset_processor():
    """Сброс глобального экземпляра процессора"""
    global _processor_instance
    if _processor_instance:
        del _processor_instance
        _processor_instance = None
