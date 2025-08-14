import cv2
import numpy as np
import asyncio
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FrameResult:
    """Результат обработки кадра"""
    frame: np.ndarray
    count: int = 0
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

class LightweightVideoProcessor:
    """Легковесный процессор видео без лишних конверсий"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.fps = 25.0
        self.total_frames = 0
        self.duration = 0.0
        self.frame_cache = {}
        self.max_cache_size = 50  # ограничиваем размер кеша
        self.model = None
        self.inference_enabled = False
        
    def open(self) -> bool:
        """Открывает видео с оптимизированными настройками"""
        try:
            # Простой OpenCV без лишних бэкендов
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                return False
                
            # Минимальный буфер для быстрого seek
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Получаем метаданные
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0
            
            logger.info(f"Opened video: fps={self.fps:.2f}, frames={self.total_frames}, duration={self.duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video {self.video_path}: {e}")
            return False
    
    def close(self):
        """Закрывает видео и очищает кеш"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame_cache.clear()
        
    def get_frame_fast(self, timestamp: float) -> Optional[np.ndarray]:
        """Сверхбыстрое получение кадра без инференса для максимальной отзывчивости"""
        if not self.cap or not self.cap.isOpened():
            return None
            
        # Проверяем кеш с более высокой точностью
        cache_key = round(timestamp, 1)  # округляем до 0.1s
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key].copy()
            
        try:
            # Используем самый быстрый метод seek
            self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                # Агрессивное кеширование для отзывчивости
                if len(self.frame_cache) < self.max_cache_size:
                    self.frame_cache[cache_key] = frame.copy()
                return frame
                
        except Exception as e:
            logger.error(f"Error getting frame at {timestamp}s: {e}")
            
        return None
        
    def get_frame_with_inference(self, timestamp: float) -> Optional[FrameResult]:
        """Получение кадра с опциональным инференсом"""
        start_time = time.time()
        frame = self.get_frame_fast(timestamp)
        
        if frame is None:
            return None
            
        result = FrameResult(frame=frame)
        
        # Опциональный инференс только при необходимости
        if self.inference_enabled and self.model is not None:
            inf_start = time.time()
            count = self._run_lightweight_inference(frame)
            result.count = count
            result.inference_time_ms = (time.time() - inf_start) * 1000
            
        result.total_time_ms = (time.time() - start_time) * 1000
        return result
        
    def enable_inference(self, model_path: str = None) -> bool:
        """Включает инференс с легковесной моделью"""
        if model_path is None:
            self.inference_enabled = False
            return True
            
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.inference_enabled = True
            logger.info(f"Inference enabled with model: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            self.inference_enabled = False
            return False
            
    def _run_lightweight_inference(self, frame: np.ndarray) -> int:
        """Упрощенный инференс только для подсчета"""
        try:
            # Минимальные параметры для скорости
            results = self.model.predict(
                frame,
                imgsz=320,  # Уменьшенный размер для скорости
                conf=0.5,
                verbose=False,
                device='cpu'  # Используем CPU для стабильности
            )
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    return len(boxes)
                    
        except Exception as e:
            logger.error(f"Inference error: {e}")
            
        return 0
        
class StreamingVideoProcessor:
    """Упрощенный процессор для стриминга"""
    
    def __init__(self, video_path: str):
        self.processor = LightweightVideoProcessor(video_path)
        self.is_playing = False
        self.current_time = 0.0
        self.play_speed = 1.0
        
    async def start_streaming(self, enable_inference: bool = False, model_path: str = None):
        """Запускает стриминг с минимальной задержкой"""
        if not self.processor.open():
            return False
            
        if enable_inference:
            self.processor.enable_inference(model_path)
            
        self.is_playing = True
        return True
        
    async def get_frame_stream(self) -> Optional[FrameResult]:
        """Получает следующий кадр для стриминга"""
        if not self.is_playing:
            return None
            
        result = self.processor.get_frame_with_inference(self.current_time)
        if result:
            # Продвигаем время
            frame_interval = 1.0 / self.processor.fps
            self.current_time += frame_interval * self.play_speed
            
            # Проверяем конец видео
            if self.current_time >= self.processor.duration:
                self.current_time = 0.0  # Зацикливаем
                
        return result
        
    def seek(self, timestamp: float):
        """Простой seek без лишних проверок"""
        self.current_time = max(0, min(timestamp, self.processor.duration))
        
    def stop(self):
        """Останавливает стриминг"""
        self.is_playing = False
        self.processor.close()

def encode_frame_fast(frame: np.ndarray, quality: int = 80) -> Optional[bytes]:
    """Оптимизированное быстрое кодирование кадра в JPEG"""
    try:
        # Оптимизированные параметры для максимальной скорости и качества
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Включаем оптимизацию
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Прогрессивный JPEG для веба
        ]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        if success:
            return buffer.tobytes()
    except Exception as e:
        logger.error(f"Encoding error: {e}")
    return None

class SimpleFrameCache:
    """Простой кеш кадров"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        
    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            # Обновляем порядок доступа
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key].copy()
        return None
        
    def put(self, key: str, frame: np.ndarray):
        if len(self.cache) >= self.max_size:
            # Удаляем самый старый
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
            
        self.cache[key] = frame.copy()
        self.access_order.append(key)
        
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
