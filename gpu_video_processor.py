"""
GPU-Accelerated Video Processor с индексированием и batch inference
Максимальная производительность для real-time обработки видео
"""

import cv2
import numpy as np
import torch
import threading
import time
import pickle
import os
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import logging
import psutil
from dataclasses import dataclass
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrameIndex:
    """Индекс кадра для быстрого доступа"""
    frame_number: int
    timestamp: float
    file_position: int
    keyframe: bool = False
    features_hash: str = None

@dataclass
class BatchInferenceRequest:
    """Запрос на batch инференс"""
    frames: List[np.ndarray]
    timestamps: List[float]
    callback: callable
    priority: int = 0

class GPUVideoProcessor:
    """Высокопроизводительный видеопроцессор с GPU ускорением"""
    
    def __init__(self, 
                 use_cuda: bool = True,
                 max_cache_size: int = 1000,
                 batch_size: int = 8,
                 index_keyframes_only: bool = False):
        
        # GPU настройки
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Batch processing
        self.batch_size = batch_size
        self.batch_queue = deque()
        self.batch_executor = ThreadPoolExecutor(max_workers=2)
        
        # Индексирование
        self.frame_indices: Dict[str, List[FrameIndex]] = {}
        self.index_keyframes_only = index_keyframes_only
        
        # Кеширование
        self.max_cache_size = max_cache_size
        self.frame_cache: Dict[str, Dict[float, np.ndarray]] = defaultdict(dict)
        self.inference_cache: Dict[str, Any] = {}
        
        # Видео объекты и метаданные
        self.videos: Dict[str, cv2.VideoCapture] = {}
        self.video_metadata: Dict[str, Dict] = {}
        
        # Производительность
        self.stats = {
            'frames_processed': 0,
            'cache_hits': 0,
            'gpu_inference_time': [],
            'batch_sizes': [],
            'index_access_time': []
        }
        
        # GPU модель (заглушка - замените на вашу модель)
        self._initialize_gpu_model()
        
        logger.info(f"GPU Video Processor initialized - Device: {self.device}")
        if self.use_cuda:
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def _initialize_gpu_model(self):
        """Инициализация GPU модели для инференса"""
        try:
            if self.use_cuda:
                # Пример: YOLOv5 или ваша модель
                # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                # self.model = self.model.to(self.device)
                # self.model.eval()
                
                # Заглушка для демонстрации
                self.model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                    torch.nn.Linear(64, 10)
                ).to(self.device)
                
                # Warm-up GPU
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                    
                logger.info("GPU model warmed up and ready")
            else:
                self.model = None
                logger.info("Using CPU inference")
                
        except Exception as e:
            logger.error(f"Failed to initialize GPU model: {e}")
            self.model = None
    
    def load_video(self, video_path: str, video_id: str) -> Dict:
        """Загрузка видео с построением индекса"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Метаданные
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.videos[video_id] = cap
            self.video_metadata[video_id] = {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height,
                'path': video_path
            }
            
            # Построение индекса
            start_time = time.time()
            self._build_frame_index(video_id)
            index_time = time.time() - start_time
            
            logger.info(f"Video {video_id} loaded and indexed in {index_time:.2f}s")
            
            return {
                'id': video_id,
                'fps': fps,
                'duration': duration,
                'total_frames': frame_count,
                'resolution': f"{width}x{height}",
                'index_size': len(self.frame_indices.get(video_id, [])),
                'index_time': index_time
            }
            
        except Exception as e:
            logger.error(f"Error loading video {video_id}: {e}")
            raise
    
    def _build_frame_index(self, video_id: str):
        """Построение индекса кадров для быстрого доступа"""
        cap = self.videos[video_id]
        indices = []
        
        frame_num = 0
        fps = self.video_metadata[video_id]['fps']
        
        # Сохраняем текущую позицию
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            file_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            timestamp = frame_num / fps
            
            # Определяем ключевые кадры (каждые N кадров или по алгоритму)
            is_keyframe = frame_num % 30 == 0  # Каждые 30 кадров
            
            # Добавляем в индекс (все кадры или только ключевые)
            if not self.index_keyframes_only or is_keyframe:
                index_entry = FrameIndex(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    file_position=int(file_pos),
                    keyframe=is_keyframe
                )
                indices.append(index_entry)
            
            frame_num += 1
            
            # Прогресс для больших видео
            if frame_num % 1000 == 0:
                logger.info(f"Indexed {frame_num} frames for {video_id}")
        
        # Восстанавливаем позицию
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        self.frame_indices[video_id] = indices
        
        # Сохраняем индекс на диск для повторного использования
        self._save_frame_index(video_id)
    
    def _save_frame_index(self, video_id: str):
        """Сохранение индекса на диск"""
        try:
            index_dir = Path("frame_indices")
            index_dir.mkdir(exist_ok=True)
            
            index_file = index_dir / f"{video_id}_index.pkl"
            with open(index_file, 'wb') as f:
                pickle.dump(self.frame_indices[video_id], f)
                
            logger.info(f"Frame index saved for {video_id}")
        except Exception as e:
            logger.warning(f"Could not save index for {video_id}: {e}")
    
    def _load_frame_index(self, video_id: str) -> bool:
        """Загрузка индекса с диска"""
        try:
            index_file = Path("frame_indices") / f"{video_id}_index.pkl"
            if index_file.exists():
                with open(index_file, 'rb') as f:
                    self.frame_indices[video_id] = pickle.load(f)
                logger.info(f"Frame index loaded for {video_id}")
                return True
        except Exception as e:
            logger.warning(f"Could not load index for {video_id}: {e}")
        return False
    
    def get_frame_fast(self, video_id: str, timestamp: float) -> Optional[np.ndarray]:
        """Быстрое получение кадра через индекс"""
        start_time = time.time()
        
        # Проверяем кеш
        cache_key = f"{video_id}_{timestamp:.3f}"
        if timestamp in self.frame_cache[video_id]:
            self.stats['cache_hits'] += 1
            return self.frame_cache[video_id][timestamp]
        
        try:
            # Поиск ближайшего индекса
            indices = self.frame_indices.get(video_id, [])
            if not indices:
                logger.warning(f"No index found for {video_id}")
                return self._get_frame_sequential(video_id, timestamp)
            
            # Бинарный поиск по временным меткам
            target_frame = int(timestamp * self.video_metadata[video_id]['fps'])
            closest_index = self._binary_search_frame(indices, target_frame)
            
            if closest_index is None:
                return None
            
            # Устанавливаем позицию через индекс
            cap = self.videos[video_id]
            cap.set(cv2.CAP_PROP_POS_FRAMES, closest_index.frame_number)
            
            ret, frame = cap.read()
            if not ret:
                return None
            
            # Кешируем результат
            if len(self.frame_cache[video_id]) < self.max_cache_size:
                self.frame_cache[video_id][timestamp] = frame.copy()
            
            access_time = time.time() - start_time
            self.stats['index_access_time'].append(access_time)
            self.stats['frames_processed'] += 1
            
            return frame
            
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def _binary_search_frame(self, indices: List[FrameIndex], target_frame: int) -> Optional[FrameIndex]:
        """Бинарный поиск ближайшего кадра в индексе"""
        if not indices:
            return None
        
        left, right = 0, len(indices) - 1
        closest = indices[0]
        
        while left <= right:
            mid = (left + right) // 2
            current = indices[mid]
            
            if abs(current.frame_number - target_frame) < abs(closest.frame_number - target_frame):
                closest = current
            
            if current.frame_number < target_frame:
                left = mid + 1
            elif current.frame_number > target_frame:
                right = mid - 1
            else:
                return current
        
        return closest
    
    def _get_frame_sequential(self, video_id: str, timestamp: float) -> Optional[np.ndarray]:
        """Fallback: последовательное получение кадра"""
        cap = self.videos[video_id]
        fps = self.video_metadata[video_id]['fps']
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        return frame if ret else None
    
    def batch_inference(self, frames: List[np.ndarray]) -> List[Any]:
        """Batch инференс для множества кадров"""
        if not frames or not self.model:
            return [None] * len(frames)
        
        start_time = time.time()
        
        try:
            # Подготовка batch
            batch_tensors = []
            for frame in frames:
                # Preprocessing для вашей модели
                tensor = self._preprocess_frame(frame)
                batch_tensors.append(tensor)
            
            # Stack в batch
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            # Инференс
            with torch.no_grad():
                batch_output = self.model(batch_input)
            
            # Постобработка
            results = []
            for i, output in enumerate(batch_output):
                result = self._postprocess_output(output, frames[i])
                results.append(result)
            
            inference_time = time.time() - start_time
            self.stats['gpu_inference_time'].append(inference_time)
            self.stats['batch_sizes'].append(len(frames))
            
            logger.info(f"Batch inference: {len(frames)} frames in {inference_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            return [None] * len(frames)
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocessing кадра для модели"""
        # Пример preprocessing
        frame = cv2.resize(frame, (640, 640))
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(frame)
    
    def _postprocess_output(self, output: torch.Tensor, original_frame: np.ndarray) -> Dict:
        """Постобработка результата инференса"""
        # Пример постобработки
        output_cpu = output.cpu().numpy()
        
        # Заглушка - замените на вашу логику
        return {
            'detections': len(output_cpu),
            'confidence': float(np.mean(output_cpu)),
            'processing_time': time.time()
        }
    
    def queue_batch_inference(self, video_id: str, timestamps: List[float], callback: callable, priority: int = 0):
        """Добавление в очередь batch инференса"""
        frames = []
        for ts in timestamps:
            frame = self.get_frame_fast(video_id, ts)
            if frame is not None:
                frames.append(frame)
        
        if frames:
            request = BatchInferenceRequest(
                frames=frames,
                timestamps=timestamps,
                callback=callback,
                priority=priority
            )
            
            self.batch_queue.append(request)
            self._process_batch_queue()
    
    def _process_batch_queue(self):
        """Обработка очереди batch инференса"""
        if len(self.batch_queue) >= self.batch_size or len(self.batch_queue) > 0:
            # Собираем batch
            batch_requests = []
            batch_frames = []
            
            while len(batch_requests) < self.batch_size and self.batch_queue:
                request = self.batch_queue.popleft()
                batch_requests.append(request)
                batch_frames.extend(request.frames)
            
            if batch_frames:
                # Запускаем инференс в отдельном потоке
                self.batch_executor.submit(self._execute_batch_inference, batch_requests, batch_frames)
    
    def _execute_batch_inference(self, requests: List[BatchInferenceRequest], frames: List[np.ndarray]):
        """Выполнение batch инференса"""
        results = self.batch_inference(frames)
        
        # Распределяем результаты по запросам
        result_idx = 0
        for request in requests:
            request_results = results[result_idx:result_idx + len(request.frames)]
            result_idx += len(request.frames)
            
            # Вызываем callback
            try:
                request.callback(request_results, request.timestamps)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Получение статистики производительности"""
        gpu_info = {}
        if self.use_cuda and torch.cuda.is_available():
            gpu_info = {
                'gpu_memory_used': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9,
                'gpu_utilization': self._get_gpu_utilization()
            }
        
        avg_inference_time = np.mean(self.stats['gpu_inference_time']) if self.stats['gpu_inference_time'] else 0
        avg_batch_size = np.mean(self.stats['batch_sizes']) if self.stats['batch_sizes'] else 0
        avg_index_time = np.mean(self.stats['index_access_time']) if self.stats['index_access_time'] else 0
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['frames_processed'], 1),
            'average_inference_time': f"{avg_inference_time:.3f}s",
            'average_batch_size': f"{avg_batch_size:.1f}",
            'average_index_access_time': f"{avg_index_time:.3f}s",
            'active_videos': len(self.videos),
            'total_indices': sum(len(idx) for idx in self.frame_indices.values()),
            'cache_size': sum(len(cache) for cache in self.frame_cache.values()),
            'batch_queue_size': len(self.batch_queue),
            **gpu_info
        }
    
    def _get_gpu_utilization(self) -> float:
        """Получение утилизации GPU"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0
    
    def clear_cache(self, video_id: str = None):
        """Очистка кеша"""
        if video_id:
            self.frame_cache.pop(video_id, None)
        else:
            self.frame_cache.clear()
        
        if self.use_cuda:
            torch.cuda.empty_cache()
    
    def close_video(self, video_id: str):
        """Закрытие видео и освобождение ресурсов"""
        if video_id in self.videos:
            self.videos[video_id].release()
            del self.videos[video_id]
        
        self.video_metadata.pop(video_id, None)
        self.frame_indices.pop(video_id, None)
        self.frame_cache.pop(video_id, None)
        
        logger.info(f"Video {video_id} closed and resources freed")
    
    def __del__(self):
        """Cleanup при удалении"""
        for cap in self.videos.values():
            cap.release()
        
        if hasattr(self, 'batch_executor'):
            self.batch_executor.shutdown(wait=True)


# Глобальный экземпляр процессора
gpu_processor = None

def get_gpu_processor() -> GPUVideoProcessor:
    """Получение singleton экземпляра GPU процессора"""
    global gpu_processor
    if gpu_processor is None:
        gpu_processor = GPUVideoProcessor(
            use_cuda=True,
            max_cache_size=2000,
            batch_size=16,
            index_keyframes_only=False
        )
    return gpu_processor

if __name__ == "__main__":
    # Тестирование
    processor = get_gpu_processor()
    print("GPU Video Processor initialized")
    print(f"Device: {processor.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
