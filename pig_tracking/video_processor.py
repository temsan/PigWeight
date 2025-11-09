"""
Интегрированный видео-процессор для системы отслеживания свиней.
Объединяет UnifiedVideoProcessor, SimpleTracker, CrossingCounter и ActDetector.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

import cv2
import numpy as np

from core.processor import get_processor, ProcessingOptions, FrameResult
from core.config import CONFIG
from pig_tracking.crossing_counter import CrossingCounter, CrossingEvent
from pig_tracking.act_detector import ActDetector, WeighingAct
from pig_tracking.weight_estimator import get_weight_estimator

# SimpleTracker будет импортирован лениво, чтобы избежать циклического импорта
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Ленивый импорт SimpleTracker
_SimpleTracker = None

def _get_simple_tracker():
    """Ленивый импорт SimpleTracker для избежания циклических импортов"""
    global _SimpleTracker
    if _SimpleTracker is None:
        # Используем отдельный модуль вместо api.app чтобы избежать инициализации БД
        from pig_tracking.simple_tracker import SimpleTracker as ST
        _SimpleTracker = ST
    return _SimpleTracker


class IntegratedVideoProcessor:
    """
    Интегрированный процессор для обработки видео с отслеживанием свиней.
    
    Объединяет:
    - UnifiedVideoProcessor: детекция и сегментация
    - SimpleTracker: отслеживание объектов
    - CrossingCounter: подсчет пересечений линий
    - ActDetector: определение актов взвешивания
    """
    
    def __init__(
        self,
        stream_id: str = "video_processor",
        conf_threshold: float = 0.30,
        img_size: int = 960,
        line_left_x: float = 0.25,
        line_right_x: float = 0.75,
        min_pigs_for_act: int = 3,
        max_interval_sec: float = 30.0
    ):
        self.stream_id = stream_id
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        
        # Компоненты обработки
        self.processor: Optional[Any] = None
        SimpleTracker = _get_simple_tracker()
        self.tracker = SimpleTracker(iou_threshold=0.3, max_age=30, dist_weight=0.2)
        self.crossing_counter = CrossingCounter(
            line_left_x=line_left_x,
            line_right_x=line_right_x,
            cooldown_sec=CONFIG.CROSS_COOLDOWN_SEC
        )
        self.act_detector = ActDetector(
            min_pigs_for_act=min_pigs_for_act,
            max_interval_sec=max_interval_sec
        )
        self.weight_estimator = get_weight_estimator()
        
        # Статистика
        self.frames_processed = 0
        self.total_detections = 0
        self.processing_times: List[float] = []
        
        logger.info(
            f"IntegratedVideoProcessor инициализирован: stream_id={stream_id}, "
            f"conf={conf_threshold}, img_size={img_size}"
        )
    
    async def initialize(self):
        """Инициализирует процессор (асинхронно)"""
        options = ProcessingOptions(
            conf_threshold=self.conf_threshold,
            img_size=self.img_size
        )
        self.processor = await get_processor(self.stream_id, options)
        logger.info(f"Процессор инициализирован для {self.stream_id}")
    
    async def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает один кадр через весь пайплайн.
        
        Args:
            frame: кадр изображения (BGR)
            timestamp: временная метка кадра
            
        Returns:
            Словарь с результатами обработки
        """
        if self.processor is None:
            raise RuntimeError("Процессор не инициализирован. Вызовите initialize() сначала.")
        
        start_time = time.time()
        ts = timestamp or time.time()
        
        # 1. Детекция и сегментация
        frame_result: FrameResult = await self.processor.process_frame_async(frame, ts)
        
        # 2. Трекинг
        detections = []
        if frame_result.bboxes:
            for i, bbox in enumerate(frame_result.bboxes):
                detections.append({
                    'bbox': bbox,
                    'confidence': frame_result.confidence
                })
        
        tracked_objects = self.tracker.update(detections)
        
        # 3. Нормализация координат для CrossingCounter
        h, w = frame.shape[:2]
        track_ids = []
        centers_x = []
        centers_y = []
        
        for obj in tracked_objects:
            track_ids.append(obj['id'])
            bbox = obj['bbox']
            # Вычисляем центр и нормализуем
            cx = (bbox[0] + bbox[2]) / 2 / w
            cy = (bbox[1] + bbox[3]) / 2 / h
            centers_x.append(cx)
            centers_y.append(cy)
        
        # 4. Подсчет пересечений с оценкой веса
        crossing_events = self.crossing_counter.process_tracks(
            track_ids, centers_x, centers_y
        )
        
        # Добавляем оценку веса для каждого пересечения
        for event in crossing_events:
            # Оцениваем вес для каждой свиньи при пересечении
            event.weight_estimate = self.weight_estimator.estimate_weight(
                pig_id=event.track_id
            )
        
        # 5. Определение актов взвешивания
        current_count = len(tracked_objects)
        completed_act = self.act_detector.update(
            crossing_events, current_count, ts
        )
        
        # Добавляем оценку веса в завершенный акт
        if completed_act:
            # Общий вес = сумма весов всех пересечений
            total_weight = sum(
                e.weight_estimate for e in completed_act.crossings 
                if e.weight_estimate
            )
            completed_act.total_weight = round(total_weight, 1) if total_weight > 0 else None
            
            # Средний вес
            if completed_act.crossings:
                completed_act.avg_weight = round(
                    total_weight / len(completed_act.crossings), 1
                ) if total_weight > 0 else None
        
        # Обновляем статистику
        self.frames_processed += 1
        self.total_detections += frame_result.detections
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Формируем результат
        result = {
            'timestamp': ts,
            'frame_number': self.frames_processed,
            'detections': frame_result.detections,
            'tracked_objects': tracked_objects,
            'current_count': current_count,
            'crossing_events': [
                {
                    'track_id': e.track_id,
                    'side': e.side,
                    'mode': e.mode,
                    'x': e.x,
                    'y': e.y,
                    'timestamp': e.timestamp,
                    'weight_estimate': e.weight_estimate
                }
                for e in crossing_events
            ],
            'crossing_stats': self.crossing_counter.get_stats(),
            'act_stats': self.act_detector.get_stats(),
            'completed_act': completed_act.to_dict() if completed_act else None,
            'processing_time': processing_time,
            'masks': frame_result.masks,
            'bboxes': frame_result.bboxes
        }
        
        return result
    
    async def process_video_file(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает видеофайл целиком или RTSP поток.
        
        Args:
            video_path: путь к видеофайлу или RTSP URL
            progress_callback: функция для отчета о прогрессе (frame_num, total_frames)
            
        Returns:
            Итоговая статистика обработки
        """
        # Проверяем если это RTSP URL
        is_rtsp = isinstance(video_path, str) and video_path.startswith("rtsp://")
        
        if not is_rtsp:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            logger.info(f"Начало обработки видео: {video_path}")
        else:
            logger.info(f"Начало обработки RTSP потока: {video_path}")
        
        # Открываем видео или RTSP поток с retry логикой
        cap = None
        max_retries = 3
        retry_delay = 2  # секунды
        
        for attempt in range(max_retries):
            try:
                cap = cv2.VideoCapture(str(video_path))
                
                # Для RTSP устанавливаем параметры подключения
                if is_rtsp:
                    # Увеличиваем таймауты для RTSP
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер
                    cap.set(cv2.CAP_PROP_FPS, 25)  # Ожидаемый FPS
                
                # Пробуем прочитать первый кадр
                ret, frame = cap.read()
                if ret:
                    logger.info(f"✓ Подключение успешно (попытка {attempt + 1}/{max_retries})")
                    break
                else:
                    cap.release()
                    cap = None
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Попытка подключения {attempt + 1} неудачна, пересоединяюсь...")
                        await asyncio.sleep(retry_delay)
                    
            except Exception as e:
                logger.warning(f"⚠️ Ошибка подключения (попытка {attempt + 1}/{max_retries}): {e}")
                if cap:
                    cap.release()
                    cap = None
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
        
        if cap is None or not cap.isOpened():
            error_msg = f"Не удалось подключиться к потоку после {max_retries} попыток: {video_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Получаем метаданные
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(
            f"Видео: {total_frames} кадров, {fps:.2f} FPS, "
            f"длительность {duration:.1f}s"
        )
        
        # Обрабатываем кадры
        frame_num = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                timestamp = frame_num / fps
                
                # Обрабатываем кадр
                await self.process_frame(frame, timestamp)
                
                # Отчет о прогрессе
                if progress_callback and frame_num % 30 == 0:
                    progress_callback(frame_num, total_frames)
                
                # Логируем прогресс каждые 5%
                if frame_num % max(1, total_frames // 20) == 0:
                    progress = (frame_num / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_actual = frame_num / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_num) / fps_actual if fps_actual > 0 else 0
                    
                    # Визуальный прогресс-бар
                    bar_length = 30
                    filled = int(bar_length * frame_num / total_frames)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    print(
                        f"\r[{bar}] {progress:.1f}% | "
                        f"{frame_num}/{total_frames} кадров | "
                        f"{fps_actual:.1f} FPS | "
                        f"ETA: {int(eta)}s",
                        end='', flush=True
                    )
        
        finally:
            cap.release()
            
            # Завершаем текущий акт, если он активен
            final_act = self.act_detector.force_complete_current_act()
            if final_act:
                logger.info(f"Завершен финальный акт #{final_act.act_id}")
        
        # Формируем итоговую статистику
        total_time = time.time() - start_time
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        summary = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'frames_processed': self.frames_processed,
            'video_duration': duration,
            'processing_time': total_time,
            'avg_fps': self.frames_processed / total_time if total_time > 0 else 0,
            'avg_frame_time': avg_processing_time,
            'total_detections': self.total_detections,
            'crossing_stats': self.crossing_counter.get_stats(),
            'act_stats': self.act_detector.get_stats()
        }
        
        logger.info(
            f"Обработка завершена: {self.frames_processed} кадров за {total_time:.1f}s, "
            f"средний FPS: {summary['avg_fps']:.1f}"
        )
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает текущую статистику"""
        return {
            'frames_processed': self.frames_processed,
            'total_detections': self.total_detections,
            'avg_processing_time': (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0
            ),
            'crossing_stats': self.crossing_counter.get_stats(),
            'act_stats': self.act_detector.get_stats()
        }
    
    def reset(self):
        """Сбрасывает все счетчики и состояние"""
        SimpleTracker = _get_simple_tracker()
        self.tracker = SimpleTracker(iou_threshold=0.3, max_age=30, dist_weight=0.2)
        self.crossing_counter.reset()
        self.act_detector.reset()
        self.frames_processed = 0
        self.total_detections = 0
        self.processing_times.clear()
        logger.info("IntegratedVideoProcessor сброшен")
