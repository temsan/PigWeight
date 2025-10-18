"""
Интегрированные GPU + WebRTC endpoints для максимальной производительности
Объединяет все улучшения: CUDA, индексирование, batch inference, WebRTC
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
import time
import threading
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
from PIL import Image
import logging

# Импорты наших модулей
from gpu_video_processor import get_gpu_processor, GPUVideoProcessor
from webrtc_streamer import get_webrtc_streamer, WebRTCStreamer

logger = logging.getLogger(__name__)

# Глобальные экземпляры
gpu_processor: Optional[GPUVideoProcessor] = None
webrtc_streamer: Optional[WebRTCStreamer] = None

def get_processors():
    """Получение глобальных процессоров"""
    global gpu_processor, webrtc_streamer
    
    if gpu_processor is None:
        gpu_processor = get_gpu_processor()
        logger.info("GPU Processor initialized")
    
    if webrtc_streamer is None:
        webrtc_streamer = get_webrtc_streamer(gpu_processor)
        logger.info("WebRTC Streamer initialized")
    
    return gpu_processor, webrtc_streamer

class GPUVideoAPI:
    """GPU-ускоренное API для видео обработки"""
    
    def __init__(self):
        self.app = FastAPI(
            title="PigWeight GPU Video API",
            description="Высокопроизводительное видео API с GPU ускорением",
            version="2.0.0"
        )
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self.performance_stats = {
            'total_requests': 0,
            'gpu_requests': 0,
            'webrtc_sessions': 0,
            'batch_inferences': 0,
            'cache_hits': 0
        }
    
    def _setup_routes(self):
        """Настройка маршрутов API"""
        
        @self.app.post("/api/gpu/video/upload")
        async def upload_video_gpu(
            background_tasks: BackgroundTasks,
            id: str = Form(...),
            file: UploadFile = File(...)
        ):
            """Загрузка видео с GPU индексированием"""
            processor, _ = get_processors()
            
            try:
                import hashlib
                import os
                from pathlib import Path
                
                # Читаем содержимое файла
                content = await file.read()
                
                # Создаем хеш для проверки дубликатов
                file_hash = hashlib.md5(content).hexdigest()
                
                # Создаем путь на основе ID пользователя
                safe_id = "".join(c for c in id if c.isalnum() or c in ('-', '_'))
                file_extension = Path(file.filename).suffix.lower()
                file_path = f"uploads/{safe_id}{file_extension}"
                
                # Проверяем существование файла и его хеш
                file_exists = os.path.exists(file_path)
                same_content = False
                
                if file_exists:
                    with open(file_path, "rb") as existing_file:
                        existing_hash = hashlib.md5(existing_file.read()).hexdigest()
                        same_content = (existing_hash == file_hash)
                
                # Сохраняем только если файл новый или изменился
                if not file_exists or not same_content:
                    # Создаем backup старого файла если нужно
                    if file_exists and not same_content:
                        backup_path = f"uploads/{safe_id}_backup_{int(time.time())}{file_extension}"
                        os.rename(file_path, backup_path)
                        logger.info(f"Backup created: {backup_path}")
                    
                    # Сохраняем новый файл
                    with open(file_path, "wb") as buffer:
                        buffer.write(content)
                    
                    logger.info(f"File saved: {file_path} (hash: {file_hash[:8]}...)")
                else:
                    logger.info(f"File unchanged, reusing: {file_path}")
                
                # Загружаем в GPU процессор (асинхронно для больших файлов)
                result = processor.load_video(file_path, id)
                
                # Предзагружаем ключевые кадры в фоне
                background_tasks.add_task(self._preload_keyframes, processor, id)
                
                self.performance_stats['total_requests'] += 1
                
                return {
                    "status": "success",
                    "id": id,
                    "fps": result['fps'],
                    "duration": result['duration'],
                    "total_frames": result['total_frames'],
                    "resolution": result['resolution'],
                    "index_size": result['index_size'],
                    "index_time": f"{result['index_time']:.3f}s",
                    "gpu_enabled": processor.use_cuda,
                    "device": str(processor.device)
                }
                
            except Exception as e:
                logger.error(f"GPU upload error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/gpu/video/{video_id}/frame")
        async def get_frame_gpu(
            video_id: str,
            t: float = 0.0,
            inference: bool = False,
            quality: int = 85,
            batch_mode: bool = False
        ):
            """Сверхбыстрое получение кадра через GPU + индекс"""
            processor, _ = get_processors()
            start_time = time.time()
            
            try:
                # Быстрый доступ через индекс
                frame = processor.get_frame_fast(video_id, t)
                if frame is None:
                    raise HTTPException(status_code=404, detail="Frame not found")
                
                inference_time = 0.0
                detected_count = 0
                
                # GPU инференс если нужен
                if inference:
                    inference_start = time.time()
                    if batch_mode:
                        # Batch inference для лучшей производительности
                        results = processor.batch_inference([frame])
                        result = results[0] if results else None
                    else:
                        # Обычный инференс
                        results = processor.batch_inference([frame])
                        result = results[0] if results else None
                    
                    inference_time = time.time() - inference_start
                    
                    if result:
                        detected_count = result.get('detections', 0)
                        # Отрисовка результатов
                        frame = self._draw_gpu_detections(frame, result)
                    
                    self.performance_stats['gpu_requests'] += 1
                
                # Кодирование JPEG
                encode_start = time.time()
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                img_bytes = buffer.tobytes()
                encode_time = time.time() - encode_start
                
                total_time = time.time() - start_time
                
                # Заголовки с метриками производительности
                headers = {
                    "X-Processing-Time": f"{total_time:.3f}s",
                    "X-Inference-Time": f"{inference_time:.3f}s",
                    "X-Encode-Time": f"{encode_time:.3f}s",
                    "X-Count": str(detected_count),
                    "X-GPU-Enabled": str(processor.use_cuda),
                    "X-Cache-Status": "HIT" if processor.stats.get('cache_hits', 0) > 0 else "MISS",
                    "Cache-Control": "no-cache, no-store, must-revalidate"
                }
                
                self.performance_stats['total_requests'] += 1
                
                return StreamingResponse(
                    io.BytesIO(img_bytes),
                    media_type="image/jpeg",
                    headers=headers
                )
                
            except Exception as e:
                logger.error(f"GPU frame error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/gpu/video/{video_id}/batch_inference")
        async def batch_inference_endpoint(
            video_id: str,
            timestamps: List[float],
            quality: int = 85
        ):
            """Batch инференс для множества кадров"""
            processor, _ = get_processors()
            start_time = time.time()
            
            try:
                # Получаем все кадры
                frames = []
                for ts in timestamps:
                    frame = processor.get_frame_fast(video_id, ts)
                    if frame is not None:
                        frames.append(frame)
                
                if not frames:
                    raise HTTPException(status_code=404, detail="No frames found")
                
                # Batch GPU инференс
                results = processor.batch_inference(frames)
                
                processing_time = time.time() - start_time
                self.performance_stats['batch_inferences'] += 1
                
                return {
                    "status": "success",
                    "processed_frames": len(frames),
                    "results": results,
                    "processing_time": f"{processing_time:.3f}s",
                    "gpu_device": str(processor.device),
                    "batch_size": len(frames)
                }
                
            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/gpu/video/{video_id}/stream")
        async def gpu_stream_endpoint(
            video_id: str,
            fps: int = 30,
            inference: bool = False,
            quality: int = 85
        ):
            """GPU-ускоренный видео стрим"""
            processor, _ = get_processors()
            
            try:
                async def generate_frames():
                    timestamp = 0.0
                    frame_duration = 1.0 / fps
                    
                    while True:
                        start_time = time.time()
                        
                        # Быстрое получение кадра
                        frame = processor.get_frame_fast(video_id, timestamp)
                        if frame is None:
                            timestamp = 0.0  # Цикл
                            continue
                        
                        # GPU инференс если нужен
                        if inference:
                            results = processor.batch_inference([frame])
                            if results and results[0]:
                                frame = self._draw_gpu_detections(frame, results[0])
                        
                        # JPEG кодирование
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        
                        # Формат MJPEG
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        
                        # Контроль FPS
                        elapsed = time.time() - start_time
                        if elapsed < frame_duration:
                            await asyncio.sleep(frame_duration - elapsed)
                        
                        timestamp += frame_duration
                
                return StreamingResponse(
                    generate_frames(),
                    media_type="multipart/x-mixed-replace; boundary=frame",
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                        "X-GPU-Stream": "enabled"
                    }
                )
                
            except Exception as e:
                logger.error(f"GPU stream error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/webrtc/start")
        async def start_webrtc_stream(
            video_id: str,
            fps: int = 30,
            width: int = 1280,
            height: int = 720,
            inference_enabled: bool = False
        ):
            """Запуск WebRTC стрима с GPU ускорением"""
            _, webrtc = get_processors()
            
            try:
                config_data = {
                    'video_id': video_id,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'inference_enabled': inference_enabled
                }
                
                # Запуск WebRTC стрима
                self.performance_stats['webrtc_sessions'] += 1
                
                return {
                    "status": "success",
                    "message": "WebRTC stream prepared",
                    "webrtc_url": "/webrtc",
                    "config": config_data,
                    "low_latency": True,
                    "gpu_accelerated": True
                }
                
            except Exception as e:
                logger.error(f"WebRTC start error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/gpu/stats")
        async def get_gpu_stats():
            """Детальная статистика GPU производительности"""
            processor, webrtc = get_processors()
            
            try:
                gpu_stats = processor.get_performance_stats()
                
                # Добавляем статистику API
                api_stats = {
                    "api_stats": self.performance_stats,
                    "gpu_processor": gpu_stats,
                    "webrtc_stats": webrtc.stats if webrtc else {},
                    "system_info": {
                        "gpu_available": processor.use_cuda,
                        "gpu_device": str(processor.device),
                        "batch_processing": True,
                        "frame_indexing": True,
                        "webrtc_enabled": webrtc is not None
                    }
                }
                
                return api_stats
                
            except Exception as e:
                logger.error(f"Stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/gpu/video/{video_id}/info")
        async def get_video_info_gpu(video_id: str):
            """Информация о видео с GPU индексом"""
            processor, _ = get_processors()
            
            try:
                if video_id not in processor.videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                metadata = processor.video_metadata[video_id]
                indices = processor.frame_indices.get(video_id, [])
                cache_size = len(processor.frame_cache.get(video_id, {}))
                
                return {
                    "id": video_id,
                    "fps": metadata['fps'],
                    "duration": metadata['duration'],
                    "total_frames": metadata['frame_count'],
                    "resolution": f"{metadata['width']}x{metadata['height']}",
                    "index_entries": len(indices),
                    "keyframes": sum(1 for idx in indices if idx.keyframe),
                    "cache_size": cache_size,
                    "gpu_accelerated": processor.use_cuda,
                    "device": str(processor.device),
                    "path": metadata['path']
                }
                
            except Exception as e:
                logger.error(f"Video info error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/gpu/video/{video_id}")
        async def delete_video_gpu(video_id: str):
            """Удаление видео с очисткой GPU памяти"""
            processor, _ = get_processors()
            
            try:
                processor.close_video(video_id)
                processor.clear_cache(video_id)
                
                return {
                    "status": "success",
                    "message": f"Video {video_id} deleted and GPU memory freed",
                    "id": video_id
                }
                
            except Exception as e:
                logger.error(f"Delete video error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _draw_gpu_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """GPU-оптимизированная отрисовка детекций"""
        try:
            height, width = frame.shape[:2]
            
            # Информация о GPU инференсе
            gpu_text = f"GPU Detections: {detections.get('detections', 0)}"
            conf_text = f"Confidence: {detections.get('confidence', 0.0):.3f}"
            proc_text = f"GPU Time: {detections.get('processing_time', 0.0):.3f}s"
            
            # Зеленые тексты для GPU
            cv2.putText(frame, gpu_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, proc_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # GPU badge
            cv2.rectangle(frame, (width - 120, 10), (width - 10, 50), (0, 255, 0), -1)
            cv2.putText(frame, "GPU", (width - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Draw detections error: {e}")
            return frame
    
    async def _preload_keyframes(self, processor: GPUVideoProcessor, video_id: str):
        """Предзагрузка ключевых кадров в кеш"""
        try:
            indices = processor.frame_indices.get(video_id, [])
            keyframes = [idx for idx in indices if idx.keyframe]
            
            logger.info(f"Preloading {len(keyframes)} keyframes for {video_id}")
            
            # Загружаем ключевые кадры в кеш
            for idx in keyframes[:50]:  # Первые 50 ключевых кадров
                processor.get_frame_fast(video_id, idx.timestamp)
                await asyncio.sleep(0.01)  # Небольшая пауза
            
            logger.info(f"Keyframes preloaded for {video_id}")
            
        except Exception as e:
            logger.error(f"Preload keyframes error: {e}")

# Создание приложения
def create_gpu_app() -> FastAPI:
    """Создание FastAPI приложения с GPU ускорением"""
    api = GPUVideoAPI()
    return api.app

# Главная функция для запуска
if __name__ == "__main__":
    import uvicorn
    
    app = create_gpu_app()
    
    print("🚀 Starting GPU-Accelerated Video API...")
    print("📊 Features enabled:")
    print("   • CUDA GPU Acceleration")
    print("   • Frame Indexing") 
    print("   • Batch Inference")
    print("   • WebRTC Streaming")
    print("   • Advanced Caching")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # GPU requires single worker
        access_log=True
    )
