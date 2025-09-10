import asyncio
import time
import contextlib
from typing import List, Dict, Any, Optional
import os
import numpy as np
import cv2
import json
import logging
from datetime import datetime

from core.frame_broker import FRAME_BROKER
from core.results_store import RESULTS_STORE
from core.preprocess import preprocess_for_model, map_polys_to_original
from core.optimized_preprocess import adaptive_preprocess, center_crop_resize
from services.model_adapter import ModelAdapter

logger = logging.getLogger("inference_worker")


class InferenceWorker:
    """Simple inference worker subscribed to FRAME_BROKER.

    Performs batching by time or size and writes results to RESULTS_STORE.
    Uses model adapter (simple ultralytics fallback here).
    """

    def __init__(self, stream_id: str, batch_size: int = 8, max_wait_ms: int = 50):
        self.stream_id = stream_id
        self.batch_size = int(batch_size)  # Увеличено до 8 по умолчанию
        self.max_wait_ms = int(max_wait_ms)
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Performance tracking (упрощено)
        self._batch_times = []
        self._inference_times = []

        # model loading deferred
        model_path = os.getenv('MODEL_PATH', '')
        self.model = ModelAdapter(model_path) if model_path else None

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        # Log final performance summary
        self._log_performance_summary()

    def _log_performance_summary(self):
        """Простая сводка производительности"""
        if not self._batch_times:
            return

        total_batches = len(self._batch_times)
        avg_batch_time = sum(self._batch_times) / total_batches
        avg_inference_time = sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0
        throughput_fps = (total_batches * self.batch_size) / sum(self._batch_times) if self._batch_times else 0

        logger.info(f"Сводка для {self.stream_id}: батчей={total_batches}, средний_fps={throughput_fps:.1f}, средний_инференс={avg_inference_time*1000:.0f}мс")

    def _log_batch_performance(self, batch_start: float, inference_start: float,
                              postprocess_start: float, batch_end: float,
                              batch_size: int, detections: int):
        """Упрощенное логирование производительности батча"""
        batch_time = batch_end - batch_start
        inference_time = postprocess_start - inference_start
        fps = batch_size / batch_time if batch_time > 0 else 0

        # Store for summary
        self._batch_times.append(batch_time)
        self._inference_times.append(inference_time)

        # Простое логирование в консоль
        logger.info(f"{self.stream_id}: размер={batch_size}, обнаружено={detections}, fps={fps:.1f}, инференс={inference_time*1000:.0f}мс")

    async def _run(self):
        q = FRAME_BROKER.subscribe(self.stream_id, max_queue=32)
        try:
            while self._running:
                # gather first item
                batch_start = time.time()
                try:
                    first = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                batch = [first]
                # collect until batch_size or max_wait
                while len(batch) < self.batch_size:
                    remaining = max(0.0, (self.max_wait_ms / 1000.0) - (time.time() - batch_start))
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # prepare frames for model
                inference_start = time.time()
                jpegs = [b.get('jpeg') for b in batch]
                frame_ids = [b.get('frame_id') for b in batch]

                imgs = []
                proc_meta = []
                orig_sizes = []
                for jpg in jpegs:
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        imgs.append(None)
                        proc_meta.append(None)
                        orig_sizes.append((0,0))
                    else:
                        orig_h, orig_w = frame.shape[:2]
                        # Используем оптимизированную предобработку для соответствия датасету
                        use_optimized = os.getenv('USE_OPTIMIZED_PREPROCESSING', 'true').lower() == 'true'
                        target_size = int(os.getenv('IMG_SIZE', '960'))
                        
                        if use_optimized:
                            # Новая оптимизированная предобработка, соответствующая датасету
                            p = center_crop_resize(frame, target_size)
                        else:
                            # Старая предобработка (для совместимости)
                            p = preprocess_for_model(frame, target_size=target_size, use_hsv=(os.getenv('USE_HSV','false').lower()=='true'))
                        imgs.append(p.get('img'))
                        # Адаптируем метаданные для совместимости
                        if 'scale' in p and 'pad' in p:
                            # Старый формат (letterbox)
                            proc_meta.append({'scale': p.get('scale'), 'pad': p.get('pad')})
                        else:
                            # Новый формат (center_crop) - создаем совместимые метаданные
                            proc_meta.append({
                                'scale': 1.0,  # center crop не масштабирует, только обрезает
                                'pad': (0, 0, 0, 0),  # padding не используется
                                'method': p.get('method', 'unknown'),
                                'original_size': p.get('original_size', (orig_w, orig_h))
                            })
                        orig_sizes.append((orig_w, orig_h))

                # run model if available
                postprocess_start = time.time()
                if self.model:
                    inputs = [im for im in imgs if im is not None]
                    results = self.model.infer(inputs)
                    # map results back to frame ids (simple mapping)
                    ri = 0
                    for fid, im, meta, orig in zip(frame_ids, imgs, proc_meta, orig_sizes):
                        if im is None or meta is None:
                            RESULTS_STORE.put(self.stream_id, fid, {'detections': 0, 'confidence': 0.0})
                        else:
                            res = results[ri] if ri < len(results) else {'detections': 0, 'confidence': 0.0}
                            # if masks present, map back to original frame coords
                            try:
                                if res and 'masks' in res and res['masks']:
                                    mapped = map_polys_to_original(res['masks'], meta['scale'], meta['pad'], orig)
                                    res['masks'] = mapped
                            except Exception:
                                pass
                            RESULTS_STORE.put(self.stream_id, fid, res)
                            # broadcast delta with throttling
                            try:
                                # dynamic import to avoid circular imports
                                from core.results_store import RESULTS_STORE as _RS
                                from api.app import STREAM_MANAGER
                                nowt = time.time()
                                MIN_INTERVAL = float(os.getenv('BROADCAST_MIN_INTERVAL', '0.09'))  # seconds ~11 Hz
                                if not hasattr(self, '_last_broadcast'):
                                    self._last_broadcast = {}
                                last = self._last_broadcast.get(self.stream_id, 0.0)
                                prev = _RS.get_for_frame(self.stream_id, fid)
                                changed = False
                                try:
                                    if prev is None:
                                        changed = True
                                    else:
                                        if prev.get('detections') != res.get('detections'):
                                            changed = True
                                        elif abs(float(prev.get('confidence', 0.0)) - float(res.get('confidence', 0.0))) > 0.01:
                                            changed = True
                                except Exception:
                                    changed = True
                                if changed and (nowt - last) >= MIN_INTERVAL:
                                    payload = {"type": "count_update", "count": int(res.get('detections', 0)), "debug": {"confidence": float(res.get('confidence', 0.0)), "model": {"path": os.getenv('MODEL_PATH',''), "name": os.path.basename(os.getenv('MODEL_PATH',''))}}}
                                    try:
                                        # fire-and-forget
                                        asyncio.create_task(STREAM_MANAGER.broadcast(self.stream_id, payload))
                                    except Exception:
                                        pass
                                    self._last_broadcast[self.stream_id] = nowt
                            except Exception:
                                pass
                            ri += 1
                else:
                    # fallback simple heuristic
                    for fid, jpg in zip(frame_ids, jpegs):
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            RESULTS_STORE.put(self.stream_id, fid, {'detections': 0, 'confidence': 0.0})
                        else:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            nz = int((gray > 10).sum() / 10000)
                            RESULTS_STORE.put(self.stream_id, fid, {'detections': max(0, nz), 'confidence': 0.5})

                # Log performance for this batch
                batch_end = time.time()
                total_detections = sum(1 for fid in frame_ids if RESULTS_STORE.get_for_frame(self.stream_id, fid) and
                                     RESULTS_STORE.get_for_frame(self.stream_id, fid).get('detections', 0) > 0)
                self._log_batch_performance(batch_start, inference_start, postprocess_start,
                                          batch_end, len(batch), total_detections)

        finally:
            FRAME_BROKER.unsubscribe(self.stream_id, q)


def start_global_worker_for(stream_id: str, batch_size: int = 8, max_wait_ms: int = 50):
    # helper to start a worker; keeps simple global map on module
    global _WORKERS
    try:
        _WORKERS
    except NameError:
        _WORKERS = {}
    if stream_id in _WORKERS:
        return _WORKERS[stream_id]
    w = InferenceWorker(stream_id, batch_size=batch_size, max_wait_ms=max_wait_ms)
    w.start()
    _WORKERS[stream_id] = w
    return w


def stop_global_worker_for(stream_id: str):
    global _WORKERS
    try:
        _WORKERS
    except NameError:
        _WORKERS = {}
    w = _WORKERS.get(stream_id)
    if not w:
        return False
    try:
        # schedule stop
        asyncio.create_task(w.stop())
    except Exception:
        pass
    try:
        del _WORKERS[stream_id]
    except Exception:
        pass
    return True


def is_worker_running(stream_id: str) -> bool:
    try:
        return bool(_WORKERS.get(stream_id))
    except Exception:
        return False


