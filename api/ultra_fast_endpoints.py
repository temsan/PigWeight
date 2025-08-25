"""
Ultra-Fast Video Processing Endpoints
Устраняет лишние этапы кодирования и декодирования для максимальной производительности
"""

import asyncio
import time
import math
import logging
import json
import os
import numpy as np
import cv2
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect, Query, Form, UploadFile, File
from fastapi.responses import Response, JSONResponse

logger = logging.getLogger("ultra_fast")

# Optional PyAV fast seek util
try:
    from api.pyav_utils import pyav_seek_read_jpeg  # returns JPEG bytes for (video_path, t, jpeg_quality)
    _PYAV_OK = True
except Exception:
    try:
        from .pyav_utils import pyav_seek_read_jpeg  # relative import variant
        _PYAV_OK = True
    except Exception:
        _PYAV_OK = False

class UltraFastProcessor:
    """
    Ультра-быстрый процессор видео с минимальными накладными расходами
    """
    
    def __init__(self):
        self.file_sessions = {}  # {file_id: {last_count: int, avg_count: float, last_stats: dict}}
        self.count_ema = {}  # экспоненциальное скользящее среднее для сглаживания
        self.tracks = {}  # трекинг объектов для стабильных цветов масок
        self.count_websockets = set()  # активные /ws/count подключения
        self.frame_cache = {}
        # Модель и путь инициализируем заранее, чтобы избежать AttributeError в init_model/process
        self.model = None
        self.model_path = None
        
        # Оптимизированные настройки
        self.target_fps = 30  # Максимальный FPS
        self.frame_skip = 1   # Обрабатываем каждый кадр для максимальной отзывчивости
        # Простой трекинг по сессиям: id -> цвет и позиции
        # self.tracks[session_id] = {
        #   'next_id': int,
        #   'objects': [{'id': int, 'cx': float, 'cy': float, 'ttl': int}],
        #   'colors': {id: (b,g,r)}
        # }
        self.tracks: dict[str, dict] = {}
    
    async def _notify_count_websockets(self, file_id: str, count: int, avg_count: float):
        """Уведомляем все активные /ws/count подключения о новом счетчике"""
        if not self.count_websockets:
            return
            
        message = {
            "type": "count_update", 
            "file_id": file_id,
            "count": count,
            "avg": avg_count
        }
        
        # Отправляем всем активным WebSocket подключениям
        disconnected = set()
        for ws in self.count_websockets.copy():
            try:
                await ws.send_text(json.dumps(message))
                logger.debug(f"[WS_SYNC] Sent count update to /ws/count: {message}")
            except Exception as e:
                logger.debug(f"[WS_SYNC] WebSocket disconnected: {e}")
                disconnected.add(ws)
        
        # Удаляем отключенные соединения
        self.count_websockets -= disconnected
        
    async def init_model(self, model_path: str):
        """Ленивая инициализация модели"""
        # Разрешаем переопределять путь через переменную окружения
        env_path = os.getenv("ULTRA_MODEL")
        desired_path = env_path or model_path
        # Если путь файловый и отсутствует — пробуем публичную модель по имени
        fallback_name = "yolo11n-seg.pt"
        if self.model is None or self.model_path != desired_path:
            try:
                from ultralytics import YOLO
                load_path = desired_path
                if isinstance(load_path, str) and os.path.sep in load_path and not os.path.exists(load_path):
                    logger.warning(f"Ultra-fast: weights not found at '{load_path}', falling back to '{fallback_name}'")
                    load_path = fallback_name
                self.model = YOLO(load_path)
                self.model_path = load_path
                logger.info(f"Ultra-fast model loaded: {load_path}")
            except Exception as e:
                logger.error(f"Ultra-fast: failed to load model '{desired_path}': {e}")
                # оставить self.model = None, процессинг вернёт кадр без детекций
                
    def encode_raw_frame(self, frame: np.ndarray) -> bytes:
        """
        Прямое кодирование кадра без промежуточных этапов
        Используем несжатый BMP для максимальной скорости
        """
        try:
            # Используем BMP для минимальной нагрузки на CPU
            success, buffer = cv2.imencode('.bmp', frame)
            if success:
                return buffer.tobytes()
            return b''
        except Exception:
            return b''
            
    def encode_jpeg(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """
        Оптимизированное JPEG кодирование с настраиваемым качеством
        """
        try:
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Включаем оптимизацию
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Прогрессивный JPEG для веба
            ]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            if success:
                return buffer.tobytes()
            return b''
        except Exception:
            return b''
            
    async def process_frame_ultra_fast(self, frame: np.ndarray, session_id: str) -> tuple[np.ndarray, dict]:
        """
        Ультра-быстрая обработка кадра с минимальными задержками
        """
        start_time = time.perf_counter()
        
        # Статистика обработки
        stats = {
            'inference_ms': 0,
            'drawing_ms': 0,
            'total_ms': 0,
            'count': 0,
            'avg_count': 0.0
        }
        
        try:
            if self.model is None:
                return frame, stats
                
            # Быстрый инференс
            inference_start = time.perf_counter()
            results = self.model.predict(
                frame,
                imgsz=416,  # Меньший размер для скорости
                conf=0.25,  # Более низкий threshold для responsiveness
                verbose=False,
                device='cpu',  # Можно переключить на 'cuda' если есть GPU
                half=False,
                retina_masks=True  # Возвращать маски в размере исходного кадра (без смещения)
            )
            stats['inference_ms'] = (time.perf_counter() - inference_start) * 1000
            
            # Быстрое извлечение детекций и масок
            detections = []
            masks_np = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if hasattr(boxes, 'xyxy'):
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                        
                        for i, box in enumerate(xyxy):
                            if i < len(conf) and conf[i] > 0.25:
                                detections.append({
                                    'bbox': box.tolist(),
                                    'confidence': float(conf[i])
                                })
                # Маски сегментации (если модель сегментационная)
                try:
                    if hasattr(result, 'masks') and result.masks is not None and hasattr(result.masks, 'data'):
                        md = result.masks.data
                        if hasattr(md, 'cpu'):
                            md = md.cpu().numpy()
                        # md shape: [N, H, W] со значениями [0..1]
                        h, w = frame.shape[:2]
                        for i in range(md.shape[0]):
                            m = md[i]
                            # Приводим размеры маски к размеру кадра
                            if m.shape[0] != h or m.shape[1] != w:
                                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                            masks_np.append(m)
                except Exception:
                    pass
            
            # Приоритетно считаем по маскам, иначе по bbox
            obj_count = len(masks_np) if len(masks_np) > 0 else len(detections)
            stats['count'] = obj_count
            # EMA среднего количества по сессии
            prev = self.count_ema.get(session_id, float(obj_count))
            ema = prev * 0.8 + float(obj_count) * 0.2
            self.count_ema[session_id] = ema
            stats['avg_count'] = ema
            
            # Быстрое рисование без тяжелых эффектов
            drawing_start = time.perf_counter()
            
            # Пастельные маски (без боксов) с трекингом инстансов для стабильных цветов
            if len(masks_np) > 0:
                # Инициализация контейнера треков для сессии
                if session_id not in self.tracks:
                    self.tracks[session_id] = {'next_id': 1, 'objects': [], 'colors': {}}
                tr = self.tracks[session_id]
                obj_prev = tr['objects']  # список прошлых объектов
                colors = tr['colors']

                # Вычисляем центроиды текущих масок
                curr = []  # [{'idx': i, 'cx': ..., 'cy': ...}]
                bin_masks = []  # list of tuples (orig_idx, bm)
                fh, fw = frame.shape[:2]
                for i, m in enumerate(masks_np):
                    bm = (m > 0.5).astype(np.uint8)
                    # Приводим к размеру кадра, чтобы исключить искажения по вертикали/горизонтали
                    if bm.shape[:2] != (fh, fw):
                        bm = cv2.resize(bm, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    if bm.max() == 0:
                        continue
                    bin_masks.append((i, bm))
                    # центроид через моменты
                    mu = cv2.moments((bm*255).astype(np.uint8))
                    if mu['m00'] > 1e-3:
                        cx = mu['m10']/mu['m00']
                        cy = mu['m01']/mu['m00']
                    else:
                        ys, xs = np.where(bm > 0)
                        if xs.size == 0:
                            cx, cy = 0.0, 0.0
                        else:
                            cx, cy = float(xs.mean()), float(ys.mean())
                    curr.append({'idx': i, 'cx': cx, 'cy': cy})

                # Сопоставление по ближайшему центроиду
                assigned = {}  # curr_idx -> id
                used_prev = set()
                max_dist = 80.0  # пикселей
                for c in curr:
                    best_id = None
                    best_d2 = max_dist*max_dist
                    for p in obj_prev:
                        if p['id'] in used_prev:
                            continue
                        dx = c['cx'] - p['cx']
                        dy = c['cy'] - p['cy']
                        d2 = dx*dx + dy*dy
                        if d2 < best_d2:
                            best_d2 = d2
                            best_id = p['id']
                    if best_id is not None:
                        assigned[c['idx']] = best_id
                        used_prev.add(best_id)

                # Новые id для не сопоставленных
                for c in curr:
                    if c['idx'] not in assigned:
                        nid = tr['next_id']
                        tr['next_id'] += 1
                        assigned[c['idx']] = nid
                        # Пастельный цвет для нового id
                        base_colors = [
                            (193, 182, 255), (250, 206, 135), (152, 251, 152),
                            (221, 160, 221), (181, 228, 255), (200, 200, 255), (255, 204, 229)
                        ]
                        colors[nid] = base_colors[(nid-1) % len(base_colors)]

                # Обновляем треки (TTL, позиции)
                new_objs = []
                for c in curr:
                    oid = assigned[c['idx']]
                    new_objs.append({'id': oid, 'cx': c['cx'], 'cy': c['cy'], 'ttl': 10})
                # уменьшение ttl у несопоставленных
                for p in obj_prev:
                    if p['id'] not in used_prev:
                        p['ttl'] = p.get('ttl', 1) - 1
                        if p['ttl'] > 0:
                            new_objs.append(p)
                tr['objects'] = new_objs

                # Рисуем маски по стабильным цветам id
                alpha = 0.35
                # копим центроиды для нумерации
                id_centroids = []  # [(oid, cx, cy)]
                for orig_idx, bm in bin_masks:
                    oid = assigned.get(orig_idx)
                    if oid is None:
                        continue
                    color = colors.get(oid, (200, 200, 255))
                    # Мягкое альфа-смешивание без резких границ
                    try:
                        h, w = bm.shape[:2]
                        # Размываем бинарную маску, чтобы получить мягкий край
                        soft = cv2.GaussianBlur((bm * 255).astype(np.uint8), (11, 11), 0).astype(np.float32) / 255.0
                        # Итоговая альфа по пикселю
                        a = (alpha * soft).reshape(h, w, 1)
                        col = np.array(color, dtype=np.float32).reshape(1, 1, 3)
                        f32 = frame.astype(np.float32)
                        frame = (f32 * (1.0 - a) + col * a).astype(np.uint8)
                    except Exception:
                        # fallback на жёсткое смешивание в пределах маски
                        colored = np.zeros_like(frame, dtype=np.uint8)
                        colored[:, :] = color
                        frame[bm == 1] = cv2.addWeighted(frame[bm == 1], 1 - alpha, colored[bm == 1], alpha, 0)
                    # центроид для нумерации
                    try:
                        mu = cv2.moments((bm*255).astype(np.uint8))
                        if mu['m00'] > 1e-3:
                            cx = int(mu['m10']/mu['m00'])
                            cy = int(mu['m01']/mu['m00'])
                            id_centroids.append((oid, cx, cy))
                    except Exception:
                        pass

                # Рисуем номера по порядку слева-направо
                try:
                    id_centroids.sort(key=lambda x: x[1])  # sort by cx
                    for num, (oid, cx, cy) in enumerate(id_centroids, start=1):
                        label = str(num)
                        # обводка для читаемости
                        cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                except Exception:
                    pass
            
            # Общий HUD: Count и Avg Count (EMA), среднее целое, вверх
            avg_int = int(math.ceil(stats['avg_count']))
            cv2.rectangle(frame, (10, 10), (260, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"Count: {int(obj_count)}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Avg: {avg_int}", (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
            
            stats['drawing_ms'] = (time.perf_counter() - drawing_start) * 1000
            
        except Exception as e:
            logger.error(f"Ultra-fast processing error: {e}")
            
        stats['total_ms'] = (time.perf_counter() - start_time) * 1000
        return frame, stats

def create_test_pattern(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Создает тестовое изображение с информацией о системе
    """
    import datetime
    
    # Создаем градиентный фон
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Градиент от темно-синего к светло-синему
    for y in range(height):
        intensity = int(50 + (y / height) * 100)  # От 50 до 150
        frame[y, :] = [intensity, intensity//2, min(255, intensity*2)]
    
    # Добавляем сетку
    grid_size = 50
    for x in range(0, width, grid_size):
        cv2.line(frame, (x, 0), (x, height), (100, 100, 100), 1)
    for y in range(0, height, grid_size):
        cv2.line(frame, (0, y), (width, y), (100, 100, 100), 1)
    
    # Центральный круг
    center = (width // 2, height // 2)
    cv2.circle(frame, center, 80, (0, 255, 255), 3)
    cv2.circle(frame, center, 60, (255, 255, 0), 2)
    cv2.circle(frame, center, 40, (255, 0, 255), 2)
    
    # Текстовая информация
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Заголовок
    title = "PigWeight Ultra-Fast Test Pattern"
    cv2.putText(frame, title, (width//2 - 200, 50), font, 0.8, (255, 255, 255), 2)
    
    # Время
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", (50, height - 80), font, 0.6, (255, 255, 255), 1)
    
    # Статус
    status_text = "No Video Sources Available - Using Test Pattern"
    cv2.putText(frame, status_text, (50, height - 50), font, 0.5, (255, 200, 100), 1)
    
    # Разрешение
    res_text = f"Resolution: {width}x{height}"
    cv2.putText(frame, res_text, (50, height - 20), font, 0.4, (200, 200, 200), 1)
    
    return frame

# Глобальный экземпляр процессора
ultra_processor = UltraFastProcessor()

async def ws_video_ultra_fast(websocket: WebSocket):
    """
    Ультра-быстрый WebSocket для видео потока с fallback механизмами
    """
    await websocket.accept()
    
    cap = None
    
    try:
        # Инициализируем модель
        model_path = "models/pig_yolo11-seg.pt"
        await ultra_processor.init_model(model_path)
        
        # Пробуем различные источники видео
        video_sources = [
            "rtsp://admin:Qwerty.123@10.15.6.24/1/1",  # Основная камера
            "rtsp://admin:Qwerty.123@10.15.6.24/1",     # Альтернативный URL
            0,  # Веб-камера по умолчанию
            1,  # Альтернативная веб-камера
        ]
        
        cap = None
        active_source = None
        
        for source in video_sources:
            logger.info(f"Trying video source: {source}")
            
            try:
                cap = cv2.VideoCapture(source)
                
                if isinstance(source, str):  # RTSP источник
                    # Короткий таймаут для RTSP
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 секунд
                    cap.set(cv2.CAP_PROP_FPS, 15)  # Умеренный FPS для стабильности
                else:  # Веб-камера
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Тестируем чтение кадра
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    active_source = source
                    logger.info(f"Successfully connected to: {source}")
                    break
                else:
                    cap.release()
                    cap = None
                    
            except Exception as e:
                logger.warning(f"Failed to connect to {source}: {e}")
                if cap:
                    cap.release()
                cap = None
                continue
        
        if cap is None:
            # Fallback к тестовому изображению
            logger.warning("No video sources available, using test pattern")
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "No video sources available. Using test pattern."
            }))
            
            # Генерируем тестовое изображение
            test_frame = create_test_pattern()
            processed_frame, stats = await ultra_processor.process_frame_ultra_fast(
                test_frame, "test_pattern"
            )
            frame_data = ultra_processor.encode_raw_frame(processed_frame)
            
            while True:
                try:
                    await websocket.send_bytes(frame_data)
                    await asyncio.sleep(1.0)  # 1 FPS для тестового паттерна
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Test pattern error: {e}")
                    break
            return
        
        # Основной цикл обработки видео
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        consecutive_errors = 0
        max_errors = 10
        
        while True:
            try:
                # Чтение кадра
                # Применяем отложенный seek, если он установлен через HTTP
                seek_to = session.pop('seek_to_frame', None)
                if isinstance(seek_to, int) and seek_to >= 0:
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                    except Exception:
                        pass

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    
                    if consecutive_errors > max_errors:
                        logger.error(f"Too many consecutive errors ({consecutive_errors}), reconnecting...")
                        cap.release()
                        
                        # Попытка переподключения
                        for retry_source in [active_source] + [s for s in video_sources if s != active_source]:
                            try:
                                cap = cv2.VideoCapture(retry_source)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                
                                ret, frame = cap.read()
                                if ret and frame is not None:
                                    active_source = retry_source
                                    consecutive_errors = 0
                                    logger.info(f"Reconnected to: {retry_source}")
                                    break
                                else:
                                    cap.release()
                            except Exception as e:
                                logger.warning(f"Reconnection to {retry_source} failed: {e}")
                                continue
                        
                        if consecutive_errors > 0:
                            await asyncio.sleep(1)
                            continue
                    else:
                        await asyncio.sleep(0.1)
                        continue
                
                consecutive_errors = 0
                
                # Ультра-быстрая обработка
                processed_frame, stats = await ultra_processor.process_frame_ultra_fast(
                    frame, "live_camera"
                )
                
                # Кодирование и отправка
                frame_data = ultra_processor.encode_raw_frame(processed_frame)
                if frame_data:
                    await websocket.send_bytes(frame_data)
                
                # FPS контроль и статистика
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    actual_fps = fps_counter / (current_time - last_fps_time)
                    logger.info(f"Ultra-fast FPS: {actual_fps:.1f} (source: {active_source})")
                    fps_counter = 0
                    last_fps_time = current_time
                
                # Динамическая задержка на основе производительности
                target_delay = 1.0 / ultra_processor.target_fps
                if stats['total_ms'] > 0:
                    processing_delay = stats['total_ms'] / 1000.0
                    actual_delay = max(0.01, target_delay - processing_delay)
                else:
                    actual_delay = target_delay
                    
                await asyncio.sleep(actual_delay)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket video error: {e}")
                consecutive_errors += 1
                
                if consecutive_errors > max_errors:
                    break
                    
                await asyncio.sleep(0.1)
                continue
                
    except Exception as e:
        logger.exception(f"Ultra-fast video stream error: {e}")
    finally:
        if cap:
            cap.release()
        try:
            await websocket.close()
        except Exception:
            pass

async def ws_video_file_ultra_fast(websocket: WebSocket, id: str = Query(...)):
    """
    Ультра-быстрый WebSocket для файлового видео
    """
    await websocket.accept()

    try:
        # Получаем сессию файла
        session = ultra_processor.file_sessions.get(id)
        if not session:
            await websocket.close(code=1000, reason="Session not found")
            return

        cap = session.get('cap')
        if not cap:
            await websocket.close(code=1000, reason="Video capture not available")
            return

        # Инициализируем модель
        model_path = "models/pig_yolo11-seg.pt"
        await ultra_processor.init_model(model_path)

        fps = session.get('fps', 25.0)
        frame_delay = 1.0 / max(1e-3, fps)

        # Очереди latest-only (размер = 1)
        frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        processed_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        # Метрики окна
        frame_counter = 0
        sum_read = sum_proc = sum_enc = sum_send = 0.0
        window_start = time.perf_counter()

        async def latest_put(q: asyncio.Queue, item):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(item)
                except Exception:
                    pass

        # Producer: apply seek and read with retries
        async def producer():
            consecutive_errors = 0
            while True:
                t_read0 = time.perf_counter()
                try:
                    lock = session.get('lock')
                    if lock is None:
                        session['lock'] = asyncio.Lock()
                        lock = session['lock']
                    async with lock:
                        seek_to = session.pop('seek_to_frame', None)
                        if isinstance(seek_to, int) and seek_to >= 0:
                            try:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_to))
                                logger.debug(f"[WS_SEEK] Applied seek_to_frame={seek_to} for session {id}")
                            except Exception:
                                pass
                        ret, frame = cap.read()
                        # Прогрев после seek: на некоторых контейнерах первый read пустой
                        if (not ret or frame is None):
                            for _ in range(2):
                                await asyncio.sleep(0)
                                ret, frame = cap.read()
                                if ret and frame is not None:
                                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(f"[WS_READ] error: {e}")
                    ret, frame = False, None

                t_read1 = time.perf_counter()
                nonlocal sum_read
                sum_read += (t_read1 - t_read0)

                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        # попробуем отскочить на начало
                        try:
                            async with session.get('lock'):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except Exception:
                            pass
                        await asyncio.sleep(0.02)
                    else:
                        await asyncio.sleep(0.005)
                    continue

                consecutive_errors = 0
                await latest_put(frame_queue, (time.perf_counter(), frame))

                # Пейсинг продюсера: читаем быстрее FPS, но без захлёба
                await asyncio.sleep(0)

        # Processor: inference and count notifications
        async def processor():
            nonlocal sum_proc
            while True:
                ts, frame = await frame_queue.get()
                t_proc0 = time.perf_counter()
                try:
                    processed_frame, stats = await ultra_processor.process_frame_ultra_fast(frame, id)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[WS_PROC] error: {e}")
                    continue
                t_proc1 = time.perf_counter()
                sum_proc += (t_proc1 - t_proc0)

                # Обновление статистики сессии и немедленная нотификация счётчика
                session['last_stats'] = stats
                session['last_count'] = stats.get('count', 0)
                try:
                    session['avg_count'] = float(stats.get('avg_count', 0.0) or 0.0)
                except Exception:
                    session['avg_count'] = 0.0
                logger.debug(f"[WS_COUNT] Updated session {id}: count={session['last_count']}, avg={session['avg_count']}")
                try:
                    await ultra_processor._notify_count_websockets(id, session['last_count'], session.get('avg_count', 0.0))
                except Exception as e:
                    logger.debug(f"[WS_COUNT] notify error: {e}")

                await latest_put(processed_queue, (time.perf_counter(), processed_frame))

        # Sender: encode and send with pacing
        async def sender():
            nonlocal frame_counter, sum_enc, sum_send, window_start
            next_send_ts = time.perf_counter()
            while True:
                ts, processed_frame = await processed_queue.get()
                t_enc0 = time.perf_counter()
                frame_data = b''
                try:
                    frame_data = ultra_processor.encode_raw_frame(processed_frame)
                except Exception:
                    frame_data = b''
                t_enc1 = time.perf_counter()
                sum_enc += (t_enc1 - t_enc0)

                t_send0 = time.perf_counter()
                if frame_data:
                    await websocket.send_bytes(frame_data)
                t_send1 = time.perf_counter()
                sum_send += (t_send1 - t_send0)

                # Пейсинг под целевой FPS
                now = time.perf_counter()
                if now < next_send_ts:
                    await asyncio.sleep(max(0.0, next_send_ts - now))
                next_send_ts = max(now, next_send_ts) + frame_delay

                # Метрики окна
                frame_counter += 1
                if frame_counter % 30 == 0:
                    dt = now - window_start
                    eff_fps = (frame_counter / dt) if dt > 0 else 0.0
                    try:
                        from api.app import perf_logger as _perf
                    except Exception:
                        _perf = None
                    if _perf:
                        _perf.info(
                            f"ultra_file_ws id={id} fps={eff_fps:.1f} "
                            f"read={sum_read/frame_counter*1000:.1f}ms proc={sum_proc/frame_counter*1000:.1f}ms "
                            f"enc={sum_enc/frame_counter*1000:.1f}ms send={sum_send/frame_counter*1000:.1f}ms"
                        )
                    frame_counter = 0
                    sum_read = sum_proc = sum_enc = sum_send = 0.0
                    window_start = now

        # Запуск задач конвейера
        producer_task = asyncio.create_task(producer())
        processor_task = asyncio.create_task(processor())
        sender_task = asyncio.create_task(sender())

        # Ожидание завершения по разрыву сокета
        await sender_task

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"File WebSocket error: {e}")
    finally:
        # Отмена задач и закрытие WS
        try:
            for t in [locals().get('producer_task'), locals().get('processor_task')]:
                if t and not t.done():
                    t.cancel()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass

async def api_video_file_seek_ultra_fast(
    id: str = Query("ultra_fast_session"),
    t: float = Query(0.0)
):
    """
    Быстрый seek без остановки WS: помечаем кадр к которому нужно прыгнуть.
    WS-петля применит его перед следующим чтением.
    """
    try:
        session = ultra_processor.file_sessions.get(id)
        if not session:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        fps = session.get('fps', 25.0)
        frame_number = int(max(0.0, t) * max(1.0, fps))
        session['seek_to_frame'] = frame_number
        return JSONResponse({"ok": True, "frame": frame_number})
    except Exception as e:
        logger.error(f"Seek ultra-fast error: {e}")
        return JSONResponse({"error": "Seek failed"}, status_code=500)

async def api_video_file_frame_ultra_fast(
    id: str = Query("ultra_fast_session"),
    t: float = Query(0.0),
    ts: int = Query(0)
):
    """
    Ультра-быстрое получение кадра из файла
    """
    try:
        session = ultra_processor.file_sessions.get(id)
        if not session:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        
        # 1) PyAV быстрый путь: мгновенный JPEG без инференса
        if _PYAV_OK:
            try:
                file_path = session.get('file_path') or ""
                q = int(os.getenv("JPEG_QUALITY", "80")) if 'os' in globals() else 80
                t0 = time.perf_counter()
                jpeg_bytes = pyav_seek_read_jpeg(file_path, float(t), q)
                dt_ms = int((time.perf_counter() - t0) * 1000)
                if jpeg_bytes:
                    return Response(jpeg_bytes, media_type="image/jpeg", headers={"X-Fast": "pyav", "X-Seek-Ms": str(dt_ms)})
            except Exception as e:
                logger.debug(f"PyAV fast seek failed, fallback to OpenCV: {e}")

        # 2) Fallback: текущая логика OpenCV + инференс
        cap = session.get('cap')
        if not cap:
            return JSONResponse({"error": "Video capture not available"}, status_code=400)
        fps = session.get('fps', 25.0)
        # Нормализуем время и кадр в допустимый диапазон
        try:
            total_frames = int(session.get('frame_count') or 0)
        except Exception:
            total_frames = 0
        frame_number = int(max(0.0, float(t)) * max(1.0, float(fps)))
        if total_frames > 0:
            frame_number = max(0, min(frame_number, max(0, total_frames - 1)))
        seek_start = time.perf_counter()
        # Безопасный seek+read под сессионным lock
        lock = session.get('lock')
        if lock is None:
            session['lock'] = asyncio.Lock()
            lock = session['lock']
        async with lock:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            # На некоторых контейнерах сразу после seek первый read возвращает False
            if not ret or frame is None:
                # Короткая задержка и повторные попытки
                for _ in range(2):
                    await asyncio.sleep(0)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
        seek_time = (time.perf_counter() - seek_start) * 1000
        if not ret or frame is None:
            # 2.1) Альтернативный seek по миллисекундам
            try:
                async with lock:
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # иногда нужно «снять» пару кадров после seek
                        for _ in range(2):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                break
            except Exception:
                pass
        
        if not ret or frame is None:
            # 2.2) Переоткрыть файл и попробовать ещё раз (одна попытка)
            try:
                src_path = session.get('file_path')
                if src_path:
                    new_cap = cv2.VideoCapture(str(src_path))
                    if new_cap.isOpened():
                        # заменить cap в сессии
                        old = session.get('cap')
                        session['cap'] = new_cap
                        try:
                            if old is not None:
                                old.release()
                        except Exception:
                            pass
                        cap = new_cap
                        async with lock:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                            ret, frame = cap.read()
            except Exception:
                pass

        if not ret or frame is None:
            # 2.3) Попробуем получить кадр через PyAV напрямую
            if _PYAV_OK:
                try:
                    file_path = session.get('file_path') or ""
                    q = int(os.getenv("JPEG_QUALITY", "80")) if 'os' in globals() else 80
                    t0 = time.perf_counter()
                    jpeg_bytes = pyav_seek_read_jpeg(file_path, float(t), q)
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    if jpeg_bytes:
                        return Response(jpeg_bytes, media_type="image/jpeg", headers={"X-Fast": "pyav-fallback", "X-Seek-Ms": str(dt_ms)})
                except Exception as e:
                    logger.debug(f"PyAV fallback after OpenCV read fail also failed: {e}")
            return JSONResponse({"error": "Failed to read frame", "details": {"t": t, "frame": frame_number}}, status_code=500)
        # Инициализация модели и обработка как раньше
        model_path = "models/pig_yolo11-seg.pt"
        await ultra_processor.init_model(model_path)
        processed_frame, stats = await ultra_processor.process_frame_ultra_fast(frame, id)
        stats['seek_ms'] = seek_time
        encode_start = time.perf_counter()
        frame_data = ultra_processor.encode_raw_frame(processed_frame)
        encode_time = (time.perf_counter() - encode_start) * 1000
        stats['encode_ms'] = encode_time
        headers = {
            "X-Seek-Ms": str(int(seek_time)),
            "X-Inference-Ms": str(int(stats['inference_ms'])),
            "X-Encode-Ms": str(int(encode_time)),
            "X-Total-Ms": str(int(stats['total_ms']))
        }
        return Response(frame_data, media_type="image/bmp", headers=headers)
        
    except Exception as e:
        logger.error(f"Ultra-fast frame error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_video_file_open_ultra_fast(
    camera: str = Form("file_cam"),
    id: str = Form("ultra_fast_session"),
    file: UploadFile = File(...)
):
    """
    Ультра-быстрое открытие видеофайла
    """
    try:
        # Закрываем предыдущую сессию
        if id in ultra_processor.file_sessions:
            old_session = ultra_processor.file_sessions[id]
            if 'cap' in old_session:
                old_session['cap'].release()
        
        # Сохраняем файл
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Открываем видео
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return JSONResponse({"error": "Failed to open video file"}, status_code=400)
            
        # Оптимизированные настройки
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Получаем метаданные
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0
        
        # Сохраняем сессию
        ultra_processor.file_sessions[id] = {
            'cap': cap,
            'file_path': str(file_path),
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'last_count': 0,
            'last_stats': {},
            # Сессионный lock для безопасного доступа к cap между WS и HTTP
            'lock': asyncio.Lock()
        }
        
        logger.info(f"Ultra-fast file opened: {file.filename}, duration: {duration:.1f}s")
        
        return JSONResponse({
            "id": id,
            "camera": camera,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "status": "ready"
        })
        
    except Exception as e:
        logger.error(f"Ultra-fast file open error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Добавляем эндпоинты к основному приложению
def add_ultra_fast_endpoints(app):
    """
    Добавляет ультра-быстрые эндпоинты к приложению FastAPI
    """
    
    @app.websocket("/ws/video_ultra_fast")
    async def websocket_video_ultra_fast(websocket: WebSocket):
        await ws_video_ultra_fast(websocket)
    
    @app.websocket("/ws/video_file_ultra_fast")
    async def websocket_video_file_ultra_fast(websocket: WebSocket, id: str = Query(...)):
        await ws_video_file_ultra_fast(websocket, id)
    
    @app.get("/api/video_file/frame_ultra_fast")
    async def get_video_file_frame_ultra_fast(
        id: str = Query("ultra_fast_session"),
        t: float = Query(0.0),
        ts: int = Query(0)
    ):
        return await api_video_file_frame_ultra_fast(id, t, ts)
    
    @app.get("/api/video_file/seek_ultra_fast")
    async def get_video_file_seek_ultra_fast(
        id: str = Query("ultra_fast_session"),
        t: float = Query(0.0)
    ):
        return await api_video_file_seek_ultra_fast(id, t)
    
    @app.post("/api/video_file/open_ultra_fast")
    async def post_video_file_open_ultra_fast(
        camera: str = Form("file_cam"),
        id: str = Form("ultra_fast_session"),
        file: UploadFile = File(...)
    ):
        return await api_video_file_open_ultra_fast(camera, id, file)
