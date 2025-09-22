import os
import logging
import time
from typing import Any, Dict, List
import numpy as np
from datetime import datetime

# Import type safety utilities
from core.type_utils import TypeSafetyManager, safe_tensor_conversion, ensure_float32, validate_tensor_compatibility

try:
    import onnxruntime as ort
    _HAVE_ONNX = True
    print("ONNX Runtime доступен для CPU inference")
except Exception as e:
    _HAVE_ONNX = False
    print(f"ONNX Runtime не доступен: {e}")
    print("   Для CPU оптимизации установите: pip install onnxruntime")

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

try:
    from ultralytics import YOLO
    _HAVE_ULTRALYTICS = True
except Exception:
    _HAVE_ULTRALYTICS = False

logger = logging.getLogger("services.model_adapter")
perf_logger = logging.getLogger("performance")


class ModelAdapter:
    """Adapter: prefer ultralytics (.pt), then ONNX, then TorchScript.

    Loads model according to file extension and exposes infer(imgs) -> list[dict].
    """

    def __init__(self, model_path: str, device: str = None):
        # Use 'auto' as the default device selection strategy
        temp_device = device or os.getenv('DEVICE', 'auto')
        self.backend = None
        self._sess = None  # ONNX Runtime session
        self._torch_model = None
        self._yolo = None

        # Performance tracking
        self._inference_times = []
        self._total_inferences = 0
        self._type_conversion_times = []
        self._dtype_performance_stats = {}  # Track performance by data type
        self._compatibility_checks = 0
        self._compatibility_failures = 0

        # Auto-detect optimal configuration
        self.device, self.use_half = self._detect_optimal_device(temp_device)
        self.model_path = self._select_best_model(model_path)
        
        # Initialize type safety manager
        self.type_manager = TypeSafetyManager(self.device)
        
        # Determine optimal inference dtype based on device and model
        self.optimal_dtype = self._determine_optimal_dtype()
        
        # Type compatibility cache for performance
        self._compatibility_cache = {}
        
        perf_logger = logging.getLogger("perf.model_adapter")

        # Priority: ONNX (CPU) > Ultralytics (GPU/CPU) > TorchScript
        if self.model_path and self.model_path.endswith('.onnx') and _HAVE_ONNX:
            # ONNX model - optimized for CPU inference
            perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Loading ONNX model from {self.model_path} for CPU inference")
            try:
                logger.info(f"Loading ONNX model from {self.model_path} for CPU inference")
                # Use CPU execution provider for maximum performance
                self._sess = ort.InferenceSession(
                    self.model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=ort.SessionOptions()
                )
                self.backend = 'onnx'
                logger.info(f"✅ ONNX model loaded successfully")
                perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] ONNX backend initialized for CPU inference")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model {self.model_path}: {e}")
                self._sess = None

        elif self.model_path and self.model_path.endswith('.pt') and _HAVE_ULTRALYTICS:
            perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Loading Ultralytics model from {self.model_path} on device={self.device}")
            try:
                logger.info(f"Loading ultralytics model from {self.model_path} on device={self.device}")
                self._yolo = YOLO(self.model_path)
                # try device/half settings
                try:
                    if self.device:
                        self._yolo.to(self.device)
                except Exception:
                    pass
                try:
                    # Only call .half() when CUDA is actually available. Some environments
                    # may set DEVICE to 'cuda' but torch.cuda.is_available() can be false
                    # (e.g., wrong drivers), which causes dtype mismatch errors.
                    use_half_env = (os.getenv('USE_HALF', 'true').lower() == 'true')
                    can_half = False
                    try:
                        import torch as _t
                        can_half = use_half_env and self.device.startswith('cuda') and getattr(_t, 'cuda', None) is not None and _t.cuda.is_available()
                    except Exception:
                        can_half = False
                    if can_half and hasattr(self._yolo, 'model'):
                        try:
                            # Принимаем half precision модели как есть, не форсируем изменения
                            logger.info('Using model native precision (likely half precision)')
                        except Exception:
                            logger.warning('Failed to configure model precision; using defaults')
                except Exception:
                    pass
                self.backend = 'ultralytics'
            except Exception as e:
                logger.warning(f"Failed to load ultralytics model {self.model_path}: {e}")
                self._yolo = None
        # ensure model on correct device and half if possible
        if self._yolo is not None:
            try:
                if self.device and hasattr(self._yolo, 'to'):
                    self._yolo.to(self.device)
            except Exception:
                pass

        if not self._yolo and _HAVE_ONNX and self.model_path.endswith('.onnx'):
            try:
                logger.info(f"Loading ONNX model from {self.model_path}")
                self._sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                self.backend = 'onnx'
            except Exception:
                logger.warning(f"Failed to load ONNX model {self.model_path}")
                self._sess = None

        if not self._yolo and not self._sess and _HAVE_TORCH and self.model_path.endswith('.pt'):
            try:
                # try jit first
                try:
                    self._torch_model = torch.jit.load(self.model_path)
                except Exception:
                    self._torch_model = torch.load(self.model_path)
                if self.device.startswith('cuda') and torch.cuda.is_available():
                    try:
                        self._torch_model.to(self.device)
                    except Exception:
                        pass
                self._torch_model.eval()
                self.backend = 'torch'
            except Exception as e:
                logger.warning(f"Failed to load Torch model {self.model_path}: {e}")
                self._torch_model = None

        logger.info(f"ModelAdapter initialized: path={self.model_path}, backend={self.backend}, device={self.device}")
        
        # Log type safety configuration
        if hasattr(self, 'type_manager'):
            type_stats = self.type_manager.get_stats()
            logger.info(f"Type safety manager initialized: {type_stats}")
            logger.info(f"Optimal inference dtype: {self.optimal_dtype}")

    def _determine_optimal_dtype(self):
        """Определяет оптимальный тип данных для инференса на основе устройства и модели"""
        try:
            if self.device.startswith('cuda') and _HAVE_TORCH:
                import torch
                if torch.cuda.is_available():
                    # Для GPU предпочитаем half precision для экономии памяти
                    if self.use_half:
                        return torch.float16
                    else:
                        return torch.float32
                else:
                    # Fallback to CPU
                    return torch.float32
            else:
                # Для CPU всегда используем float32 для стабильности
                if _HAVE_TORCH:
                    import torch
                    return torch.float32
                else:
                    return np.float32
        except Exception as e:
            logger.warning(f"Failed to determine optimal dtype: {e}")
            return np.float32

    def _check_tensor_compatibility(self, tensor1, tensor2, cache_key: str = None) -> bool:
        """Проверяет совместимость типов тензоров с кэшированием"""
        if cache_key and cache_key in self._compatibility_cache:
            return self._compatibility_cache[cache_key]
        
        self._compatibility_checks += 1
        start_time = time.time()
        
        try:
            is_compatible = validate_tensor_compatibility(tensor1, tensor2)
            if cache_key:
                self._compatibility_cache[cache_key] = is_compatible
            
            # Track performance
            check_time = time.time() - start_time
            if not hasattr(self, '_compatibility_check_times'):
                self._compatibility_check_times = []
            self._compatibility_check_times.append(check_time)
            
            if not is_compatible:
                self._compatibility_failures += 1
                
            return is_compatible
        except Exception as e:
            logger.warning(f"Compatibility check failed: {e}")
            self._compatibility_failures += 1
            return False

    def _track_dtype_performance(self, dtype_str: str, inference_time: float, batch_size: int):
        """Отслеживает производительность для различных типов данных"""
        if dtype_str not in self._dtype_performance_stats:
            self._dtype_performance_stats[dtype_str] = {
                'total_time': 0.0,
                'total_inferences': 0,
                'total_samples': 0,
                'avg_time_per_sample': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        stats = self._dtype_performance_stats[dtype_str]
        stats['total_time'] += inference_time
        stats['total_inferences'] += 1
        stats['total_samples'] += batch_size
        stats['avg_time_per_sample'] = stats['total_time'] / stats['total_samples']
        stats['min_time'] = min(stats['min_time'], inference_time)
        stats['max_time'] = max(stats['max_time'], inference_time)

    def _handle_inference_error(self, error: Exception, context: str = "") -> None:
        """Handle and log inference errors with context"""
        error_msg = str(error).lower()
        
        if "dtype" in error_msg or "half" in error_msg or "float" in error_msg:
            logger.error(f"Type conversion error in {context}: {error}")
            logger.info("This error is related to tensor type mismatches (c10::Half vs float)")
            logger.info("The ModelAdapter will attempt automatic type conversion")
        elif "cuda" in error_msg or "device" in error_msg:
            logger.error(f"Device error in {context}: {error}")
            logger.info("This error is related to GPU/CPU device mismatches")
        else:
            logger.error(f"General inference error in {context}: {error}")
    
    def get_type_stats(self) -> dict:
        """Get type safety statistics"""
        if hasattr(self, 'type_manager'):
            return self.type_manager.get_stats()
        return {"type_manager": "not_initialized"}
    
    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics"""
        stats = {
            "total_inferences": self._total_inferences,
            "backend": self.backend,
            "device": self.device,
            "optimal_dtype": str(self.optimal_dtype),
            "use_half": self.use_half
        }
        
        # Inference timing stats
        if self._inference_times:
            stats["inference_timing"] = {
                "avg_time": sum(self._inference_times) / len(self._inference_times),
                "min_time": min(self._inference_times),
                "max_time": max(self._inference_times),
                "total_samples": len(self._inference_times)
            }
        
        # Type conversion stats
        if self._type_conversion_times:
            stats["type_conversion"] = {
                "avg_time": sum(self._type_conversion_times) / len(self._type_conversion_times),
                "total_conversions": len(self._type_conversion_times),
                "total_time": sum(self._type_conversion_times)
            }
        
        # Compatibility check stats
        stats["compatibility_checks"] = {
            "total_checks": self._compatibility_checks,
            "failures": self._compatibility_failures,
            "success_rate": (self._compatibility_checks - self._compatibility_failures) / max(1, self._compatibility_checks),
            "cache_size": len(self._compatibility_cache)
        }
        
        # Per-dtype performance stats
        stats["dtype_performance"] = self._dtype_performance_stats.copy()
        
        # Type manager stats
        if hasattr(self, 'type_manager'):
            stats["type_manager"] = self.type_manager.get_stats()
        
        return stats
    
    def reset_performance_stats(self):
        """Reset all performance statistics"""
        self._inference_times.clear()
        self._type_conversion_times.clear()
        self._dtype_performance_stats.clear()
        self._compatibility_cache.clear()
        self._total_inferences = 0
        self._compatibility_checks = 0
        self._compatibility_failures = 0
        
        if hasattr(self, 'type_manager'):
            self.type_manager.clear_cache()
        
        logger.info("Performance statistics reset")
    
    def optimize_for_device(self):
        """Optimize model settings for current device"""
        try:
            if self.backend == 'ultralytics' and self._yolo is not None:
                # Re-evaluate optimal settings
                old_dtype = self.optimal_dtype
                self.optimal_dtype = self._determine_optimal_dtype()
                
                if old_dtype != self.optimal_dtype:
                    logger.info(f"Updated optimal dtype: {old_dtype} -> {self.optimal_dtype}")
                
                # Update type manager
                if hasattr(self, 'type_manager'):
                    self.type_manager.optimal_dtype = self.optimal_dtype
                
                # Try to optimize model for current device
                if hasattr(self._yolo, 'to') and self.device:
                    self._yolo.to(self.device)
                    logger.info(f"Model moved to device: {self.device}")
                
                return True
        except Exception as e:
            logger.warning(f"Device optimization failed: {e}")
            return False

    def _detect_optimal_device(self, requested_device: str = "auto"):
        """Автоматически определяет оптимальное устройство и настройки"""
        use_half = os.getenv('USE_HALF', 'true').lower() == 'true'
        
        # Проверяем доступность CUDA
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        # Логика автовыбора
        if requested_device == "auto":
            if cuda_available:
                logger.info("✅ CUDA доступна, выбран автоматический режим: cuda:0")
                return 'cuda:0', use_half
            else:
                logger.info("📱 CUDA недоступна, выбран автоматический режим: cpu")
                return 'cpu', False
        
        # Логика для явного запроса устройства
        if requested_device.startswith('cuda') and cuda_available:
            logger.info(f"✅ CUDA доступна, используем: {requested_device}")
            return requested_device, use_half
        elif requested_device.startswith('cuda') and not cuda_available:
            logger.warning(f"⚠️ CUDA недоступна, переключаемся на CPU")
            return 'cpu', False
        else:
            logger.info(f"📱 Используем CPU")
            return 'cpu', False
    
    def _select_best_model(self, model_path: str):
        """Выбирает лучшую доступную модель (ONNX для CPU, PT для GPU)"""
        base_path = model_path.rsplit('.', 1)[0]  # Убираем расширение
        
        # Список приоритетов в зависимости от устройства
        if self.device.startswith('cuda'):
            # Для GPU: предпочитаем PyTorch, потом ONNX
            candidates = [
                f"{base_path}.pt",
                f"{base_path}.onnx",
                model_path  # Исходный путь как fallback
            ]
        else:
            # Для CPU: предпочитаем ONNX, потом PyTorch
            candidates = [
                f"{base_path}.onnx",
                f"{base_path}.pt", 
                model_path  # Исходный путь как fallback
            ]
        
        # Ищем первую доступную модель
        for candidate in candidates:
            if os.path.exists(candidate):
                if candidate != model_path:
                    logger.info(f"🔄 Автоматически выбрана модель: {candidate} (вместо {model_path})")
                return candidate
        
        # Если ничего не найдено, возвращаем исходный путь
        logger.warning(f"⚠️ Оптимальная модель не найдена, используем: {model_path}")
        return model_path

    def infer(self, imgs: List[np.ndarray]) -> List[Dict[str, Any]]:
        import time
        inference_start = time.time()
        batch_size = len(imgs)
        
        # Pre-inference type compatibility checks and conversions
        conversion_start = time.time()
        processed_imgs = []
        input_dtype_str = "unknown"
        
        try:
            for i, img in enumerate(imgs):
                if img is None:
                    processed_imgs.append(img)
                    continue
                
                # Track input data type
                if hasattr(img, 'dtype'):
                    input_dtype_str = str(img.dtype)
                
                # Apply type safety conversions
                safe_img = self.type_manager.prepare_tensor(img, self.optimal_dtype)
                processed_imgs.append(safe_img)
                
                # Log type conversion if it occurred
                if safe_img is not img:
                    logger.debug(f"Applied type conversion for image {i}: {type(img)} -> {type(safe_img)}")
            
            conversion_time = time.time() - conversion_start
            self._type_conversion_times.append(conversion_time)
            
        except Exception as e:
            logger.warning(f"Type conversion failed, using original images: {e}")
            processed_imgs = imgs
            conversion_time = time.time() - conversion_start

        out: List[Dict[str, Any]] = []

        # ONNX inference (CPU optimized)
        if self.backend == 'onnx' and self._sess is not None:
            try:
                perf_logger = logging.getLogger("perf.model_adapter")
                perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Running ONNX inference on {len(imgs)} images")

                # Track inference performance
                inference_start = time.time()

                # Get input details
                input_name = self._sess.get_inputs()[0].name
                input_shape = self._sess.get_inputs()[0].shape
                # Parse YOLO outputs (simplified - adapt based on your model)
                # This is a complex parsing logic that depends on the exact output format of your YOLOv11 ONNX model.
                # The following is a generic placeholder for segmentation models. You MUST adapt it to your model's output signature.
                results = []
                for img in imgs:
                    if img is None:
                        results.append({'detections': 0, 'confidence': 0.0, 'masks': []})
                        continue
                    
                    try:
                        # Preprocess for ONNX
                        import cv2
                        input_shape = self._sess.get_inputs()[0].shape
                        target_h, target_w = input_shape[2], input_shape[3]
                        
                        # Letterbox resize (common for ONNX models)
                        h, w = img.shape[:2]
                        scale = min(target_w / w, target_h / h)
                        unpad_w, unpad_h = int(round(w * scale)), int(round(h * scale))
                        pad_x, pad_y = (target_w - unpad_w) // 2, (target_h - unpad_h) // 2
                        
                        if (h, w) != (unpad_h, unpad_w):
                            img_resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
                        else:
                            img_resized = img

                        padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
                        padded_img[pad_y:pad_y+unpad_h, pad_x:pad_x+unpad_w] = img_resized
                        
                        # HWC to CHW, normalize
                        img_tensor = np.transpose(padded_img, (2, 0, 1)).astype(np.float32) / 255.0
                        img_tensor = np.expand_dims(img_tensor, axis=0)

                        # Run inference
                        outputs = self._sess.run(None, {self._sess.get_inputs()[0].name: img_tensor})
                        
                        # --- OUTPUT PARSING ---
                        # This part is highly model-specific.
                        # Assuming outputs[0] is boxes [batch, num_dets, 5+num_classes]
                        # and outputs[1] is masks [batch, num_dets, mask_h, mask_w]
                        
                        pred = outputs[0][0] # First image in batch
                        
                        # Filter by confidence
                        conf_mask = pred[:, 4] > 0.30 # Using CONF_THRESHOLD
                        pred = pred[conf_mask]
                        
                        if len(pred) == 0:
                            results.append({'detections': 0, 'confidence': 0.0, 'masks': []})
                            continue

                        # Extract boxes, scores, and class IDs
                        boxes = pred[:, :4]
                        scores = pred[:, 4]
                        
                        # Convert boxes from xywh to xyxy
                        boxes[:, 0] -= boxes[:, 2] / 2
                        boxes[:, 1] -= boxes[:, 3] / 2
                        boxes[:, 2] += boxes[:, 0]
                        boxes[:, 3] += boxes[:, 1]
                        
                        # Scale boxes back to original image size
                        boxes[:, [0, 2]] -= pad_x
                        boxes[:, [1, 3]] -= pad_y
                        boxes[:, :4] /= scale
                        
                        # Clip boxes
                        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
                        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

                        # Simplified result
                        results.append({
                            'detections': len(boxes),
                            'confidence': float(np.mean(scores)) if len(scores) > 0 else 0.0,
                            'masks': [], # Mask parsing is complex and omitted for this fix
                            'bboxes': boxes.tolist(), # Добавляем bboxes
                            'centroids': [] # Центроиды пока не вычисляем
                        })

                    except Exception as e:
                        logger.warning(f"ONNX inference error for image: {e}")
                        results.append({'detections': 0, 'confidence': 0.0, 'masks': [], 'bboxes': [], 'centroids': []})

                out = results
                
                # Track performance metrics for ONNX
                total_inference_time = time.time() - inference_start
                self._inference_times.append(total_inference_time)
                self._total_inferences += 1
                self._track_dtype_performance(input_dtype_str, total_inference_time, batch_size)
                
                return out

            except Exception as e:
                logger.error(f"ONNX inference error: {e}", exc_info=True)
                return [{'detections': 0, 'confidence': 0.0, 'masks': [], 'bboxes': [], 'centroids': []} for _ in imgs]

        # Ultralytics inference (GPU/CPU)
        elif self.backend == 'ultralytics' and self._yolo is not None:
            try:
                perf_logger = logging.getLogger("perf.model_adapter")
                perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Running Ultralytics inference on {len(imgs)} images")
                imgsz = int(os.getenv('IMG_SIZE', '960'))
                conf = float(os.getenv('CONF_THRESHOLD', '0.30'))
                
                # Apply type safety before inference
                processed_imgs = []
                for img in imgs:
                    try:
                        # Ensure consistent data types
                        safe_img = ensure_float32(img) if hasattr(img, 'dtype') else img
                        processed_imgs.append(safe_img)
                    except Exception as e:
                        logger.warning(f"Type conversion failed for image: {e}")
                        processed_imgs.append(img)
                
                # Run inference with type safety
                try:
                    if self.use_half:
                        results = self._yolo.predict(processed_imgs, imgsz=imgsz, conf=conf, verbose=False, retina_masks=True, half=True)
                    else:
                        results = self._yolo.predict(processed_imgs, imgsz=imgsz, conf=conf, verbose=False, retina_masks=True, half=False)
                except Exception as type_error:
                    self._handle_inference_error(type_error, "ultralytics_predict")
                    if "dtype" in str(type_error).lower() or "half" in str(type_error).lower():
                        logger.warning(f"Retrying inference with explicit float32 mode")
                        # Retry with explicit float32
                        results = self._yolo.predict(processed_imgs, imgsz=imgsz, conf=conf, verbose=False, retina_masks=True, half=False)
                    else:
                        raise type_error
                
                # Process results with type safety
                for r in results:
                    if r is None:
                        out.append({'detections': 0, 'confidence': 0.0, 'masks': [], 'bboxes': [], 'centroids': []})
                        continue

                    current_masks = []
                    current_bboxes = []
                    current_centroids = []
                    current_confidence = 0.0

                    try:
                        if hasattr(r, 'masks') and r.masks is not None:
                            polys = r.masks.xy
                            current_masks = polys
                            
                            # Safe confidence extraction
                            if hasattr(r, 'boxes') and r.boxes is not None and r.boxes.conf.numel() > 0:
                                conf_tensor = ensure_float32(r.boxes.conf.cpu())
                                current_confidence = float(np.mean(conf_tensor.numpy()))

                            # Extract bboxes from masks or r.boxes
                            if hasattr(r, 'boxes') and r.boxes is not None:
                                bbox_tensor = ensure_float32(r.boxes.xyxy.cpu())
                                current_bboxes = bbox_tensor.numpy().tolist()
                                # Calculate centroids from bboxes
                                current_centroids = [
                                    ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                                    for box in current_bboxes
                                ]
                            elif polys:
                                # Fallback: calculate bboxes from mask polygons
                                current_bboxes = [
                                    [np.min(p[:, 0]), np.min(p[:, 1]), np.max(p[:, 0]), np.max(p[:, 1])]
                                    for p in polys
                                ]
                                current_centroids = [
                                    (np.mean(p[:, 0]), np.mean(p[:, 1]))
                                    for p in polys
                                ]

                        elif hasattr(r, 'boxes') and r.boxes is not None:
                            # If no masks, use only boxes
                            bbox_tensor = ensure_float32(r.boxes.xyxy.cpu())
                            current_bboxes = bbox_tensor.numpy().tolist()
                            
                            if r.boxes.conf.numel() > 0:
                                conf_tensor = ensure_float32(r.boxes.conf.cpu())
                                current_confidence = float(np.mean(conf_tensor.numpy()))
                            
                            current_centroids = [
                                ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                                for box in current_bboxes
                            ]

                    except Exception as processing_error:
                        logger.warning(f"Error processing result tensors: {processing_error}")
                        # Fallback to empty result for this image
                        current_masks = []
                        current_bboxes = []
                        current_centroids = []
                        current_confidence = 0.0

                    out.append({
                        'detections': len(current_bboxes),
                        'confidence': current_confidence,
                        'masks': current_masks,
                        'bboxes': current_bboxes,
                        'centroids': current_centroids
                    })
                
                # Track performance metrics for Ultralytics
                total_inference_time = time.time() - inference_start
                self._inference_times.append(total_inference_time)
                self._total_inferences += 1
                self._track_dtype_performance(input_dtype_str, total_inference_time, batch_size)
                
                # Log performance metrics periodically
                if self._total_inferences % 10 == 0:
                    avg_time = sum(self._inference_times[-10:]) / min(10, len(self._inference_times))
                    logger.debug(f"Average inference time (last 10): {avg_time:.3f}s, backend: {self.backend}, dtype: {input_dtype_str}")
                
                return out
            except Exception as e:
                self._handle_inference_error(e, "ultralytics_inference")
                logger.error(f"Model inference error (ultralytics) on {self.model_path}: {e}", exc_info=True)
                return [{'detections': 0, 'confidence': 0.0, 'masks': [], 'bboxes': [], 'centroids': []} for _ in imgs]

        # ONNX/Torch placeholders (not fully implemented here)
        if self.backend == 'onnx' and self._sess is not None:
            return [{'detections': 0, 'confidence': 0.0, 'masks': [], 'bboxes': [], 'centroids': []} for _ in imgs]
        if self.backend == 'torch' and self._torch_model is not None:
            return [{'detections': 0, 'confidence': 0.0, 'masks': [], 'bboxes': [], 'centroids': []} for _ in imgs]

        # Track performance metrics before returning
        total_inference_time = time.time() - inference_start
        self._inference_times.append(total_inference_time)
        self._total_inferences += 1
        
        # Track performance by data type
        self._track_dtype_performance(input_dtype_str, total_inference_time, batch_size)
        
        # Log performance metrics periodically
        if self._total_inferences % 10 == 0:
            avg_time = sum(self._inference_times[-10:]) / min(10, len(self._inference_times))
            logger.debug(f"Average inference time (last 10): {avg_time:.3f}s, dtype: {input_dtype_str}")

        # fallback
        return [{'detections': 0, 'confidence': 0.0, 'masks': [], 'bboxes': [], 'centroids': []} for _ in imgs]


