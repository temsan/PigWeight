import os
import logging
from typing import Any, Dict, List
import numpy as np
from datetime import datetime

try:
    import onnxruntime as ort
    _HAVE_ONNX = True
    print("✅ ONNX Runtime доступен для CPU inference")
except Exception as e:
    _HAVE_ONNX = False
    print(f"⚠️  ONNX Runtime не доступен: {e}")
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


class ModelAdapter:
    """Adapter: prefer ultralytics (.pt), then ONNX, then TorchScript.

    Loads model according to file extension and exposes infer(imgs) -> list[dict].
    """

    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or os.getenv('DEVICE', 'cpu')
        self.backend = None
        self._sess = None  # ONNX Runtime session
        self._torch_model = None
        self._yolo = None

        # Performance tracking
        self._inference_times = []
        self._total_inferences = 0

        perf_logger = logging.getLogger("perf.model_adapter")

        # Priority: ONNX (CPU) > Ultralytics (GPU/CPU) > TorchScript
        if model_path and model_path.endswith('.onnx') and _HAVE_ONNX:
            # ONNX model - optimized for CPU inference
            perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Loading ONNX model from {model_path} for CPU inference")
            try:
                logger.info(f"Loading ONNX model from {model_path} for CPU inference")
                # Use CPU execution provider for maximum performance
                self._sess = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=ort.SessionOptions()
                )
                self.backend = 'onnx'
                logger.info(f"✅ ONNX model loaded successfully")
                perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] ONNX backend initialized for CPU inference")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model {model_path}: {e}")
                self._sess = None

        elif model_path and model_path.endswith('.pt') and _HAVE_ULTRALYTICS:
            perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Loading Ultralytics model from {model_path} on device={self.device}")
            try:
                logger.info(f"Loading ultralytics model from {model_path} on device={self.device}")
                self._yolo = YOLO(model_path)
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
                            self._yolo.model.half()
                        except Exception:
                            logger.warning('Failed to set model to half precision; continuing with float32')
                except Exception:
                    pass
                self.backend = 'ultralytics'
            except Exception as e:
                logger.warning(f"Failed to load ultralytics model {model_path}: {e}")
                self._yolo = None
        # ensure model on correct device and half if possible
        if self._yolo is not None:
            try:
                if self.device and hasattr(self._yolo, 'to'):
                    self._yolo.to(self.device)
            except Exception:
                pass

        if not self._yolo and _HAVE_ONNX and model_path.endswith('.onnx'):
            try:
                logger.info(f"Loading ONNX model from {model_path}")
                self._sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.backend = 'onnx'
            except Exception:
                logger.warning(f"Failed to load ONNX model {model_path}")
                self._sess = None

        if not self._yolo and not self._sess and _HAVE_TORCH and model_path.endswith('.pt'):
            try:
                # try jit first
                try:
                    self._torch_model = torch.jit.load(model_path)
                except Exception:
                    self._torch_model = torch.load(model_path)
                if self.device.startswith('cuda') and torch.cuda.is_available():
                    try:
                        self._torch_model.to(self.device)
                    except Exception:
                        pass
                self._torch_model.eval()
                self.backend = 'torch'
            except Exception as e:
                logger.warning(f"Failed to load Torch model {model_path}: {e}")
                self._torch_model = None

        logger.info(f"ModelAdapter initialized: path={model_path}, backend={self.backend}, device={self.device}")

    def infer(self, imgs: List[np.ndarray]) -> List[Dict[str, Any]]:
        import time
        inference_start = time.time()

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

                results = []
                for img in imgs:
                    if img is None:
                        results.append({'detections': 0, 'confidence': 0.0})
                        continue

                    try:
                        # Preprocess image for ONNX
                        import cv2
                        if img.shape[:2] != (input_shape[2], input_shape[3]):
                            # Resize if needed
                            img_resized = cv2.resize(img, (input_shape[3], input_shape[2]))
                        else:
                            img_resized = img

                        # Convert to NCHW format and normalize
                        if img_resized.ndim == 3:
                            img_tensor = np.transpose(img_resized, (2, 0, 1))  # HWC -> CHW
                        else:
                            img_tensor = img_resized

                        img_tensor = img_tensor.astype(np.float32) / 255.0
                        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension

                        # Run inference
                        outputs = self._sess.run(None, {input_name: img_tensor})

                        # Parse YOLO outputs (simplified - adapt based on your model)
                        # This is a basic implementation, you may need to adjust based on your model's output format
                        detections = 0
                        confidence = 0.0

                        if outputs and len(outputs) > 0:
                            # Assuming first output contains detections
                            output = outputs[0]
                            # Count non-zero detections (simplified)
                            if output.shape[0] > 0:
                                detections = int(np.sum(output > 0.5))  # Threshold-based counting
                                confidence = float(np.mean(output[output > 0.5])) if detections > 0 else 0.0

                        results.append({
                            'detections': detections,
                            'confidence': confidence
                        })

                    except Exception as e:
                        logger.warning(f"ONNX inference error for image: {e}")
                        results.append({'detections': 0, 'confidence': 0.0})

                out = results
                self._total_inferences += len(imgs)
                inference_time = time.time() - inference_start
                self._inference_times.append(inference_time)

                perf_logger.info(".2f")
                return out

            except Exception as e:
                logger.error(f"ONNX inference error: {e}", exc_info=True)
                return [{'detections': 0, 'confidence': 0.0} for _ in imgs]

        # Ultralytics inference (GPU/CPU)
        elif self.backend == 'ultralytics' and self._yolo is not None:
            try:
                perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Running Ultralytics inference on {len(imgs)} images")
                imgsz = int(os.getenv('IMG_SIZE', '960'))
                conf = float(os.getenv('CONF_THRESHOLD', '0.30'))
                # ultralytics expects positional source argument or 'source=', not 'imgs='
                # pass list of images as first positional argument
                results = self._yolo.predict(imgs, imgsz=imgsz, conf=conf, verbose=False, retina_masks=True)
                for r in results:
                    if r is None:
                        out.append({'detections': 0, 'confidence': 0.0})
                        continue
                    # masks
                    if hasattr(r, 'masks') and r.masks is not None:
                        polys = r.masks.xy
                        dets = len(polys)
                        conf_avg = 0.0
                        out.append({'detections': int(dets), 'confidence': float(conf_avg), 'masks': polys})
                    else:
                        # fallback to boxes
                        try:
                            boxes = getattr(r, 'boxes', None)
                            if boxes is not None and hasattr(boxes, 'xyxy'):
                                confs = getattr(boxes, 'conf', None)
                                num = len(boxes.xyxy) if hasattr(boxes, 'xyxy') else 0
                                avg_conf = float(np.mean(confs.tolist())) if confs is not None and len(confs) else 0.0
                                out.append({'detections': int(num), 'confidence': float(avg_conf)})
                            else:
                                out.append({'detections': 0, 'confidence': 0.0})
                        except Exception:
                            out.append({'detections': 0, 'confidence': 0.0})
                inference_end_time = time.time()
                total_inference_time = inference_end_time - time.time()  # from the start of infer method
                perf_logger.info(".2f")
                return out
            except Exception as e:
                logger.error(f"Model inference error (ultralytics) on {self.model_path}: {e}", exc_info=True)
                perf_logger.error(".3f")
                # fallback to empty
                return [{'detections': 0, 'confidence': 0.0} for _ in imgs]

        # ONNX/Torch placeholders (not fully implemented here)
        if self.backend == 'onnx' and self._sess is not None:
            return [{'detections': 0, 'confidence': 0.0} for _ in imgs]
        if self.backend == 'torch' and self._torch_model is not None:
            return [{'detections': 0, 'confidence': 0.0} for _ in imgs]

        # fallback
        return [{'detections': 0, 'confidence': 0.0} for _ in imgs]


