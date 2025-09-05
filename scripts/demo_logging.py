#!/usr/bin/env python3
"""
Демонстрация системы логирования PigWeight.

Показывает все добавленные точки логирования в ключевых компонентах:
- API endpoints (WebRTC, streams, WebSocket)
- Frame Broker (publish/subscribe)
- Results Store (put/get operations)
- Model Adapter (ONNX/Ultralytics inference)
- Inference Worker (batch processing)

Использование:
python scripts/demo_logging.py
"""

import asyncio
import time
import logging
import json
from datetime import datetime
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/demo_logging.log', mode='w')
    ]
)

# Импорт компонентов
from core.frame_broker import FRAME_BROKER
from core.results_store import RESULTS_STORE
from services.model_adapter import ModelAdapter
from services.inference_worker import InferenceWorker

async def demo_frame_broker():
    """Демонстрация логирования Frame Broker"""
    print("\n" + "="*60)
    print("🎬 DEMO: Frame Broker Logging")
    print("="*60)

    # Создаем тестовый кадр
    test_frame = b"test_jpeg_data_" + b"x" * 1000  # 1KB frame
    stream_id = "demo_stream"

    print(f"📤 Publishing frame for {stream_id}...")
    await FRAME_BROKER.publish(stream_id, 1, time.time(), test_frame)

    print(f"📥 Subscribing to {stream_id}...")
    queue = FRAME_BROKER.subscribe(stream_id)

    print(f"📤 Publishing another frame...")
    await FRAME_BROKER.publish(stream_id, 2, time.time(), test_frame)

    await asyncio.sleep(0.1)  # Allow async operations to complete

def demo_results_store():
    """Демонстрация логирования Results Store"""
    print("\n" + "="*60)
    print("💾 DEMO: Results Store Logging")
    print("="*60)

    stream_id = "demo_stream"

    # Добавляем тестовый результат
    test_result = {
        'detections': 3,
        'confidence': 0.85,
        'masks': []  # simplified
    }

    print(f"💾 Storing result for {stream_id} frame 1...")
    RESULTS_STORE.put(stream_id, 1, test_result)

    print(f"📖 Getting latest result for {stream_id}...")
    latest = RESULTS_STORE.get_latest(stream_id)
    if latest:
        print(f"   ✅ Retrieved: detections={latest.get('detections', 0)}")

    print(f"🔍 Getting result for {stream_id} frame 1...")
    specific = RESULTS_STORE.get_for_frame(stream_id, 1)
    if specific:
        print(f"   ✅ Retrieved: detections={specific.get('detections', 0)}")

def demo_model_adapter():
    """Демонстрация логирования Model Adapter"""
    print("\n" + "="*60)
    print("🤖 DEMO: Model Adapter Logging")
    print("="*60)

    # Тестовая модель (если существует)
    model_path = "models/pig_yolo11-seg.v4.pt"
    if os.path.exists(model_path):
        print(f"📋 Testing with existing model: {model_path}")
        adapter = ModelAdapter(model_path)

        print(f"🔧 Backend: {adapter.backend}")
        print(f"🎯 Device: {adapter.device}")

        # Тестовый inference (если модель загружена)
        if adapter.backend:
            print("🚀 Running test inference...")
            # Создаем тестовое изображение
            import numpy as np
            test_img = np.random.randint(0, 255, (960, 960, 3), dtype=np.uint8)

            try:
                results = adapter.infer([test_img])
                print(f"   ✅ Inference successful: {len(results)} results")
                if results:
                    print(f"   📊 Detections: {results[0].get('detections', 0)}")
            except Exception as e:
                print(f"   ❌ Inference failed: {e}")
        else:
            print("⚠️  Model not loaded - no backend available")
    else:
        print(f"📋 Testing with dummy model (file not found: {model_path})")
        adapter = ModelAdapter("dummy.onnx")  # Should trigger ONNX path
        print(f"🔧 Backend: {adapter.backend}")

def demo_inference_worker():
    """Демонстрация логирования Inference Worker"""
    print("\n" + "="*60)
    print("⚙️  DEMO: Inference Worker Logging")
    print("="*60)

    stream_id = "demo_worker"
    worker = InferenceWorker(stream_id, batch_size=4)

    print(f"🔧 Created worker for {stream_id}")
    print(f"📊 Batch size: {worker.batch_size}")
    print(f"⏱️  Max wait: {worker.max_wait_ms}ms")

    # Имитация работы (без реального запуска)
    print("📝 Worker would log performance on real frames:")
    print("   - Batch processing times")
    print("   - Inference latencies")
    print("   - Throughput metrics")
    print("   - Performance summaries")

def demo_api_simulation():
    """Имитация API логирования"""
    print("\n" + "="*60)
    print("🌐 DEMO: API Endpoints Logging")
    print("="*60)

    perf_logger = logging.getLogger("perf.api")

    print("📡 Simulating API operations...")

    # Имитация WebRTC offer
    perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] WebRTC offer processed for stream demo_stream")

    # Имитация stream start
    perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting stream demo_stream with source rtsp://demo")

    # Имитация WebSocket
    perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket connection established for stream demo_stream")

    print("✅ API logging simulation complete")

def main():
    """Основная функция демонстрации"""
    print("🐷 PIGWEIGHT LOGGING SYSTEM DEMO")
    print("Показываем все точки логирования в системе")
    print("="*80)

    # Синхронные демо
    demo_results_store()
    demo_model_adapter()
    demo_inference_worker()
    demo_api_simulation()

    # Асинхронные демо
    asyncio.run(demo_frame_broker())

    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    print("📋 Logged data saved to: logs/demo_logging.log")
    print("🔍 Check the log file to see all performance metrics!")
    print("\n🎯 Key logging points added:")
    print("   • API endpoints (WebRTC, streams, WebSocket)")
    print("   • Frame Broker (publish/subscribe operations)")
    print("   • Results Store (put/get with TTL)")
    print("   • Model Adapter (ONNX/Ultralytics inference)")
    print("   • Inference Worker (batch processing & summaries)")

if __name__ == "__main__":
    main()
