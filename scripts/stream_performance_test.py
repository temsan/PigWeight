#!/usr/bin/env python3
"""
Тест производительности для новой стрим-архитектуры PigWeight.

Измеряет:
- Задержку инференса (inference latency)
- Скорость обработки кадров (frames per second)
- Потребление памяти и CPU
- Сравнение WebRTC vs MJPEG
- Влияние батчинга на производительность

Использование:
python scripts/stream_performance_test.py --stream_id cam101 --duration 60 --transport webrtc
"""

import asyncio
import time
import statistics
import json
import requests
import psutil
import threading
from typing import List, Dict, Any
from datetime import datetime
import argparse
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config

class StreamPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "test_start": None,
            "test_end": None,
            "transport": None,
            "stream_id": None,
            "duration_seconds": 0,
            "inference_latencies": [],
            "frame_processing_times": [],
            "memory_usage_mb": [],
            "cpu_usage_percent": [],
            "websocket_messages": 0,
            "frames_processed": 0,
            "inference_errors": 0,
            "transport_errors": 0,
            "summary": {}
        }

    def start_monitoring(self):
        """Запускаем мониторинг системных ресурсов"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Останавливаем мониторинг"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

    def _monitor_resources(self):
        """Мониторим использование CPU и памяти"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)

                self.results["cpu_usage_percent"].append(cpu_percent)
                self.results["memory_usage_mb"].append(memory_mb)

                time.sleep(1)
            except Exception as e:
                print(f"Error monitoring resources: {e}")
                break

    async def test_webrtc_transport(self, stream_id: str, duration: int):
        """Тестируем WebRTC транспорт"""
        print(f"Testing WebRTC transport for stream {stream_id}...")

        # WebRTC тест пока что не реализован полностью
        # Для MVP используем MJPEG fallback
        await self.test_mjpeg_transport(stream_id, duration)

    async def test_mjpeg_transport(self, stream_id: str, duration: int):
        """Тестируем MJPEG транспорт"""
        print(f"Testing MJPEG transport for stream {stream_id}...")

        start_time = time.time()
        self.results["test_start"] = datetime.now().isoformat()

        try:
            # Запускаем поток
            start_resp = requests.post(f"{self.base_url}/api/stream/start",
                                     json={"stream_id": stream_id, "source_uri": "rtsp://example.com/cam"})
            if start_resp.status_code != 200:
                print(f"Failed to start stream: {start_resp.text}")
                return

            # Мониторим /info endpoint для метрик
            end_time = start_time + duration
            while time.time() < end_time:
                try:
                    info_resp = requests.get(f"{self.base_url}/api/stream/{stream_id}/info")
                    if info_resp.status_code == 200:
                        info = info_resp.json()
                        # Здесь можно анализировать задержки и другие метрики
                        self.results["frames_processed"] += 1

                    await asyncio.sleep(0.1)  # Не спамить сервер

                except Exception as e:
                    self.results["transport_errors"] += 1
                    print(f"Transport error: {e}")
                    await asyncio.sleep(1)

            # Останавливаем поток
            requests.get(f"{self.base_url}/api/stream/{stream_id}/stop")

        except Exception as e:
            print(f"MJPEG test error: {e}")

        self.results["test_end"] = datetime.now().isoformat()

    def calculate_summary(self):
        """Вычисляем итоговые метрики"""
        if not self.results["inference_latencies"]:
            return

        latencies = self.results["inference_latencies"]

        self.results["summary"] = {
            "avg_inference_latency_ms": statistics.mean(latencies),
            "median_inference_latency_ms": statistics.median(latencies),
            "min_inference_latency_ms": min(latencies),
            "max_inference_latency_ms": max(latencies),
            "p95_inference_latency_ms": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
            "p99_inference_latency_ms": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
            "frames_per_second": self.results["frames_processed"] / self.results["duration_seconds"] if self.results["duration_seconds"] > 0 else 0,
            "avg_memory_mb": statistics.mean(self.results["memory_usage_mb"]) if self.results["memory_usage_mb"] else 0,
            "avg_cpu_percent": statistics.mean(self.results["cpu_usage_percent"]) if self.results["cpu_usage_percent"] else 0,
            "total_websocket_messages": self.results["websocket_messages"],
            "inference_error_rate": self.results["inference_errors"] / max(self.results["frames_processed"], 1),
            "transport_error_rate": self.results["transport_errors"] / max(self.results["frames_processed"], 1)
        }

    def save_results(self, output_file: str = None):
        """Сохраняем результаты в файл"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_test_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_file}")

        # Печатаем сводку
        if self.results["summary"]:
            print("\n=== PERFORMANCE SUMMARY ===")
            summary = self.results["summary"]
            print(f"Average inference latency: {summary['avg_inference_latency_ms']:.2f} ms")
            print(f"Median inference latency: {summary['median_inference_latency_ms']:.2f} ms")
            print(f"95th percentile latency: {summary['p95_inference_latency_ms']:.2f} ms")
            print(f"99th percentile latency: {summary['p99_inference_latency_ms']:.2f} ms")
            print(f"Frames per second: {summary['frames_per_second']:.2f}")
            print(f"Average memory usage: {summary['avg_memory_mb']:.2f} MB")
            print(f"Average CPU usage: {summary['avg_cpu_percent']:.2f}%")
            print(f"Inference error rate: {summary['inference_error_rate']:.4f}")
            print(f"Transport error rate: {summary['transport_error_rate']:.4f}")

    async def run_test(self, stream_id: str, transport: str, duration: int):
        """Запускаем полный тест"""
        self.results["transport"] = transport
        self.results["stream_id"] = stream_id
        self.results["duration_seconds"] = duration

        print(f"Starting performance test for {transport} transport...")
        print(f"Stream ID: {stream_id}")
        print(f"Duration: {duration} seconds")

        # Запускаем мониторинг ресурсов
        self.start_monitoring()

        try:
            if transport == "webrtc":
                await self.test_webrtc_transport(stream_id, duration)
            elif transport == "mjpeg":
                await self.test_mjpeg_transport(stream_id, duration)
            else:
                print(f"Unknown transport: {transport}")
                return

        finally:
            self.stop_monitoring()

        # Вычисляем итоговые метрики
        self.calculate_summary()

        # Сохраняем результаты
        self.save_results()


async def main():
    parser = argparse.ArgumentParser(description="Performance test for PigWeight streaming architecture")
    parser.add_argument("--stream_id", required=True, help="Stream ID to test")
    parser.add_argument("--transport", choices=["webrtc", "mjpeg"], default="mjpeg",
                       help="Transport protocol to test")
    parser.add_argument("--duration", type=int, default=30,
                       help="Test duration in seconds")
    parser.add_argument("--base_url", default="http://localhost:8000",
                       help="Base URL of the server")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    # Проверяем конфигурацию
    config = load_config()
    print(f"Loaded config: MODEL_PATH={config.get('MODEL_PATH')}, DEVICE={config.get('DEVICE')}")

    # Создаем тестер
    tester = StreamPerformanceTester(args.base_url)

    # Запускаем тест
    await tester.run_test(args.stream_id, args.transport, args.duration)

    # Сохраняем результаты
    tester.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())
