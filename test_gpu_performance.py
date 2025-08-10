"""
Тестирование GPU ускорения и WebRTC функциональности
Проверка всех улучшений производительности
"""

import asyncio
import time
import requests
import numpy as np
import cv2
from typing import List, Dict
import json
import concurrent.futures
from pathlib import Path

# Конфигурация тестов
API_BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = "test_video.mp4"
TEST_VIDEO_ID = "performance_test"

class PerformanceTester:
    """Класс для тестирования производительности GPU обработки"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.results = {}
    
    def create_test_video(self, duration: int = 10, fps: int = 30):
        """Создание тестового видео"""
        print(f"🎬 Creating test video: {duration}s @ {fps}fps")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(TEST_VIDEO_PATH, fourcc, fps, (640, 480))
        
        total_frames = duration * fps
        for i in range(total_frames):
            # Создаем кадр с движущимся объектом
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Фон
            frame[:] = (50, 50, 100)
            
            # Движущийся круг
            x = int(320 + 200 * np.sin(i * 0.1))
            y = int(240 + 100 * np.cos(i * 0.05))
            cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
            
            # Текст с номером кадра
            cv2.putText(frame, f"Frame: {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Test video created: {TEST_VIDEO_PATH}")
    
    def test_upload_performance(self):
        """Тест скорости загрузки с индексированием"""
        print("\n🚀 Testing GPU upload with indexing...")
        
        start_time = time.time()
        
        with open(TEST_VIDEO_PATH, 'rb') as f:
            files = {'file': f}
            data = {'id': TEST_VIDEO_ID}
            
            response = requests.post(
                f"{self.base_url}/api/gpu/video/upload",
                files=files,
                data=data
            )
        
        upload_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            self.results['upload'] = {
                'time': upload_time,
                'index_time': result.get('index_time', 'N/A'),
                'index_size': result.get('index_size', 'N/A'),
                'gpu_enabled': result.get('gpu_enabled', False),
                'device': result.get('device', 'N/A')
            }
            
            print(f"✅ Upload completed in {upload_time:.2f}s")
            print(f"   📊 Index time: {result.get('index_time')}")
            print(f"   📈 Index size: {result.get('index_size')} entries")
            print(f"   🔥 GPU enabled: {result.get('gpu_enabled')}")
            print(f"   💻 Device: {result.get('device')}")
        else:
            print(f"❌ Upload failed: {response.text}")
    
    def test_frame_access_speed(self, num_tests: int = 50):
        """Тест скорости доступа к кадрам через индекс"""
        print(f"\n⚡ Testing frame access speed ({num_tests} requests)...")
        
        timestamps = np.linspace(0, 9, num_tests)  # 0-9 секунд
        times = []
        
        for i, ts in enumerate(timestamps):
            start_time = time.time()
            
            response = requests.get(
                f"{self.base_url}/api/gpu/video/{TEST_VIDEO_ID}/frame",
                params={'t': ts, 'quality': 85}
            )
            
            request_time = time.time() - start_time
            times.append(request_time)
            
            if i % 10 == 0:
                print(f"   Frame {i+1}/{num_tests}: {request_time:.3f}s")
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        self.results['frame_access'] = {
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_requests': num_tests
        }
        
        print(f"✅ Average frame access: {avg_time:.3f}s")
        print(f"   🏃‍♂️ Fastest: {min_time:.3f}s")
        print(f"   🐌 Slowest: {max_time:.3f}s")
    
    def test_gpu_inference_performance(self, num_tests: int = 20):
        """Тест производительности GPU инференса"""
        print(f"\n🧠 Testing GPU inference performance ({num_tests} requests)...")
        
        timestamps = np.linspace(0, 9, num_tests)
        times = []
        inference_times = []
        
        for i, ts in enumerate(timestamps):
            start_time = time.time()
            
            response = requests.get(
                f"{self.base_url}/api/gpu/video/{TEST_VIDEO_ID}/frame",
                params={'t': ts, 'inference': True, 'quality': 85}
            )
            
            total_time = time.time() - start_time
            times.append(total_time)
            
            if response.status_code == 200:
                inference_time = response.headers.get('X-Inference-Time', '0s')
                inference_times.append(float(inference_time.replace('s', '')))
                
                if i % 5 == 0:
                    print(f"   Inference {i+1}/{num_tests}: {total_time:.3f}s (GPU: {inference_time})")
        
        avg_total = np.mean(times)
        avg_inference = np.mean(inference_times) if inference_times else 0
        
        self.results['gpu_inference'] = {
            'average_total_time': avg_total,
            'average_inference_time': avg_inference,
            'total_requests': num_tests
        }
        
        print(f"✅ Average total time: {avg_total:.3f}s")
        print(f"   🔥 Average GPU inference: {avg_inference:.3f}s")
    
    def test_batch_inference(self, batch_sizes: List[int] = [1, 4, 8, 16]):
        """Тест batch инференса для разных размеров batch"""
        print(f"\n📦 Testing batch inference performance...")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            timestamps = np.linspace(0, 9, batch_size).tolist()
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/gpu/video/{TEST_VIDEO_ID}/batch_inference",
                json=timestamps
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                batch_results[batch_size] = {
                    'processing_time': processing_time,
                    'server_time': result.get('processing_time', 'N/A'),
                    'processed_frames': result.get('processed_frames', 0)
                }
                
                print(f"     ✅ Batch {batch_size}: {processing_time:.3f}s total")
                print(f"        Server time: {result.get('processing_time')}")
            else:
                print(f"     ❌ Batch {batch_size} failed: {response.text}")
        
        self.results['batch_inference'] = batch_results
    
    def test_concurrent_access(self, num_concurrent: int = 10):
        """Тест параллельного доступа к кадрам"""
        print(f"\n🔀 Testing concurrent access ({num_concurrent} parallel requests)...")
        
        def get_random_frame():
            ts = np.random.uniform(0, 9)
            start_time = time.time()
            
            response = requests.get(
                f"{self.base_url}/api/gpu/video/{TEST_VIDEO_ID}/frame",
                params={'t': ts, 'quality': 85}
            )
            
            return time.time() - start_time, response.status_code == 200
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(get_random_frame) for _ in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        times = [r[0] for r in results]
        success_count = sum(r[1] for r in results)
        
        self.results['concurrent_access'] = {
            'total_time': total_time,
            'average_request_time': np.mean(times),
            'success_rate': success_count / num_concurrent,
            'concurrent_requests': num_concurrent
        }
        
        print(f"✅ Concurrent test completed in {total_time:.2f}s")
        print(f"   📊 Average request time: {np.mean(times):.3f}s")
        print(f"   ✅ Success rate: {success_count}/{num_concurrent}")
    
    def test_cache_performance(self):
        """Тест эффективности кеширования"""
        print(f"\n💾 Testing cache performance...")
        
        # Первый запрос (cache miss)
        ts = 5.0
        start_time = time.time()
        response1 = requests.get(
            f"{self.base_url}/api/gpu/video/{TEST_VIDEO_ID}/frame",
            params={'t': ts, 'quality': 85}
        )
        miss_time = time.time() - start_time
        
        # Второй запрос (cache hit)
        start_time = time.time()
        response2 = requests.get(
            f"{self.base_url}/api/gpu/video/{TEST_VIDEO_ID}/frame",
            params={'t': ts, 'quality': 85}
        )
        hit_time = time.time() - start_time
        
        cache_status1 = response1.headers.get('X-Cache-Status', 'UNKNOWN')
        cache_status2 = response2.headers.get('X-Cache-Status', 'UNKNOWN')
        
        self.results['cache_performance'] = {
            'miss_time': miss_time,
            'hit_time': hit_time,
            'speedup': miss_time / hit_time if hit_time > 0 else 0,
            'cache_status_1': cache_status1,
            'cache_status_2': cache_status2
        }
        
        print(f"✅ Cache miss: {miss_time:.3f}s ({cache_status1})")
        print(f"✅ Cache hit: {hit_time:.3f}s ({cache_status2})")
        print(f"🚀 Cache speedup: {miss_time/hit_time:.1f}x" if hit_time > 0 else "N/A")
    
    def get_system_stats(self):
        """Получение статистики системы"""
        print(f"\n📈 Getting system statistics...")
        
        try:
            response = requests.get(f"{self.base_url}/api/gpu/stats")
            if response.status_code == 200:
                stats = response.json()
                self.results['system_stats'] = stats
                
                print("✅ System Statistics:")
                gpu_info = stats.get('gpu_processor', {})
                print(f"   🔥 Frames processed: {gpu_info.get('frames_processed', 'N/A')}")
                print(f"   💾 Cache hit rate: {gpu_info.get('cache_hit_rate', 'N/A')}")
                print(f"   ⚡ Avg inference time: {gpu_info.get('average_inference_time', 'N/A')}")
                print(f"   📦 Avg batch size: {gpu_info.get('average_batch_size', 'N/A')}")
                
                system_info = stats.get('system_info', {})
                print(f"   🎮 GPU available: {system_info.get('gpu_available', False)}")
                print(f"   💻 GPU device: {system_info.get('gpu_device', 'N/A')}")
            else:
                print(f"❌ Failed to get stats: {response.text}")
        
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def run_all_tests(self):
        """Запуск всех тестов производительности"""
        print("🧪 Starting comprehensive GPU performance tests...")
        print("=" * 60)
        
        # Создаем тестовое видео
        self.create_test_video()
        
        # Запускаем тесты
        self.test_upload_performance()
        time.sleep(1)  # Небольшая пауза между тестами
        
        self.test_frame_access_speed()
        time.sleep(1)
        
        self.test_gpu_inference_performance()
        time.sleep(1)
        
        self.test_batch_inference()
        time.sleep(1)
        
        self.test_concurrent_access()
        time.sleep(1)
        
        self.test_cache_performance()
        time.sleep(1)
        
        self.get_system_stats()
        
        # Сводка результатов
        self.print_summary()
    
    def print_summary(self):
        """Печать сводки результатов"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("="*60)
        
        if 'upload' in self.results:
            upload = self.results['upload']
            print(f"📤 Upload Performance:")
            print(f"   Time: {upload['time']:.2f}s")
            print(f"   GPU: {upload['gpu_enabled']} ({upload['device']})")
            print(f"   Index: {upload['index_size']} entries in {upload['index_time']}")
        
        if 'frame_access' in self.results:
            frames = self.results['frame_access']
            print(f"\n⚡ Frame Access:")
            print(f"   Average: {frames['average_time']:.3f}s")
            print(f"   Range: {frames['min_time']:.3f}s - {frames['max_time']:.3f}s")
        
        if 'gpu_inference' in self.results:
            gpu = self.results['gpu_inference']
            print(f"\n🧠 GPU Inference:")
            print(f"   Total: {gpu['average_total_time']:.3f}s")
            print(f"   GPU only: {gpu['average_inference_time']:.3f}s")
        
        if 'cache_performance' in self.results:
            cache = self.results['cache_performance']
            print(f"\n💾 Cache Performance:")
            print(f"   Speedup: {cache['speedup']:.1f}x")
            print(f"   Miss: {cache['miss_time']:.3f}s → Hit: {cache['hit_time']:.3f}s")
        
        print(f"\n🎯 Recommendations:")
        
        if 'frame_access' in self.results:
            avg_time = self.results['frame_access']['average_time']
            if avg_time < 0.05:
                print(f"   ✅ Excellent frame access speed ({avg_time:.3f}s)")
            elif avg_time < 0.1:
                print(f"   👍 Good frame access speed ({avg_time:.3f}s)")
            else:
                print(f"   ⚠️  Frame access could be faster ({avg_time:.3f}s)")
        
        if 'cache_performance' in self.results:
            speedup = self.results['cache_performance']['speedup']
            if speedup > 5:
                print(f"   ✅ Excellent cache performance ({speedup:.1f}x speedup)")
            elif speedup > 2:
                print(f"   👍 Good cache performance ({speedup:.1f}x speedup)")
            else:
                print(f"   ⚠️  Cache could be more effective ({speedup:.1f}x speedup)")
        
        print("\n🏁 Performance testing completed!")
        
        # Сохраняем результаты
        with open('performance_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("📄 Results saved to: performance_results.json")

def main():
    """Главная функция для запуска тестов"""
    print("🚀 GPU Video Processing Performance Tester")
    print("🔧 Make sure the GPU server is running on http://localhost:8000")
    
    # Проверяем доступность сервера
    try:
        response = requests.get(f"{API_BASE_URL}/api/gpu/stats", timeout=5)
        if response.status_code == 200:
            print("✅ Server is accessible")
        else:
            print("⚠️  Server response not OK")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Start the server with: python gpu_endpoints.py")
        return
    
    # Запускаем тесты
    tester = PerformanceTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
