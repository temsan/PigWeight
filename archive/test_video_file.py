#!/usr/bin/env python3
"""
Тест загрузки и воспроизведения видеофайла в ультра-быстрой системе
"""

import asyncio
import websockets
import requests
import time
import os
from pathlib import Path

class VideoFileTest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        
    def create_test_video(self):
        """Создает простое тестовое видео"""
        try:
            import cv2
            import numpy as np
            
            # Создаем простое тестовое видео
            test_video_path = "test_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (640, 480))
            
            for i in range(50):  # 5 секунд при 10 FPS
                # Создаем цветной кадр
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                color = (i * 5, 100, 255 - i * 3)
                cv2.rectangle(frame, (50, 50), (590, 430), color, -1)
                
                # Добавляем текст с номером кадра
                cv2.putText(frame, f'Frame {i+1}/50', (200, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            print(f"✅ Тестовое видео создано: {test_video_path}")
            return test_video_path
            
        except ImportError:
            print("⚠️  OpenCV не найден, используем существующий видеофайл")
            return None
        except Exception as e:
            print(f"❌ Ошибка создания видео: {e}")
            return None
    
    async def test_file_upload_and_playback(self, video_file_path=None):
        """Тестирует загрузку и воспроизведение видеофайла"""
        print("\n🎬 Тест загрузки и воспроизведения видеофайла")
        print("-" * 50)
        
        # Создаем или используем тестовое видео
        if not video_file_path:
            video_file_path = self.create_test_video()
        
        if not video_file_path or not os.path.exists(video_file_path):
            print("❌ Тестовое видео недоступно")
            return
        
        try:
            # 1. Загружаем видеофайл
            print("📤 Загружаем видеофайл...")
            with open(video_file_path, 'rb') as f:
                files = {'file': (os.path.basename(video_file_path), f, 'video/mp4')}
                data = {
                    'camera': 'file_cam',
                    'id': 'test_session_123'
                }
                
                response = requests.post(
                    f"{self.base_url}/api/video_file/open_ultra_fast",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                file_info = response.json()
                print(f"✅ Файл загружен успешно:")
                print(f"   • Длительность: {file_info.get('duration', 'N/A')} сек")
                print(f"   • FPS: {file_info.get('fps', 'N/A')}")
                print(f"   • Кадров: {file_info.get('frame_count', 'N/A')}")
                print(f"   • ID сессии: {file_info.get('id', 'N/A')}")
            else:
                print(f"❌ Ошибка загрузки: HTTP {response.status_code}")
                print(f"   Ответ: {response.text}")
                return
            
            # 2. Тестируем получение отдельных кадров
            print("\n🎯 Тестируем получение кадров...")
            session_id = file_info.get('id', 'test_session_123')
            duration = file_info.get('duration', 5)
            
            test_times = [0, duration * 0.25, duration * 0.5, duration * 0.75, duration * 0.9]
            
            for t in test_times:
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{self.base_url}/api/video_file/frame_ultra_fast",
                        params={'id': session_id, 't': t},
                        timeout=5
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        frame_size = len(response.content)
                        total_time = (end_time - start_time) * 1000
                        
                        # Получаем заголовки с метриками
                        seek_ms = response.headers.get('X-Seek-Ms', 'N/A')
                        infer_ms = response.headers.get('X-Inference-Ms', 'N/A')
                        encode_ms = response.headers.get('X-Encode-Ms', 'N/A')
                        
                        print(f"   ✅ Кадр t={t:.1f}s: {frame_size} байт ({total_time:.1f}ms)")
                        print(f"      └─ Seek: {seek_ms}ms, Infer: {infer_ms}ms, Encode: {encode_ms}ms")
                    else:
                        print(f"   ❌ Кадр t={t:.1f}s: HTTP {response.status_code}")
                
                except Exception as e:
                    print(f"   ❌ Кадр t={t:.1f}s: {e}")
            
            # 3. Тестируем WebSocket поток
            print("\n🌊 Тестируем WebSocket поток видеофайла...")
            await self.test_websocket_file_stream(session_id)
            
        except Exception as e:
            print(f"❌ Ошибка теста видеофайла: {e}")
    
    async def test_websocket_file_stream(self, session_id):
        """Тестирует WebSocket поток для видеофайла"""
        try:
            uri = f"{self.ws_url}/ws/video_file_ultra_fast?id={session_id}"
            
            frame_count = 0
            start_time = time.time()
            
            async with websockets.connect(uri) as websocket:
                print("   ✅ WebSocket соединение установлено")
                
                # Получаем кадры в течение 5 секунд
                timeout_time = start_time + 5
                
                while time.time() < timeout_time and frame_count < 20:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        
                        if isinstance(message, bytes):
                            frame_count += 1
                            if frame_count % 5 == 0:  # Каждый 5-й кадр
                                print(f"   📺 Получен кадр {frame_count}: {len(message)} байт")
                    
                    except asyncio.TimeoutError:
                        print("   ⚠️  Таймаут получения кадра")
                        break
                
                total_time = time.time() - start_time
                fps = frame_count / total_time if total_time > 0 else 0
                
                print(f"\n   📊 Результаты WebSocket потока:")
                print(f"   • Получено кадров: {frame_count}")
                print(f"   • Время: {total_time:.1f}s")
                print(f"   • Средний FPS: {fps:.1f}")
                
                if frame_count > 0:
                    print("   ✅ WebSocket поток работает!")
                else:
                    print("   ❌ WebSocket поток не работает")
                    
        except Exception as e:
            print(f"   ❌ Ошибка WebSocket потока: {e}")
    
    def cleanup_test_files(self):
        """Удаляет тестовые файлы"""
        test_files = ["test_video.mp4"]
        for file in test_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"🗑️  Удален файл: {file}")
                except Exception as e:
                    print(f"⚠️  Не удалось удалить {file}: {e}")

async def main():
    print("🎬 Тест системы воспроизведения видеофайлов")
    print("=" * 50)
    
    test = VideoFileTest()
    
    try:
        # Проверяем доступность сервера
        response = requests.get(f"{test.base_url}/api/health", timeout=3)
        if response.status_code != 200:
            print("❌ Сервер недоступен")
            return
        
        print("✅ Сервер доступен")
        
        # Запускаем тест
        await test.test_file_upload_and_playback()
        
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
    
    finally:
        # Очищаем тестовые файлы
        test.cleanup_test_files()

if __name__ == "__main__":
    asyncio.run(main())
