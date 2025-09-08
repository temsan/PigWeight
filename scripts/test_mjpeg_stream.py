#!/usr/bin/env python3
"""
Быстрый тест MJPEG потока для диагностики черного экрана.
"""
import requests
import time
import cv2
import numpy as np
from io import BytesIO

def test_mjpeg_stream():
    """Тестирует MJPEG поток напрямую"""
    print("🔍 Тестируем MJPEG поток...")
    
    # Сначала запускаем поток
    stream_id = "test_stream"
    video_file = "temp/0825.mp4"
    
    try:
        # Запускаем поток
        print(f"🚀 Запускаем поток: {stream_id}")
        start_resp = requests.post(
            f"http://localhost:8000/api/stream/start?stream_id={stream_id}&source_uri={video_file}",
            timeout=10
        )
        
        if start_resp.status_code != 200:
            print(f"❌ Ошибка запуска потока: {start_resp.status_code}")
            print(start_resp.text)
            return False
        
        print("✅ Поток запущен")
        
        # Ждем немного для инициализации
        time.sleep(3)
        
        # Проверяем MJPEG feed
        print("📡 Проверяем MJPEG feed...")
        feed_url = f"http://localhost:8000/api/stream/{stream_id}/feed"
        
        response = requests.get(feed_url, stream=True, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ MJPEG feed недоступен: {response.status_code}")
            return False
        
        print(f"✅ MJPEG feed доступен: {response.headers.get('content-type')}")
        
        # Читаем первые кадры
        frames_received = 0
        black_frames = 0
        
        buffer = b""
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue
                
            buffer += chunk
            
            # Ищем JPEG кадры
            while True:
                # Ищем начало JPEG (FF D8)
                start = buffer.find(b'\xff\xd8')
                if start == -1:
                    break
                    
                # Ищем конец JPEG (FF D9)
                end = buffer.find(b'\xff\xd9', start + 2)
                if end == -1:
                    break
                    
                # Извлекаем JPEG кадр
                jpeg_data = buffer[start:end + 2]
                buffer = buffer[end + 2:]
                
                frames_received += 1
                
                try:
                    # Декодируем кадр
                    nparr = np.frombuffer(jpeg_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        mean_val = np.mean(frame)
                        print(f"📄 Кадр {frames_received}: {frame.shape}, mean={mean_val:.1f}")
                        
                        if mean_val < 10:
                            black_frames += 1
                            print(f"⚫ Черный кадр #{black_frames}")
                        else:
                            print(f"✅ Нормальный кадр")
                    else:
                        print(f"❌ Кадр {frames_received}: не удается декодировать")
                        
                except Exception as e:
                    print(f"❌ Ошибка декодирования кадра {frames_received}: {e}")
                
                # Тестируем только первые 5 кадров
                if frames_received >= 5:
                    break
            
            if frames_received >= 5:
                break
        
        print(f"\n📊 Результат:")
        print(f"Всего кадров: {frames_received}")
        print(f"Черных кадров: {black_frames}")
        print(f"Нормальных кадров: {frames_received - black_frames}")
        
        if black_frames == frames_received:
            print("❌ ВСЕ КАДРЫ ЧЕРНЫЕ - проблема подтверждена!")
        elif black_frames > 0:
            print("⚠️ Есть черные кадры - частичная проблема")
        else:
            print("✅ Черных кадров нет - проблема решена!")
            
        return black_frames == 0
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False
    
    finally:
        # Останавливаем поток
        try:
            requests.post(f"http://localhost:8000/api/stream/stop?stream_id={stream_id}")
            print("🛑 Поток остановлен")
        except:
            pass

def test_static_frame():
    """Тестирует получение статического кадра"""
    print("\n🖼️ Тестируем статический кадр...")
    
    try:
        response = requests.get("http://localhost:8000/api/stream/test_stream/frame", timeout=10)
        
        if response.status_code == 200:
            # Декодируем кадр
            nparr = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                mean_val = np.mean(frame)
                print(f"✅ Статический кадр: {frame.shape}, mean={mean_val:.1f}")
                
                if mean_val < 10:
                    print("⚫ Статический кадр черный")
                    return False
                else:
                    print("✅ Статический кадр нормальный")
                    return True
            else:
                print("❌ Не удается декодировать статический кадр")
                return False
        else:
            print(f"❌ Статический кадр недоступен: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка получения статического кадра: {e}")
        return False

def main():
    print("🚀 ТЕСТ ЧЕРНОГО ЭКРАНА")
    print("=" * 50)
    
    # Проверяем доступность сервера
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print("✅ Сервер доступен")
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        print("💡 Запустите сервер: python main.py")
        return
    
    # Тест 1: MJPEG поток
    mjpeg_ok = test_mjpeg_stream()
    
    # Тест 2: Статический кадр
    static_ok = test_static_frame()
    
    print("\n" + "=" * 50)
    print("📋 ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 50)
    print(f"🎬 MJPEG поток: {'✅ OK' if mjpeg_ok else '❌ ЧЕРНЫЕ КАДРЫ'}")
    print(f"🖼️ Статический кадр: {'✅ OK' if static_ok else '❌ ЧЕРНЫЙ'}")
    
    if mjpeg_ok and static_ok:
        print("\n🎉 ПРОБЛЕМА С ЧЕРНЫМ ЭКРАНОМ РЕШЕНА!")
    elif not mjpeg_ok:
        print("\n❌ Проблема в MJPEG потоке - кадры черные")
        print("💡 Проверьте VideoStream и инференс")
    elif not static_ok:
        print("\n❌ Проблема в статических кадрах")
        print("💡 Проверьте обработку кадров")

if __name__ == "__main__":
    main()
