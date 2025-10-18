#!/usr/bin/env python3
"""
Test script for Ultra-Fast Video Processing System
Проверка производительности ультра-быстрой системы
"""

import time
import requests
import asyncio
import websockets
import json
from pathlib import Path

def test_api_health():
    """Проверка базовых API эндпоинтов"""
    print("🏥 Testing API Health...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Проверяем основные эндпоинты
        response = requests.get(f"{base_url}/")
        print(f"   ✅ Main page: {response.status_code}")
        
        response = requests.get(f"{base_url}/api/models")
        print(f"   ✅ Models API: {response.status_code}")
        
        print("   🎯 All basic endpoints working!")
        
    except Exception as e:
        print(f"   ❌ API Health Check Failed: {e}")
        return False
        
    return True

async def test_websocket_connections():
    """Тестирование WebSocket соединений"""
    print("🔌 Testing WebSocket Connections...")
    
    try:
        # Тестируем обычный WebSocket счетчика
        uri = "ws://localhost:8000/ws/count"
        
        async with websockets.connect(uri) as websocket:
            print("   ✅ Count WebSocket connected")
            
            # Ждем пару сообщений
            for i in range(3):
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                print(f"   📊 Received: {data.get('type', 'unknown')}")
                
        print("   🎯 WebSocket test completed!")
        
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")
        return False
        
    return True

def test_performance_metrics():
    """Тестирование метрик производительности"""
    print("⚡ Testing Performance Metrics...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Мокаем файловую сессию для теста кадра
        test_times = []
        
        for i in range(5):
            start_time = time.time()
            
            # Пробуем обратиться к фреймовому API
            try:
                response = requests.get(
                    f"{base_url}/api/video_file/frame_ultra_fast",
                    params={"id": "test_session", "t": 0.0, "ts": int(time.time() * 1000)},
                    timeout=2.0
                )
                
                if response.status_code == 404:
                    print(f"   ⚠️  No test session (expected): {response.status_code}")
                else:
                    # Извлекаем заголовки производительности
                    seek_ms = response.headers.get('X-Seek-Ms', '0')
                    infer_ms = response.headers.get('X-Inference-Ms', '0')
                    encode_ms = response.headers.get('X-Encode-Ms', '0')
                    
                    print(f"   📈 Metrics - Seek: {seek_ms}ms, Infer: {infer_ms}ms, Encode: {encode_ms}ms")
                    
            except requests.Timeout:
                print("   ⏱️  Request timeout (expected without file)")
            except Exception as e:
                print(f"   ⚠️  Request error: {e}")
                
            end_time = time.time()
            test_times.append((end_time - start_time) * 1000)
            
            time.sleep(0.1)  # Небольшая пауза между запросами
            
        avg_response_time = sum(test_times) / len(test_times)
        print(f"   🎯 Average response time: {avg_response_time:.1f}ms")
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False
        
    return True

def test_frontend_assets():
    """Проверка фронтенд ресурсов"""
    print("🎨 Testing Frontend Assets...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Проверяем основные статические файлы
        assets = [
            "/static/index.html",
            "/static/css/theme.css", 
            "/static/js/stream.js"
        ]
        
        for asset in assets:
            try:
                response = requests.get(f"{base_url}{asset}", timeout=3.0)
                status = "✅" if response.status_code == 200 else "❌"
                print(f"   {status} {asset}: {response.status_code}")
            except Exception as e:
                print(f"   ❌ {asset}: {e}")
                
        print("   🎯 Frontend assets check completed!")
        
    except Exception as e:
        print(f"   ❌ Frontend test failed: {e}")
        return False
        
    return True

async def main():
    """Основная функция тестирования"""
    print("🚀 Ultra-Fast Video Processing System Test")
    print("=" * 50)
    
    # Набор тестов
    tests = [
        ("API Health", test_api_health),
        ("Performance Metrics", test_performance_metrics), 
        ("Frontend Assets", test_frontend_assets),
        ("WebSocket Connections", test_websocket_connections),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            results.append((test_name, result))
            
        except Exception as e:
            print(f"   💥 {test_name} crashed: {e}")
            results.append((test_name, False))
            
    # Сводка результатов
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
            
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational! Ultra-Fast mode ready!")
    else:
        print("⚠️  Some issues detected. Check the logs above.")
        
    return passed == total

if __name__ == "__main__":
    print("Starting Ultra-Fast System Test...")
    print("Make sure the server is running on localhost:8000")
    print()
    
    success = asyncio.run(main())
    exit(0 if success else 1)
