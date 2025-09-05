#!/usr/bin/env python3
"""
Быстрый тест WebRTC и отображения масок
"""

import asyncio
import aiohttp
import json
import time

async def test_webrtc_connection():
    """Тест WebRTC соединения"""
    print("🧪 Тестирование WebRTC соединения...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Проверяем доступность сервера
            async with session.get('http://localhost:8000/health') as resp:
                if resp.status == 200:
                    print("✅ Сервер доступен")
                else:
                    print(f"❌ Сервер недоступен: {resp.status}")
                    return False
            
            # Проверяем WebSocket соединение
            async with session.ws_connect('ws://localhost:8000/ws/cam101') as ws:
                print("✅ WebSocket соединение установлено")
                
                # Ждем данные в течение 5 секунд
                timeout = 5
                start_time = time.time()
                messages_received = 0
                
                while time.time() - start_time < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            messages_received += 1
                            print(f"📨 Получено сообщение #{messages_received}: {data.get('type', 'unknown')}")
                            
                            # Проверяем наличие данных о детекции
                            if 'count' in data:
                                print(f"🎯 Детекция: {data['count']} объектов")
                            if 'confidence' in data.get('debug', {}):
                                print(f"🎯 Уверенность: {data['debug']['confidence']:.3f}")
                                
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print(f"❌ WebSocket ошибка: {ws.exception()}")
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                        
                print(f"📊 Получено сообщений: {messages_received}")
                return messages_received > 0
                
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

async def test_webrtc_offer():
    """Тест WebRTC offer/answer"""
    print("\n🧪 Тестирование WebRTC offer/answer...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Простой тестовый offer
            test_offer = {
                "type": "offer",
                "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=video 0 RTP/AVP 96\r\nc=IN IP4 127.0.0.1\r\na=rtpmap:96 H264/90000\r\n"
            }
            
            async with session.post(
                'http://localhost:8000/webrtc/offer', 
                json={'offer': test_offer, 'stream_id': 'cam101'}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("✅ WebRTC offer принят")
                    if 'answer' in data:
                        print("✅ WebRTC answer получен")
                        return True
                    else:
                        print("❌ WebRTC answer не получен")
                        return False
                else:
                    print(f"❌ WebRTC offer отклонен: {resp.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Ошибка WebRTC тестирования: {e}")
        return False

async def main():
    print("🚀 Тестирование PigWeight WebRTC и масок...")
    print("=" * 50)
    
    # Ждем запуска сервера
    print("⏳ Ожидание запуска сервера (10 сек)...")
    await asyncio.sleep(10)
    
    # Тест WebSocket соединения
    ws_ok = await test_webrtc_connection()
    
    # Тест WebRTC offer
    webrtc_ok = await test_webrtc_offer()
    
    print("\n" + "=" * 50)
    print("📊 Результаты тестирования:")
    print(f"   WebSocket: {'✅ OK' if ws_ok else '❌ FAIL'}")
    print(f"   WebRTC:    {'✅ OK' if webrtc_ok else '❌ FAIL'}")
    
    if ws_ok and webrtc_ok:
        print("\n🎉 Все тесты прошли успешно!")
        print("🌐 Откройте http://localhost:8000 в браузере")
        print("📹 WebRTC видео должно работать")
        print("🎭 Маски детекции должны отображаться")
    else:
        print("\n⚠️  Обнаружены проблемы. Проверьте логи сервера.")
    
    return ws_ok and webrtc_ok

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
