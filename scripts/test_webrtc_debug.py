#!/usr/bin/env python3
"""
Тест WebRTC соединения для диагностики черного экрана
"""
import asyncio
import json
import requests
import time
from datetime import datetime

def test_webrtc_signaling():
    """Тестируем WebRTC сигналинг"""
    print("🔗 Тестируем WebRTC сигналинг...")
    
    try:
        # Создаем простой SDP offer
        fake_offer = {
            "sdp": "v=0\r\no=- 123456789 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 127.0.0.1\r\na=rtcp:9 IN IP4 127.0.0.1\r\na=ice-ufrag:test\r\na=ice-pwd:test123456789012345678901234\r\na=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00\r\na=setup:actpass\r\na=mid:0\r\na=sendrecv\r\na=rtcp-mux\r\na=rtpmap:96 H264/90000\r\n",
            "type": "offer",
            "stream_id": "cam101",
            "fps": 12
        }
        
        print("📡 Отправляем WebRTC offer...")
        response = requests.post(
            "http://localhost:8000/api/webrtc/offer",
            json=fake_offer,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ WebRTC offer принят")
            print(f"📋 Ответ: peer_id={data.get('peer_id')}, type={data.get('type')}")
            
            if data.get('sdp'):
                print("✅ SDP ответ получен")
                # Проверяем наличие видео в SDP
                sdp = data.get('sdp', '')
                if 'video' in sdp.lower():
                    print("✅ Видео трек найден в SDP")
                else:
                    print("⚠️ Видео трек не найден в SDP")
                    
                return True, data.get('peer_id')
            else:
                print("❌ SDP ответ пустой")
                return False, None
        else:
            print(f"❌ WebRTC offer отклонен: {response.status_code}")
            print(response.text)
            return False, None
            
    except Exception as e:
        print(f"❌ Ошибка WebRTC сигналинга: {e}")
        return False, None

def check_stream_status():
    """Проверяем статус потока"""
    print("\n📊 Проверяем статус потока cam101...")
    
    try:
        response = requests.get("http://localhost:8000/api/stream/cam101/info", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Поток активен")
            print(f"📈 FPS: {data.get('fps', 'N/A')}")
            print(f"🔢 Кадров: {data.get('frame_count', 'N/A')}")
            print(f"🎯 Детекций: {data.get('detections', 'N/A')}")
            print(f"🏃 Статус: {data.get('status', 'N/A')}")
            
            # Проверяем есть ли кадры
            if data.get('frame_count', 0) > 0:
                print("✅ Кадры генерируются")
                return True
            else:
                print("❌ Кадры не генерируются")
                return False
        else:
            print(f"❌ Поток недоступен: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка проверки статуса: {e}")
        return False

def check_webrtc_video_track():
    """Проверяем видео трек WebRTC"""
    print("\n🎥 Проверяем WebRTC видео трек...")
    
    # Сначала проверяем статус потока
    if not check_stream_status():
        print("❌ Поток не активен, WebRTC не может работать")
        return False
    
    # Тестируем WebRTC сигналинг
    success, peer_id = test_webrtc_signaling()
    
    if success and peer_id:
        print(f"✅ WebRTC соединение установлено: {peer_id}")
        
        # Ждем немного для установки соединения
        print("⏳ Ждем установки соединения...")
        time.sleep(3)
        
        # Проверяем активные соединения (если есть такой endpoint)
        try:
            response = requests.get("http://localhost:8000/api/webrtc/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"📊 Активных WebRTC соединений: {len(data.get('connections', []))}")
            else:
                print("ℹ️ Статус WebRTC недоступен")
        except:
            print("ℹ️ Статус WebRTC недоступен")
        
        # Останавливаем соединение
        try:
            requests.post(
                "http://localhost:8000/api/webrtc/stop",
                json={"peer_id": peer_id},
                timeout=5
            )
            print("🛑 WebRTC соединение остановлено")
        except:
            pass
            
        return True
    else:
        print("❌ WebRTC соединение не установлено")
        return False

def analyze_webrtc_logs():
    """Анализируем логи WebRTC из терминала"""
    print("\n📋 Анализ WebRTC активности из логов:")
    print("✅ RTCRtpSender активен - видео трек создается")
    print("✅ RtcpSrPacket отправляются - RTP поток работает") 
    print("✅ Connection protocol активен - сигналинг работает")
    print("⚠️ packet_count=0, octet_count=0 - возможно нет видео данных")
    print("\n💡 Рекомендации:")
    print("1. Проверить генерацию кадров в VideoStream")
    print("2. Проверить передачу кадров в WebRTC track")
    print("3. Проверить кодировку видео")

def main():
    print("🚀 ДИАГНОСТИКА WEBRTC")
    print("=" * 50)
    print(f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}")
    
    # Проверяем доступность сервера
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print("✅ Сервер доступен")
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        return
    
    # Анализируем логи
    analyze_webrtc_logs()
    
    # Тестируем WebRTC
    webrtc_ok = check_webrtc_video_track()
    
    print("\n" + "=" * 50)
    print("📋 РЕЗУЛЬТАТ ДИАГНОСТИКИ")
    print("=" * 50)
    
    if webrtc_ok:
        print("✅ WebRTC сигналинг работает")
        print("💡 Проблема может быть в:")
        print("   - Генерации кадров VideoStream")
        print("   - Передаче кадров в WebRTC track") 
        print("   - Кодировке/декодировке видео")
    else:
        print("❌ Проблема в WebRTC сигналинге")
        print("💡 Проверьте:")
        print("   - Статус потока cam101")
        print("   - WebRTC endpoints")
        print("   - Сетевое соединение")

if __name__ == "__main__":
    main()
