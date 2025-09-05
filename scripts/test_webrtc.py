#!/usr/bin/env python3
import requests
import json

def test_webrtc_offer():
    """Тестируем WebRTC offer endpoint"""
    print("=== Testing WebRTC Offer ===")

    # Создаем тестовый SDP offer
    offer_sdp = """v=0
o=- 123456789 123456789 IN IP4 127.0.0.1
s=Test
c=IN IP4 127.0.0.1
t=0 0
m=video 9 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
a=sendrecv
"""

    payload = {
        "sdp": offer_sdp,
        "type": "offer",
        "stream_id": "cam101",
        "fps": 12
    }

    try:
        print("Отправляем WebRTC offer...")
        response = requests.post('http://localhost:8000/api/webrtc/offer',
                               json=payload,
                               headers={'Content-Type': 'application/json'},
                               timeout=10)

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ WebRTC offer принят!")
            print(f"Peer ID: {data.get('peer_id')}")
            print(f"Answer SDP: {data.get('sdp')[:100]}...")
        else:
            print(f"❌ Ошибка: {response.text}")

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")

if __name__ == "__main__":
    test_webrtc_offer()
