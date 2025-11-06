"""
WebSocket endpoints
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import asyncio
import time

router = APIRouter(tags=["websocket"])

# Global limits and throttling
MAX_WEBSOCKET_CONNECTIONS = 10
WEBSOCKET_MAX_FPS = 10
WEBSOCKET_FRAME_INTERVAL = 1.0 / WEBSOCKET_MAX_FPS
_current_ws_connections = 0

@router.websocket("/ws/count")
async def websocket_endpoint(websocket: WebSocket, id: str = Query(...)):
    """WebSocket endpoint для получения данных подсчета"""
    global _current_ws_connections

    # Capacity check
    if _current_ws_connections >= MAX_WEBSOCKET_CONNECTIONS:
        await websocket.close(code=1008, reason="Server at capacity")
        return

    await websocket.accept()
    _current_ws_connections += 1
    
    try:
        # Простейший цикл с троттлингом отправки (не чаще 10 fps)
        last_send_ts = 0.0
        while True:
            # Ожидаем сообщения от клиента (пинг/управление), но не блокируем отправку
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=WEBSOCKET_FRAME_INTERVAL)
            except asyncio.TimeoutError:
                pass

            now = time.time()
            if now - last_send_ts < WEBSOCKET_FRAME_INTERVAL:
                # Слишком часто — подождём до следующего слота
                await asyncio.sleep(max(0.0, WEBSOCKET_FRAME_INTERVAL - (now - last_send_ts)))
                now = time.time()

            # Отправляем тестовые данные (заглушка)
            await websocket.send_json({
                "type": "count_update",
                "stream_id": id,
                "count": 0,
                "left_in": 0,
                "right_in": 0,
                "timestamp": now
            })
            last_send_ts = now
            
    except WebSocketDisconnect:
        # Клиент отключился
        pass
    except Exception as e:
        # Ошибка в WebSocket соединении
        try:
            await websocket.close()
        except:
            pass
    finally:
        _current_ws_connections = max(0, _current_ws_connections - 1)