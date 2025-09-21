"""
WebSocket endpoints
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

router = APIRouter(tags=["websocket"])

@router.websocket("/ws/count")
async def websocket_endpoint(websocket: WebSocket, id: str = Query(...)):
    """WebSocket endpoint для получения данных подсчета"""
    await websocket.accept()
    
    try:
        # Здесь будет логика WebSocket соединения
        # Пока просто держим соединение открытым
        while True:
            # Ожидаем сообщения от клиента
            await websocket.receive_text()
            
            # Отправляем тестовые данные
            await websocket.send_json({
                "type": "count_update",
                "stream_id": id,
                "count": 0,
                "left_in": 0,
                "right_in": 0,
                "timestamp": 0
            })
            
    except WebSocketDisconnect:
        # Клиент отключился
        pass
    except Exception as e:
        # Ошибка в WebSocket соединении
        try:
            await websocket.close()
        except:
            pass