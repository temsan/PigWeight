import asyncio
import logging
import os
import uuid as _uuid
from typing import Dict, Any

import cv2
import numpy as np
from av import VideoFrame as _VideoFrame
from fastapi import APIRouter, Body, Query, FastAPI, HTTPException
from fastapi.responses import JSONResponse

# aiortc components
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
    _HAVE_AIORTC = True
except ImportError:
    _HAVE_AIORTC = False

router = APIRouter()

_PEER_CONNECTIONS: Dict[str, RTCPeerConnection] = {}

def init_webrtc(app: FastAPI, stream_manager: Any, frame_broker: Any, config: Any):
    logger = logging.getLogger(__name__)
    perf_logger = logging.getLogger("perf.webrtc")

    class BrokerVideoTrack(VideoStreamTrack):
        def __init__(self, stream_id: str, fps: float = 15.0):
            super().__init__()
            self.stream_id = stream_id
            self.fps = float(fps)
            self.frame_duration = 1.0 / max(1.0, self.fps)
            self._last_good_jpeg: bytes | None = None

        async def recv(self):
            jpeg = None
            source = "none"

            if frame_broker is not None:
                latest = frame_broker.get_latest(self.stream_id)
                if latest and latest.get('jpeg'):
                    jpeg_data = latest.get('jpeg')
                    source = "FRAME_BROKER"
                    if not isinstance(jpeg_data, bytes):
                        logger.warning(f"BrokerVideoTrack {self.stream_id}: FRAME_BROKER returned non-bytes jpeg: {type(jpeg_data)}")
                        jpeg_data = None
                else:
                    logger.debug(f"BrokerVideoTrack {self.stream_id}: FRAME_BROKER returned empty or no jpeg")

            if jpeg is None:
                stream = stream_manager.streams.get(self.stream_id)
                if stream:
                    try:
                        jpeg = await stream.get_jpeg()
                        source = "StreamManager"
                    except Exception as e:
                        logger.warning(f"BrokerVideoTrack {self.stream_id}: StreamManager.get_jpeg failed: {e}")

            if jpeg:
                self._last_good_jpeg = jpeg
            elif self._last_good_jpeg:
                jpeg = self._last_good_jpeg
                source = f"cached:{source}"
            else:
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                vf = _VideoFrame.from_ndarray(black, format='bgr24')
                pts, time_base = await self.next_timestamp()
                vf.pts = pts
                vf.time_base = time_base
                return vf

            if isinstance(jpeg, dict):
                jpeg_data = jpeg.get('jpeg')
            else:
                jpeg_data = jpeg
            
            if not jpeg_data:
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                vf = _VideoFrame.from_ndarray(black, format='bgr24')
                pts, time_base = await self.next_timestamp()
                vf.pts = pts
                vf.time_base = time_base
                return vf

            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)

            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                img_rgb = img

            vf = _VideoFrame.from_ndarray(img_rgb, format='rgb24')
            pts, time_base = await self.next_timestamp()
            vf.pts = pts
            vf.time_base = time_base
            return vf

    @router.post('/offer')
    async def api_webrtc_offer(payload: Dict[str, Any]):
        if not _HAVE_AIORTC:
            return JSONResponse({'error': 'aiortc not available'}, status_code=500)
        try:
            sdp = payload.get('sdp')
            typ = payload.get('type')
            stream_id = payload.get('stream_id')
            fps = float(payload.get('fps') or config['TARGET_FPS'])
            
            if not all([sdp, typ, stream_id]):
                return JSONResponse({'error': 'invalid payload'}, status_code=400)

            pc = RTCPeerConnection()
            peer_id = _uuid.uuid4().hex
            _PEER_CONNECTIONS[peer_id] = pc

            @pc.on('connectionstatechange')
            async def on_state():
                if pc.connectionState == 'failed' or pc.connectionState == 'closed':
                    if peer_id in _PEER_CONNECTIONS:
                        await _PEER_CONNECTIONS.pop(peer_id).close()

            offer = RTCSessionDescription(sdp=sdp, type=typ)
            await pc.setRemoteDescription(offer)
            
            track = BrokerVideoTrack(stream_id=stream_id, fps=fps)
            pc.addTrack(track)
            
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                'peer_id': peer_id,
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            }
        except Exception as e:
            logger.exception(f"Error handling WebRTC offer for stream_id={payload.get('stream_id')}")
            return JSONResponse({'error': str(e)}, status_code=500)

    @router.post('/candidate')
    async def api_webrtc_candidate(body: Dict[str, Any]):
        peer_id = body.get('peer_id')
        candidate = body.get('candidate')
        pc = _PEER_CONNECTIONS.get(peer_id)
        if not pc:
            return JSONResponse({'error': 'peer not found'}, status_code=404)
        
        await pc.addIceCandidate(RTCIceCandidate(**candidate))
        return {'status': 'ok'}

    @router.post('/stop')
    async def api_webrtc_stop(body: Dict[str, Any]):
        peer_id = body.get('peer_id')
        pc = _PEER_CONNECTIONS.pop(peer_id, None)
        if pc:
            await pc.close()
        return {'status': 'stopped'}

    app.include_router(router, prefix="/api/webrtc", tags=["webrtc"])