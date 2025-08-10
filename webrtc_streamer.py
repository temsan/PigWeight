"""
WebRTC Streamer для real-time видео потоков
Низкая задержка и высокая производительность
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Set
import numpy as np
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import socketio
from aiohttp import web, hdrs
from aiohttp.web import middleware
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Конфигурация стрима"""
    video_id: str
    fps: int = 30
    width: int = 1280
    height: int = 720
    bitrate: int = 2000000  # 2 Mbps
    inference_enabled: bool = False
    quality: int = 85

@dataclass
class PeerInfo:
    """Информация о peer соединении"""
    peer_id: str
    peer_connection: RTCPeerConnection
    stream_config: StreamConfig
    connected_at: float
    last_frame_time: float = 0
    frames_sent: int = 0
    data_sent: int = 0

class GPUVideoStreamTrack(VideoStreamTrack):
    """Custom VideoStreamTrack с GPU ускорением"""
    
    def __init__(self, video_processor, stream_config: StreamConfig):
        super().__init__()
        self.video_processor = video_processor
        self.config = stream_config
        self.current_timestamp = 0.0
        self.frame_duration = 1.0 / stream_config.fps
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        # Статистика
        self.stats = {
            'frames_generated': 0,
            'frames_dropped': 0,
            'average_generation_time': 0.0,
            'last_inference_time': 0.0
        }
        
        logger.info(f"GPU VideoStreamTrack created for {stream_config.video_id}")
    
    async def next_frame(self):
        """Генерация следующего кадра"""
        start_time = time.time()
        
        try:
            # Получаем кадр через GPU процессор
            frame = self.video_processor.get_frame_fast(
                self.config.video_id, 
                self.current_timestamp
            )
            
            if frame is None:
                # Если кадр не найден, возвращаемся к началу
                self.current_timestamp = 0.0
                frame = self.video_processor.get_frame_fast(
                    self.config.video_id, 
                    self.current_timestamp
                )
            
            if frame is None:
                # Создаем черный кадр как fallback
                frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            
            # Инференс если включен
            if self.config.inference_enabled:
                inference_start = time.time()
                # Batch inference для лучшей производительности
                results = self.video_processor.batch_inference([frame])
                if results and results[0]:
                    frame = self._draw_detections(frame, results[0])
                self.stats['last_inference_time'] = time.time() - inference_start
            
            # Изменение размера если нужно
            if frame.shape[:2] != (self.config.height, self.config.width):
                frame = cv2.resize(frame, (self.config.width, self.config.height))
            
            # Конвертация цвета BGR -> RGB для WebRTC
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Создаем av.VideoFrame
            from av import VideoFrame
            av_frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            av_frame.pts = self.frame_count
            av_frame.time_base = 1 / self.config.fps
            
            # Обновляем временные метки
            self.current_timestamp += self.frame_duration
            self.frame_count += 1
            
            # Статистика
            generation_time = time.time() - start_time
            self.stats['frames_generated'] += 1
            self.stats['average_generation_time'] = (
                (self.stats['average_generation_time'] * (self.stats['frames_generated'] - 1) + generation_time) 
                / self.stats['frames_generated']
            )
            
            # Контроль FPS
            elapsed = time.time() - self.last_frame_time
            if elapsed < self.frame_duration:
                await asyncio.sleep(self.frame_duration - elapsed)
            
            self.last_frame_time = time.time()
            
            return av_frame
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            self.stats['frames_dropped'] += 1
            
            # Fallback черный кадр
            black_frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            from av import VideoFrame
            av_frame = VideoFrame.from_ndarray(black_frame, format='rgb24')
            av_frame.pts = self.frame_count
            av_frame.time_base = 1 / self.config.fps
            
            return av_frame
    
    def _draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Отрисовка результатов детекции на кадре"""
        try:
            # Пример отрисовки (замените на вашу логику)
            height, width = frame.shape[:2]
            
            # Добавляем информацию о детекции
            info_text = f"Detections: {detections.get('detections', 0)}"
            confidence_text = f"Conf: {detections.get('confidence', 0.0):.2f}"
            
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Пример: рисуем случайные bbox для демонстрации
            num_detections = detections.get('detections', 0)
            if isinstance(num_detections, (int, float)) and num_detections > 0:
                for i in range(min(int(num_detections), 5)):  # Максимум 5 bbox
                    x1 = int(np.random.uniform(0, width * 0.7))
                    y1 = int(np.random.uniform(0, height * 0.7))
                    x2 = int(x1 + np.random.uniform(50, 200))
                    y2 = int(y1 + np.random.uniform(50, 150))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Pig {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return frame

class WebRTCStreamer:
    """WebRTC стример с GPU ускорением"""
    
    def __init__(self, video_processor):
        self.video_processor = video_processor
        self.peers: Dict[str, PeerInfo] = {}
        self.active_streams: Dict[str, StreamConfig] = {}
        self.relay = MediaRelay()
        
        # Socket.IO для сигнализации
        self.sio = socketio.AsyncServer(cors_allowed_origins="*")
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # Статистика
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_data_sent': 0,
            'streams_created': 0
        }
        
        self._setup_routes()
        self._setup_socketio_handlers()
        
        logger.info("WebRTC Streamer initialized")
    
    def _setup_routes(self):
        """Настройка HTTP маршрутов"""
        
        @middleware
        async def cors_handler(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        self.app.middlewares.append(cors_handler)
        
        # Статичные файлы для WebRTC клиента
        self.app.router.add_get('/', self._serve_index)
        self.app.router.add_get('/webrtc', self._serve_webrtc_client)
        self.app.router.add_get('/stats', self._get_webrtc_stats)
        
        # API для управления стримами
        self.app.router.add_post('/api/webrtc/stream/start', self._start_stream)
        self.app.router.add_post('/api/webrtc/stream/stop', self._stop_stream)
        self.app.router.add_get('/api/webrtc/streams', self._list_streams)
    
    def _setup_socketio_handlers(self):
        """Настройка Socket.IO обработчиков"""
        
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"WebRTC client connected: {sid}")
            self.stats['total_connections'] += 1
            self.stats['active_connections'] += 1
        
        @self.sio.event
        async def disconnect(sid):
            logger.info(f"WebRTC client disconnected: {sid}")
            self.stats['active_connections'] -= 1
            
            # Закрываем peer connection если есть
            await self._cleanup_peer(sid)
        
        @self.sio.event
        async def offer(sid, data):
            """Обработка WebRTC offer"""
            try:
                peer_id = sid
                stream_config = StreamConfig(**data.get('config', {}))
                
                # Создаем peer connection
                pc = RTCPeerConnection()
                
                # Добавляем видео трек
                video_track = GPUVideoStreamTrack(self.video_processor, stream_config)
                pc.addTrack(video_track)
                
                # Обработчики событий
                @pc.on("connectionstatechange")
                async def on_connectionstatechange():
                    logger.info(f"Connection state: {pc.connectionState}")
                    if pc.connectionState == "closed":
                        await self._cleanup_peer(peer_id)
                
                @pc.on("datachannel")
                def on_datachannel(channel):
                    logger.info(f"Data channel: {channel.label}")
                
                # Устанавливаем remote description
                offer_sdp = RTCSessionDescription(
                    sdp=data['sdp'], 
                    type=data['type']
                )
                await pc.setRemoteDescription(offer_sdp)
                
                # Создаем answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                # Сохраняем peer info
                peer_info = PeerInfo(
                    peer_id=peer_id,
                    peer_connection=pc,
                    stream_config=stream_config,
                    connected_at=time.time()
                )
                self.peers[peer_id] = peer_info
                
                # Отправляем answer
                await self.sio.emit('answer', {
                    'sdp': pc.localDescription.sdp,
                    'type': pc.localDescription.type
                }, room=sid)
                
                logger.info(f"WebRTC stream started for {stream_config.video_id}")
                self.stats['streams_created'] += 1
                
            except Exception as e:
                logger.error(f"Error handling offer: {e}")
                await self.sio.emit('error', {'message': str(e)}, room=sid)
        
        @self.sio.event
        async def ice_candidate(sid, data):
            """Обработка ICE кандидатов"""
            try:
                peer_info = self.peers.get(sid)
                if peer_info and peer_info.peer_connection:
                    from aiortc import RTCIceCandidate
                    candidate = RTCIceCandidate(
                        component=data['candidate']['component'],
                        foundation=data['candidate']['foundation'],
                        ip=data['candidate']['ip'],
                        port=data['candidate']['port'],
                        priority=data['candidate']['priority'],
                        protocol=data['candidate']['protocol'],
                        type=data['candidate']['type']
                    )
                    await peer_info.peer_connection.addIceCandidate(candidate)
                    
            except Exception as e:
                logger.error(f"Error handling ICE candidate: {e}")
    
    async def _cleanup_peer(self, peer_id: str):
        """Очистка peer connection"""
        if peer_id in self.peers:
            peer_info = self.peers[peer_id]
            try:
                await peer_info.peer_connection.close()
            except:
                pass
            del self.peers[peer_id]
            logger.info(f"Peer {peer_id} cleaned up")
    
    async def _serve_index(self, request):
        """Главная страница"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>WebRTC Video Streamer</title>
        </head>
        <body>
            <h1>WebRTC Video Streamer</h1>
            <p><a href="/webrtc">Open WebRTC Client</a></p>
            <p><a href="/stats">View Statistics</a></p>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def _serve_webrtc_client(self, request):
        """WebRTC клиент"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>WebRTC Client</title>
            <script src="/socket.io/socket.io.js"></script>
        </head>
        <body>
            <h1>WebRTC Video Stream</h1>
            <div>
                <label>Video ID: <input type="text" id="videoId" value="ultra_video"></label>
                <label>FPS: <input type="number" id="fps" value="30" min="1" max="60"></label>
                <label>Resolution: 
                    <select id="resolution">
                        <option value="640x480">640x480</option>
                        <option value="1280x720" selected>1280x720</option>
                        <option value="1920x1080">1920x1080</option>
                    </select>
                </label>
                <label><input type="checkbox" id="inference"> Enable AI Inference</label>
                <br><br>
                <button onclick="startStream()">Start Stream</button>
                <button onclick="stopStream()">Stop Stream</button>
            </div>
            <br>
            <video id="videoElement" autoplay muted style="width: 100%; max-width: 800px;"></video>
            <div id="stats"></div>
            
            <script>
                const socket = io();
                let pc = null;
                let stream = null;
                
                socket.on('answer', async (data) => {
                    if (pc) {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    }
                });
                
                socket.on('error', (data) => {
                    console.error('WebRTC Error:', data.message);
                    alert('Error: ' + data.message);
                });
                
                async function startStream() {
                    try {
                        // Создаем RTCPeerConnection
                        pc = new RTCPeerConnection({
                            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                        });
                        
                        // Обработчик удаленного стрима
                        pc.ontrack = (event) => {
                            console.log('Received remote stream');
                            const videoElement = document.getElementById('videoElement');
                            videoElement.srcObject = event.streams[0];
                        };
                        
                        // Обработчик ICE кандидатов
                        pc.onicecandidate = (event) => {
                            if (event.candidate) {
                                socket.emit('ice_candidate', {
                                    candidate: event.candidate
                                });
                            }
                        };
                        
                        // Получаем конфигурацию
                        const videoId = document.getElementById('videoId').value;
                        const fps = parseInt(document.getElementById('fps').value);
                        const resolution = document.getElementById('resolution').value.split('x');
                        const inference = document.getElementById('inference').checked;
                        
                        const config = {
                            video_id: videoId,
                            fps: fps,
                            width: parseInt(resolution[0]),
                            height: parseInt(resolution[1]),
                            inference_enabled: inference
                        };
                        
                        // Создаем offer
                        const offer = await pc.createOffer();
                        await pc.setLocalDescription(offer);
                        
                        // Отправляем offer
                        socket.emit('offer', {
                            sdp: offer.sdp,
                            type: offer.type,
                            config: config
                        });
                        
                    } catch (error) {
                        console.error('Error starting stream:', error);
                        alert('Error: ' + error.message);
                    }
                }
                
                function stopStream() {
                    if (pc) {
                        pc.close();
                        pc = null;
                    }
                    const videoElement = document.getElementById('videoElement');
                    videoElement.srcObject = null;
                }
                
                // Статистика
                setInterval(async () => {
                    if (pc) {
                        const stats = await pc.getStats();
                        let statsText = '<h3>WebRTC Stats:</h3>';
                        stats.forEach(report => {
                            if (report.type === 'outbound-rtp' && report.mediaType === 'video') {
                                statsText += `Frames sent: ${report.framesSent}<br>`;
                                statsText += `Bytes sent: ${report.bytesSent}<br>`;
                                statsText += `Bitrate: ${Math.round(report.bytesSent * 8 / report.timestamp * 1000)} bps<br>`;
                            }
                        });
                        document.getElementById('stats').innerHTML = statsText;
                    }
                }, 2000);
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def _get_webrtc_stats(self, request):
        """Статистика WebRTC"""
        stats = {
            'server_stats': self.stats,
            'active_peers': len(self.peers),
            'peer_details': []
        }
        
        for peer_id, peer_info in self.peers.items():
            peer_stats = {
                'peer_id': peer_id,
                'connected_duration': time.time() - peer_info.connected_at,
                'stream_config': asdict(peer_info.stream_config),
                'frames_sent': peer_info.frames_sent,
                'data_sent': peer_info.data_sent
            }
            stats['peer_details'].append(peer_stats)
        
        return web.json_response(stats)
    
    async def _start_stream(self, request):
        """API для запуска стрима"""
        try:
            data = await request.json()
            stream_config = StreamConfig(**data)
            
            # Проверяем что видео существует
            if stream_config.video_id not in self.video_processor.videos:
                return web.json_response(
                    {'error': f'Video {stream_config.video_id} not found'}, 
                    status=404
                )
            
            self.active_streams[stream_config.video_id] = stream_config
            
            return web.json_response({
                'status': 'success',
                'message': f'Stream prepared for {stream_config.video_id}',
                'config': asdict(stream_config)
            })
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def _stop_stream(self, request):
        """API для остановки стрима"""
        try:
            data = await request.json()
            video_id = data.get('video_id')
            
            if video_id in self.active_streams:
                del self.active_streams[video_id]
            
            # Закрываем все peer connections для этого видео
            peers_to_close = []
            for peer_id, peer_info in self.peers.items():
                if peer_info.stream_config.video_id == video_id:
                    peers_to_close.append(peer_id)
            
            for peer_id in peers_to_close:
                await self._cleanup_peer(peer_id)
            
            return web.json_response({
                'status': 'success',
                'message': f'Stream stopped for {video_id}',
                'peers_closed': len(peers_to_close)
            })
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def _list_streams(self, request):
        """Список активных стримов"""
        return web.json_response({
            'active_streams': [asdict(config) for config in self.active_streams.values()],
            'total_peers': len(self.peers)
        })
    
    def run(self, host='0.0.0.0', port=8080):
        """Запуск WebRTC сервера"""
        logger.info(f"Starting WebRTC server on {host}:{port}")
        web.run_app(self.app, host=host, port=port)

# Глобальный экземпляр стримера
webrtc_streamer = None

def get_webrtc_streamer(video_processor=None):
    """Получение singleton экземпляра WebRTC стримера"""
    global webrtc_streamer
    if webrtc_streamer is None:
        if video_processor is None:
            from gpu_video_processor import get_gpu_processor
            video_processor = get_gpu_processor()
        webrtc_streamer = WebRTCStreamer(video_processor)
    return webrtc_streamer

if __name__ == "__main__":
    # Тестирование
    from gpu_video_processor import get_gpu_processor
    
    processor = get_gpu_processor()
    streamer = get_webrtc_streamer(processor)
    
    print("WebRTC Streamer starting...")
    streamer.run(port=8080)
