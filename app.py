from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from rtsp_manager import RTSPManager
from server import install_requirements, download_model, convert_to_onnx

app = Flask(__name__)
CORS(app)
rtsp_manager = RTSPManager()

@app.route('/start_stream', methods=['POST'])
def start_stream():
    data = request.json
    camera_id = data.get('camera_id')
    rtsp_url = data.get('rtsp_url')
    
    if not camera_id or not rtsp_url:
        return jsonify({'error': 'Missing camera_id or rtsp_url'}), 400
    
    stream_path = rtsp_manager.start_stream(camera_id, rtsp_url)
    if stream_path:
        return jsonify({'stream_url': stream_path})
    return jsonify({'error': 'Failed to start stream'}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    data = request.json
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({'error': 'Missing camera_id'}), 400
    
    rtsp_manager.stop_stream(camera_id)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Инициализация модели и другие подготовительные действия
    install_requirements()
    download_model()
    convert_to_onnx()
    
    # Запуск сервера
    app.run(host='0.0.0.0', port=5000)
