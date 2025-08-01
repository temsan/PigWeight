from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import ssl
import os
import sys
import urllib.request
import subprocess
import torch
from rtsp_manager import RTSPManager

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
MODEL_PATH = "models/yolo11n.pt"
ONNX_PATH = "models/yolo11n.onnx"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model to {MODEL_PATH}...")
        ensure_dir(os.path.dirname(MODEL_PATH))
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully")
    else:
        print(f"Model already exists at {MODEL_PATH}")

def install_requirements():
    print("Installing required packages from requirements.txt...")
    try:
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if not os.path.exists(requirements_path):
            print("Error: requirements.txt not found!")
            raise FileNotFoundError("requirements.txt not found")
            
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_path
        ])
        print("All required packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {str(e)}")
        raise

def convert_to_onnx():
    if not os.path.exists(ONNX_PATH) or os.path.getmtime(MODEL_PATH) > os.path.getmtime(ONNX_PATH):
        print(f"Converting model to ONNX format...")
        ensure_dir(os.path.dirname(ONNX_PATH))
        
        # Install required packages if not already installed
        install_requirements()
        
        # Load YOLOv5 model
        import torch
        from ultralytics import YOLO
        
        model = YOLO(MODEL_PATH)
        
        # Export to ONNX
        model.export(format='onnx', opset=12)
        
        # Rename the exported file to our target name
        default_onnx = MODEL_PATH.replace('.pt', '.onnx')
        if os.path.exists(default_onnx) and default_onnx != ONNX_PATH:
            os.replace(default_onnx, ONNX_PATH)
            
        print("Model converted to ONNX successfully")
    else:
        print(f"ONNX model already exists at {ONNX_PATH} and is up to date")

def create_app():
    app = Flask(__name__, static_folder='PigWeight', static_url_path='')
    CORS(app)
    rtsp_manager = RTSPManager()

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.route('/api/stream/start', methods=['POST'])
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

    @app.route('/api/stream/stop', methods=['POST'])
    def stop_stream():
        data = request.json
        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({'error': 'Missing camera_id'}), 400
        rtsp_manager.stop_stream(camera_id)
        return jsonify({'status': 'success'})

    @app.route('/stream/<path:filename>')
    def serve_stream(filename):
        return send_from_directory('stream', filename)

    return app

def main():
    try:
        install_requirements()
        download_model()
        convert_to_onnx()
    except Exception as e:
        print(f"Error preparing model: {str(e)}")
        return
    print('Server starting at http://localhost:8000')
    app = create_app()
    app.run(host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
