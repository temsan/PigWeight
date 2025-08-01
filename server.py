from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl
import os
import sys
import urllib.request
import subprocess
import torch

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

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.end_headers()

def main():
    # Download and convert model
    try:
        download_model()
        convert_to_onnx()
    except Exception as e:
        print(f"Error preparing model: {str(e)}")
        return

    # Start HTTP server
    port = 8000
    httpd = HTTPServer(('0.0.0.0', port), CORSRequestHandler)
    print(f'Server started at http://localhost:{port}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    main()
