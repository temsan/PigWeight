import os
import sys
import urllib.request
import subprocess
import logging
from core.config import MODEL_URL, MODEL_PATH, ONNX_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            sys.__executable__ if hasattr(sys, "__executable__") else sys.executable,
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
        
        # Import ultralytics and load model
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            
            # Export to ONNX
            model.export(format='onnx', opset=12)
            
            # Rename the exported file to our target name
            default_onnx = MODEL_PATH.replace('.pt', '.onnx')
            if os.path.exists(default_onnx) and default_onnx != ONNX_PATH:
                os.replace(default_onnx, ONNX_PATH)
                
            print("Model converted to ONNX successfully")
        except ImportError as e:
            print(f"Error importing ultralytics: {str(e)}")
            print("Please install ultralytics package")
            raise
        except Exception as e:
            print(f"Error converting model to ONNX: {str(e)}")
            raise
    else:
        print(f"ONNX model already exists at {ONNX_PATH} and is up to date")

def main():
    try:
        # Initialize models and requirements
        install_requirements()
        download_model()
        convert_to_onnx()
        
        # Ensure required directories exist
        ensure_dir('models')
        ensure_dir('stream')
        
        # Import ASGI app after all requirements are installed
        # Перешли на FastAPI — запуск через uvicorn
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        logger.info(f'Starting server at http://{host}:{port}')
        logger.info(f'API Health Check: http://{host}:{port}/api/health')
        try:
            import uvicorn
            # Запускаем уже импортированный объект приложения для явной связки процесса
            from api.app import app as fastapi_app
            uvicorn.run(fastapi_app, host=host, port=port, reload=False, log_level="info")
        except Exception as e:
            logger.error(f'Error starting server via uvicorn: {str(e)}')
            raise
    except Exception as e:
        logger.error(f'Error starting server: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()
