import os
import sys
import urllib.request
import subprocess
import logging
from pathlib import Path

# Load .env early so DEBUG and other vars are available here
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config from environment ---
DETECTION_MODE = os.getenv("DETECTION_MODE", "balanced")
PIG_MODEL_PATH = os.getenv("PIG_MODEL_PATH", "models/pig_yolo11-seg.pt")
BALANCED_MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo11n.pt")
ONNX_PATH = os.getenv("ONNX_PATH", "models/yolo11n.onnx")

# Set model path based on detection mode
if DETECTION_MODE == "pig-only":
    MODEL_PATH = PIG_MODEL_PATH
else:
    MODEL_PATH = BALANCED_MODEL_PATH

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

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
        # Ensure required directories exist
        ensure_dir('models')
        ensure_dir('stream')
        ensure_dir('uploads')

        # Import ASGI app and start server
        logger.info(f'Starting server at http://{HOST}:{PORT}')
        logger.info(f'API Health Check: http://{HOST}:{PORT}/api/health')
        logger.info(f'Debug mode: {DEBUG}')

        try:
            import uvicorn
            if DEBUG:
                # Для reload uvicorn требует import string
                uvicorn.run(
                    "api.app:app",
                    host=HOST,
                    port=PORT,
                    reload=True,
                    log_level="debug"
                )
            else:
                from api.app import app as fastapi_app
                uvicorn.run(
                    fastapi_app,
                    host=HOST,
                    port=PORT,
                    reload=False,
                    log_level="info"
                )
        except Exception as e:
            logger.error(f'Error starting server via uvicorn: {str(e)}')
            raise
    except Exception as e:
        logger.error(f'Error starting server: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()
