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
DETECTION_MODE = os.getenv("DETECTION_MODE", "pig-only")
# Единый путь до модели берём из MODEL_PATH; PIG_MODEL_PATH поддерживается как Legacy
MODEL_PATH_ENV = os.getenv("MODEL_PATH")
PIG_MODEL_PATH = os.getenv("PIG_MODEL_PATH")
BALANCED_MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo11n.pt")
ONNX_PATH = os.getenv("ONNX_PATH", "models/yolo11n.onnx")
MODEL_URL = os.getenv("MODEL_URL", "")

# Debug and hot-reload settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
RELOAD = os.getenv("RELOAD", "true" if DEBUG else "false").lower() == "true"

# Set model path based on detection mode
if DETECTION_MODE == "pig-only":
    MODEL_PATH = MODEL_PATH_ENV or PIG_MODEL_PATH
    if not MODEL_PATH:
        raise RuntimeError("MODEL_PATH не задан в .env (или укажите Legacy PIG_MODEL_PATH)")
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
    """Установка минимальных зависимостей"""
    print("📦 Installing minimal dependencies...")
    try:
        # Используем основной requirements файл (теперь минимальный)
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if not os.path.exists(requirements_path):
            print("❌ requirements.txt not found!")
            return

        python_exe = sys.executable
        subprocess.check_call([
            python_exe, "-m", "pip", "install", "-r", requirements_path
        ])
        print("✅ Dependencies installed successfully")

        # Быстрая проверка системы
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🎯 CUDA ready: {gpu_name}")
            else:
                print("💻 Using CPU (CUDA not available)")
        except ImportError:
            print("⚠️ PyTorch not available")

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation error: {e}")
        print("💡 Try installing minimal dependencies manually:")
        print("   pip install fastapi uvicorn torch ultralytics opencv-python")

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

        # install_requirements()       

        # Import ASGI app and start server
        logger.info(f'Starting server at http://{HOST}:{PORT}')
        logger.info(f'API Health Check: http://{HOST}:{PORT}/api/health')
        logger.info(f'Debug mode: {DEBUG}, Hot-reload: {RELOAD}')

        try:
            import uvicorn
            # Для reload uvicorn требует import string, а не объект приложения
            # Используем RELOAD из .env или автоматически включаем при DEBUG, если не указано
            app_str = "api.app:app" if RELOAD or DEBUG else None
            
            if app_str is None:
                # Если RELOAD=False и DEBUG=False, загружаем приложение напрямую для производительности
                from api.app import app as fastapi_app
                app = fastapi_app
            else:
                app = app_str
                
            uvicorn.run(
                app,
                host=HOST,
                port=PORT,
                reload=RELOAD or DEBUG,  # Релоад включается, если включён RELOAD или DEBUG
                log_level="debug" if DEBUG else "info",
                # Исключаем директории, которые часто меняются и провоцируют бесконечный reload
                reload_excludes=[
                    "logs",
                    "uploads",
                    "records",
                    "models",
                ]
            )
        except Exception as e:
            logger.error(f'Error starting server via uvicorn: {str(e)}')
            raise
    except Exception as e:
        logger.error(f'Error starting server: {str(e)}')
        sys.exit(1)

def main_with_args():
    """Главная функция с обработкой аргументов командной строки"""
    import argparse

    parser = argparse.ArgumentParser(description='PigWeight - Video Processing Server')
    parser.add_argument('--install', action='store_true',
                       help='Установить все зависимости')

    args = parser.parse_args()

    if args.install:
        install_requirements()
        return

    # Запуск сервера
    main()

if __name__ == '__main__':
    main_with_args()
