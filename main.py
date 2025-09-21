import os
import sys
import urllib.request
import subprocess
import logging
import argparse
from pathlib import Path

# --- Argument Parsing ---
# This needs to be done before config is imported and instantiated
parser = argparse.ArgumentParser(description='PigWeight - Video Processing Server')
parser.add_argument('--profile', choices=['ULTRA_PERFORMANCE', 'BALANCED', 'POWER_SAVING', 'CPU_ONLY'], help='Performance profile')
parser.add_argument('--install', action='store_true', help='Установить все зависимости')
args, unknown = parser.parse_known_args()

if args.profile:
    from core.config import apply_performance_profile
    apply_performance_profile(args.profile)

# --- Config and Logging --- 
# Now that the profile is potentially set via env vars, we can import the config
from core.config import setup_logging, CONFIG

# --- Helper Functions ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_model():
    if not os.path.exists(CONFIG.MODEL_PATH):
        print(f"Downloading model to {CONFIG.MODEL_PATH}...")
        ensure_dir(os.path.dirname(CONFIG.MODEL_PATH))
        urllib.request.urlretrieve(CONFIG.MODEL_URL, CONFIG.MODEL_PATH)
        print("Model downloaded successfully")
    else:
        print(f"Model already exists at {CONFIG.MODEL_PATH}")

def install_requirements():
    """Установка минимальных зависимостей"""
    print("📦 Installing minimal dependencies...")
    try:
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if not os.path.exists(requirements_path):
            print("❌ requirements.txt not found!")
            return

        python_exe = sys.executable
        subprocess.check_call([
            python_exe, "-m", "pip", "install", "-r", requirements_path
        ])
        print("✅ Dependencies installed successfully")

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
    onnx_path = CONFIG.get('ONNX_PATH')
    if not onnx_path: return
    if not os.path.exists(onnx_path) or os.path.getmtime(CONFIG.MODEL_PATH) > os.path.getmtime(onnx_path):
        print(f"Converting model to ONNX format...")
        ensure_dir(os.path.dirname(onnx_path))
        
        try:
            from ultralytics import YOLO
            model = YOLO(CONFIG.MODEL_PATH)
            model.export(format='onnx', opset=12)
            default_onnx = CONFIG.MODEL_PATH.replace('.pt', '.onnx')
            if os.path.exists(default_onnx) and default_onnx != onnx_path:
                os.replace(default_onnx, onnx_path)
            print("Model converted to ONNX successfully")
        except Exception as e:
            print(f"Error converting model to ONNX: {str(e)}")
            raise
    else:
        print(f"ONNX model already exists at {onnx_path} and is up to date")

def main():
    try:
        logger = setup_logging(debug=CONFIG.DEBUG)
        
        ensure_dir('models')
        ensure_dir('stream')
        ensure_dir('uploads')

        logger.info(f'Запуск сервера на http://{CONFIG.HOST}:{CONFIG.PORT}')
        logger.info(f'Проверка здоровья API: http://{CONFIG.HOST}:{CONFIG.PORT}/api/health')
        logger.info(f'Режим отладки: {CONFIG.DEBUG}, Горячая перезагрузка: {CONFIG.RELOAD}')

        try:
            import uvicorn
            app_str = "api.app:app" if CONFIG.RELOAD or CONFIG.DEBUG else None
            
            if app_str is None:
                from api.app import app as fastapi_app
                app = fastapi_app
            else:
                app = app_str
                
            uvicorn.run(
                app,
                host=CONFIG.HOST,
                port=CONFIG.PORT,
                reload=CONFIG.RELOAD,
                log_level="debug" if CONFIG.DEBUG else "info",
                reload_dirs=[
                    str(Path(__file__).parent / "api"),
                    str(Path(__file__).parent / "core"),
                    str(Path(__file__).parent / "services"),
                    str(Path(__file__).parent / "static"),
                ],
                reload_excludes=[
                    "logs", "logs/**", "*.log",
                    "uploads", "uploads/**",
                    "records", "records/**",
                    "models", "models/**",
                ]
            )
        except Exception as e:
            logger.error(f'Ошибка запуска сервера через uvicorn: {str(e)}')
            raise
    except Exception as e:
        logger.error(f'Ошибка запуска сервера: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    if args.install:
        install_requirements()
    else:
        main()