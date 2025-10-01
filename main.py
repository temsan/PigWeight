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
parser.add_argument('--runtime', choices=['auto', 'pytorch', 'onnx-gpu', 'onnx-cpu', 'cpu'], default='auto', help='Выбор рантайма (по умолчанию: auto)')
parser.add_argument('--install', action='store_true', help='Установить все зависимости')
args, unknown = parser.parse_known_args()

# Автоматический выбор оптимального рантайма и профиля
if args.runtime == 'auto':
    from core.config import detect_optimal_runtime, apply_runtime_optimizations
    runtime_info = detect_optimal_runtime()
    apply_runtime_optimizations(runtime_info)
elif args.runtime != 'auto':
    # Ручной выбор рантайма (профиль всё равно выбирается автоматически)
    from core.config import detect_optimal_runtime, apply_runtime_optimizations
    runtime_info = detect_optimal_runtime()
    
    if args.runtime == 'pytorch':
        runtime_info['runtime'] = 'pytorch'
        runtime_info['device'] = 'auto'
        os.environ['PREFER_ONNX'] = 'false'
    elif args.runtime == 'onnx-gpu':
        runtime_info['runtime'] = 'onnx-gpu'
        runtime_info['device'] = 'cuda:0'
        runtime_info['provider'] = 'CUDAExecutionProvider'
        os.environ['PREFER_ONNX'] = 'true'
        os.environ['ONNX_PROVIDER'] = 'CUDAExecutionProvider'
    elif args.runtime == 'onnx-cpu':
        runtime_info['runtime'] = 'onnx-cpu'
        runtime_info['device'] = 'cpu'
        runtime_info['provider'] = 'CPUExecutionProvider'
        runtime_info['profile'] = 'CPU_ONLY'  # Принудительно CPU профиль
        os.environ['PREFER_ONNX'] = 'true'
        os.environ['ONNX_PROVIDER'] = 'CPUExecutionProvider'
    elif args.runtime == 'cpu':
        runtime_info['runtime'] = 'cpu'
        runtime_info['device'] = 'cpu'
        runtime_info['profile'] = 'CPU_ONLY'
        runtime_info['use_half'] = False
        os.environ['DEVICE'] = 'cpu'
        os.environ['USE_HALF'] = 'false'
    
    apply_runtime_optimizations(runtime_info)

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

def display_runtime_info():
    """Отображает информацию о выбранном рантайме."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"🔥 GPU: {gpu_name} ({gpu_memory}GB VRAM)")
        else:
            print("💻 CPU: CUDA недоступен")
    except ImportError:
        print("⚠️ PyTorch не установлен")

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"🧠 ONNX Runtime: {', '.join(providers)}")
    except ImportError:
        print("⚠️ ONNX Runtime недоступен")
    
    # Отображаем текущие настройки
    device = os.getenv('DEVICE', 'auto')
    use_half = os.getenv('USE_HALF', 'auto')
    profile = os.getenv('TARGET_FPS', 'не установлен')
    print(f"⚙️ Настройки: device={device}, half_precision={use_half}, target_fps={profile}")

def install_requirements():
    """Установка минимальных зависимостей"""
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
        
        # Отображаем информацию о рантайме
        if not CONFIG.DEBUG:
            print("🚀 PigWeight - Система видеоаналитики")
            print("=" * 50)
            display_runtime_info()
            print("=" * 50)
        
        ensure_dir('models')
        ensure_dir('stream')
        ensure_dir('uploads')

        if CONFIG.DEBUG:
            logger.info(f'Запуск сервера на http://{CONFIG.HOST}:{CONFIG.PORT}')
            logger.info(f'Проверка здоровья API: http://{CONFIG.HOST}:{CONFIG.PORT}/api/health')
            logger.info(f'Режим отладки: {CONFIG.DEBUG}, Горячая перезагрузка: {CONFIG.RELOAD}')
        else:
            # В production показываем только основную информацию
            print(f'🚀 PigWeight сервер запущен на http://{CONFIG.HOST}:{CONFIG.PORT}')

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
                log_level="debug" if CONFIG.DEBUG else "warning",
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