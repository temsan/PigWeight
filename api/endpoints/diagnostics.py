"""
Diagnostics endpoints for troubleshooting
"""

import os
import sys
import platform
import psutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["diagnostics"])

def check_browser_compatibility(user_agent: str) -> Dict[str, Any]:
    """Проверка совместимости браузера"""
    compatibility = {
        "supported": True,
        "warnings": [],
        "recommendations": []
    }
    
    if not user_agent:
        compatibility["warnings"].append("User-Agent не определен")
        return compatibility
    
    user_agent_lower = user_agent.lower()
    
    # Проверка на старые браузеры
    if "msie" in user_agent_lower or "trident" in user_agent_lower:
        compatibility["supported"] = False
        compatibility["warnings"].append("Internet Explorer не поддерживается")
        compatibility["recommendations"].append("Используйте современный браузер: Chrome, Firefox, Safari или Edge")
    
    # Проверка на мобильные браузеры
    if any(mobile in user_agent_lower for mobile in ["mobile", "android", "iphone", "ipad"]):
        compatibility["warnings"].append("Мобильный браузер может иметь ограниченную функциональность")
        compatibility["recommendations"].append("Для полной функциональности используйте десктопный браузер")
    
    # Проверка на WebRTC поддержку (косвенно)
    if "chrome" in user_agent_lower:
        compatibility["webrtc_support"] = "excellent"
    elif "firefox" in user_agent_lower:
        compatibility["webrtc_support"] = "good"
    elif "safari" in user_agent_lower:
        compatibility["webrtc_support"] = "limited"
    else:
        compatibility["webrtc_support"] = "unknown"
        compatibility["warnings"].append("Неизвестная поддержка WebRTC")
    
    return compatibility

def check_system_resources() -> Dict[str, Any]:
    """Проверка системных ресурсов"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        resources = {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
                "status": "ok" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 1),
                "status": "ok" if disk.free > 1024**3 else "warning" if disk.free > 512**6 else "critical"
            },
            "cpu": {
                "percent_used": cpu_percent,
                "status": "ok" if cpu_percent < 70 else "warning" if cpu_percent < 85 else "critical"
            }
        }
        
        return resources
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return {"error": str(e)}

def check_dependencies() -> Dict[str, Any]:
    """Проверка зависимостей и конфигурации"""
    deps = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependencies": {},
        "configuration": {}
    }
    
    # Проверка ключевых зависимостей
    required_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"), 
        ("opencv-python", "cv2"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("psutil", "psutil")
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            deps["dependencies"][package_name] = "installed"
        except ImportError:
            deps["dependencies"][package_name] = "missing"
    
    # Проверка конфигурации
    config_vars = [
        "MODEL_PATH", "DEVICE", "DEBUG", "HOST", "PORT",
        "CAM_DEFAULT", "JPEG_QUALITY", "BATCH_SIZE"
    ]
    
    for var in config_vars:
        value = os.getenv(var)
        deps["configuration"][var] = value if value else "not_set"
    
    return deps

def check_model_files() -> Dict[str, Any]:
    """Проверка файлов моделей"""
    model_status = {
        "models_found": [],
        "models_missing": [],
        "models_directory_exists": False
    }
    
    models_dir = Path("models")
    if models_dir.exists():
        model_status["models_directory_exists"] = True
        
        # Ищем файлы моделей
        model_extensions = [".pt", ".onnx", ".engine"]
        for ext in model_extensions:
            model_files = list(models_dir.glob(f"*{ext}"))
            for model_file in model_files:
                model_status["models_found"].append({
                    "name": model_file.name,
                    "size_mb": round(model_file.stat().st_size / (1024**2), 2),
                    "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })
    
    # Проверка конкретных моделей из конфигурации
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        if not Path(model_path).exists():
            model_status["models_missing"].append(model_path)
    
    return model_status

def generate_troubleshooting_suggestions(diagnostics: Dict[str, Any]) -> List[str]:
    """Генерация предложений по устранению проблем"""
    suggestions = []
    
    # Проверка ресурсов
    resources = diagnostics.get("system_resources", {})
    
    if resources.get("memory", {}).get("status") == "critical":
        suggestions.append("⚠️ Критически мало памяти. Закройте другие приложения или увеличьте RAM")
    elif resources.get("memory", {}).get("status") == "warning":
        suggestions.append("⚠️ Высокое использование памяти. Рекомендуется закрыть ненужные приложения")
    
    if resources.get("disk", {}).get("status") == "critical":
        suggestions.append("⚠️ Критически мало места на диске. Освободите место для корректной работы")
    
    if resources.get("cpu", {}).get("status") == "critical":
        suggestions.append("⚠️ Высокая загрузка CPU. Система может работать медленно")
    
    # Проверка зависимостей
    deps = diagnostics.get("dependencies", {})
    missing_deps = [pkg for pkg, status in deps.get("dependencies", {}).items() if status == "missing"]
    if missing_deps:
        suggestions.append(f"❌ Отсутствуют зависимости: {', '.join(missing_deps)}. Установите их через pip")
    
    # Проверка моделей
    models = diagnostics.get("model_files", {})
    if not models.get("models_directory_exists"):
        suggestions.append("❌ Директория models не найдена. Создайте её и загрузите модели")
    elif not models.get("models_found"):
        suggestions.append("❌ Файлы моделей не найдены. Загрузите модели в директорию models/")
    
    # Проверка браузера
    browser = diagnostics.get("browser_compatibility", {})
    if not browser.get("supported"):
        suggestions.append("❌ Браузер не поддерживается. Используйте современный браузер")
    
    if browser.get("webrtc_support") == "limited":
        suggestions.append("⚠️ Ограниченная поддержка WebRTC. Видеопоток может работать нестабильно")
    
    if not suggestions:
        suggestions.append("✅ Система настроена корректно. Проблем не обнаружено")
    
    return suggestions

@router.get("/diagnostics")
async def run_diagnostics(request: Request):
    """Запуск полной диагностики системы"""
    try:
        user_agent = request.headers.get("user-agent", "")
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "browser_compatibility": check_browser_compatibility(user_agent),
            "system_resources": check_system_resources(),
            "dependencies": check_dependencies(),
            "model_files": check_model_files()
        }
        
        # Генерируем предложения по устранению проблем
        diagnostics["troubleshooting_suggestions"] = generate_troubleshooting_suggestions(diagnostics)
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/diagnostics/quick")
async def quick_diagnostics():
    """Быстрая диагностика основных компонентов"""
    try:
        quick_check = {
            "timestamp": datetime.now().isoformat(),
            "status": "checking"
        }
        
        # Проверка памяти
        memory = psutil.virtual_memory()
        quick_check["memory_ok"] = memory.percent < 85
        
        # Проверка диска
        disk = psutil.disk_usage('/')
        quick_check["disk_ok"] = disk.free > 512 * 1024 * 1024  # 512MB
        
        # Проверка модели
        model_path = os.getenv("MODEL_PATH")
        quick_check["model_ok"] = model_path and Path(model_path).exists()
        
        # Общий статус
        all_ok = all([
            quick_check["memory_ok"],
            quick_check["disk_ok"], 
            quick_check["model_ok"]
        ])
        
        quick_check["status"] = "ok" if all_ok else "issues_detected"
        
        return quick_check
        
    except Exception as e:
        logger.error(f"Error in quick diagnostics: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/diagnostics/browser-test")
async def browser_test(request: Request):
    """Тест совместимости браузера"""
    user_agent = request.headers.get("user-agent", "")
    
    test_results = {
        "user_agent": user_agent,
        "compatibility": check_browser_compatibility(user_agent),
        "features_to_test": [
            "WebRTC support",
            "File API support", 
            "WebSocket support",
            "Canvas support",
            "Local Storage support",
            "Chart.js loading",
            "External script loading"
        ],
        "recommendations": []
    }
    
    # Добавляем рекомендации на основе браузера
    if "chrome" in user_agent.lower():
        test_results["recommendations"].append("✅ Chrome обеспечивает лучшую совместимость")
    elif "firefox" in user_agent.lower():
        test_results["recommendations"].append("✅ Firefox хорошо поддерживается")
    elif "safari" in user_agent.lower():
        test_results["recommendations"].append("⚠️ Safari может иметь ограничения с WebRTC")
    else:
        test_results["recommendations"].append("⚠️ Рекомендуется использовать Chrome или Firefox")
    
    return test_results