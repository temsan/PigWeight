#!/usr/bin/env python3
"""
Быстрое исправление и перезапуск системы
"""

import os
import subprocess
import sys
import time
import signal

def kill_python_processes():
    """Завершить все процессы Python, кроме текущего"""
    try:
        current_pid = os.getpid()
        if os.name == 'nt':  # Windows
            # Получаем список всех процессов python и завершаем только те, что не текущий
            result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[3:]:  # Пропускаем заголовки
                    if 'python.exe' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                if pid != current_pid:
                                    subprocess.run(['taskkill', '/pid', str(pid), '/f'],
                                                 capture_output=True, text=True)
                            except (ValueError, IndexError):
                                pass

            # То же для pythonw.exe
            result = subprocess.run(['tasklist', '/fi', 'imagename eq pythonw.exe'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[3:]:
                    if 'pythonw.exe' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                if pid != current_pid:
                                    subprocess.run(['taskkill', '/pid', str(pid), '/f'],
                                                 capture_output=True, text=True)
                            except (ValueError, IndexError):
                                pass
        else:  # Unix/Linux
            subprocess.run(['pkill', '-f', 'python'], capture_output=True)
        print("✅ Процессы Python завершены")
    except Exception as e:
        print(f"⚠️  Ошибка завершения процессов: {e}")

def wait_for_port_free(port=8000, timeout=10):
    """Ждать освобождения порта"""
    import socket
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result != 0:  # Порт свободен
                    return True
        except Exception:
            return True
        time.sleep(0.5)
    
    return False

def main():
    print("🔧 Быстрое исправление и перезапуск PigWeight...")
    
    # 1. Завершаем процессы
    print("\n1. Завершение процессов...")
    kill_python_processes()
    
    # 2. Ждем освобождения порта
    print("2. Ожидание освобождения порта...")
    if wait_for_port_free():
        print("✅ Порт 8000 свободен")
    else:
        print("⚠️  Порт может быть еще занят")
    
    # 3. Устанавливаем переменные окружения
    print("3. Настройка переменных окружения...")
    env_vars = {
        'USE_OPTIMIZED_PREPROCESSING': 'true',
        'PREPROCESSING_METHOD': 'adaptive',
        'ANTI_LETTERBOX': 'false',
        'MODEL_PATH': 'models/pig_yolo11-seg.v4.pt',
        'DEVICE': 'cpu',
        'BATCH_SIZE': '4',  # Уменьшаем для стабильности
        'IMG_SIZE': '960',
        'CONF_THRESHOLD': '0.30',
        'DEBUG': 'false'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    # 4. Запускаем систему
    print("\n4. Запуск системы...")
    try:
        # Импортируем и запускаем
        from main import main as app_main
        print("✅ Система запущена!")
        print("🌐 Откройте http://localhost:8000 в браузере")
        print("🎭 WebRTC и маски должны работать")
        print("\nДля остановки нажмите Ctrl+C")
        
        app_main()
        
    except KeyboardInterrupt:
        print("\n🛑 Система остановлена пользователем")
        return 0
    except Exception as e:
        print(f"\n❌ Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
