#!/usr/bin/env python3
"""Запуск обработки видео с выводом в консоль"""
import subprocess
import sys

cmd = [sys.executable, "console_app.py", "--video", "uploads/0825.mp4"]
print(f"Запуск: {' '.join(cmd)}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)
