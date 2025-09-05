@echo off
REM Quick start batch file for Windows - BALANCED profile by default with CUDA 12.9

echo 🚀 PigWeight Quick Start - CUDA 12.9 Optimized
echo 📊 Default Profile: BALANCED (60 FPS, 50ms latency)
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please install Python 3.11+ and add to PATH
    pause
    exit /b 1
)

REM Check if optimized configuration exists
if not exist .env.optimized (
    echo ❌ .env.optimized not found
    echo Please run setup first or copy configuration
    pause
    exit /b 1
)

echo ✅ Starting optimized server...
echo 💻 Dashboard: http://localhost:8000/static/dashboard.html
echo 📊 Metrics: ws://localhost:8765/ws/metrics
echo.

REM Start the server with BALANCED profile (default)
python start_optimized.py --profile BALANCED

if errorlevel 1 (
    echo.
    echo ❌ Server failed to start
    echo Check logs for details
    pause
    exit /b 1
)

echo.
echo 👋 Server stopped
pause