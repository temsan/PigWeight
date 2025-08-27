# GEMINI.md

## Project Overview

This project is a pig weighing system that uses computer vision to analyze video streams. It is a web application with a FastAPI backend and a simple HTML/JavaScript frontend. The backend uses OpenCV for video processing and a YOLO model for pig detection. The application can process live RTSP streams or uploaded video files. The frontend provides a user interface for viewing the video stream, controlling the application, and viewing the results of the analysis.

**Key Technologies:**

*   **Backend:** Python, FastAPI, Uvicorn, OpenCV, YOLO (Ultralytics)
*   **Frontend:** HTML, JavaScript
*   **Real-time Communication:** WebSockets

**Architecture:**

*   **`main.py`:** The main entry point of the application. It initializes the environment, downloads the required models, and starts the Uvicorn server.
*   **`api/app.py`:** The core of the backend. It defines the FastAPI application, including all API endpoints and WebSocket handlers. It manages the camera streams, video processing, and model inference.
*   **`static/`:** Contains the frontend files, including `index.html`, CSS, and JavaScript.
*   **`models/`:** Stores the YOLO models used for pig detection.
*   **`uploads/`:**  A directory for storing uploaded video files.
*   **`scripts/`:** Contains utility scripts, such as for cleaning the uploads directory.

## Building and Running

**1. Installation:**

```bash
# Create a virtual environment
python -m venv .venv
# Activate the virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

**2. Running the Application:**

```bash
python main.py
```

The application will be available at `http://localhost:8000`.

## Development Conventions

*   The project uses a `.env` file for configuration. An example is provided in `.env.example`.
*   The backend code is located in the `api/` directory.
*   The frontend code is in the `static/` directory.
*   The project uses `requirements.txt` to manage Python dependencies.
*   The `clean_uploads.bat` and `scripts/clean_uploads.py` scripts can be used to clean the `uploads/` directory.
