
import os
from pathlib import Path

# --- Directories ---
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RECORDS_DIR = BASE_DIR / "records"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)
