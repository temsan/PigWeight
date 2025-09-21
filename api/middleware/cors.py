"""
CORS middleware configuration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app: FastAPI):
    """Настройка CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # В продакшене указать конкретные домены
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )