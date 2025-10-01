"""Системные endpoints: состояние сервиса, доступные файлы."""

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from api.dependencies import STREAM_MANAGER
from core.config import CONFIG

router = APIRouter(prefix="/api/system", tags=["system"])

VIDEO_DIR = Path("uploads")
RECORDS_DIR = Path("records")


def _collect_local_videos(limit: int = 10) -> List[Dict[str, Any]]:
    videos: List[Dict[str, Any]] = []
    if not VIDEO_DIR.exists():
        return videos
    for file_path in sorted(VIDEO_DIR.glob("*")):
        if not file_path.is_file():
            continue
        size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
        videos.append({
            "id": file_path.stem,
            "filename": file_path.name,
            "size_mb": size_mb,
        })
        if len(videos) >= limit:
            break
    return videos


def _collect_active_streams() -> List[str]:
    streams = []
    if STREAM_MANAGER and getattr(STREAM_MANAGER, "streams", None):
        streams = list(STREAM_MANAGER.streams.keys())
    return streams


@router.get("/status")
def system_status() -> Dict[str, Any]:
    """Краткое состояние сервисов: модель, устройство, локальные файлы."""
    status: Dict[str, Any] = {
        "status": "online",
        "model_path": CONFIG.MODEL_PATH,
        "device": CONFIG.DEVICE,
        "available_videos": _collect_local_videos(),
        "active_streams": _collect_active_streams(),
    }

    if RECORDS_DIR.exists():
        status["records_total"] = len(list(RECORDS_DIR.glob("act_*.json")))
    else:
        status["records_total"] = 0

    return status
