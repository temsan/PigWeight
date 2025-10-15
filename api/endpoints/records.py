"""
Эндпоинты и утилиты для работы с актами взвешивания.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import RECORDS_DIR

router = APIRouter(prefix="/api", tags=["records"])

logger = logging.getLogger(__name__)


def _get_records_dir() -> Path:
    if RECORDS_DIR is None:
        raise RuntimeError("RECORDS_DIR не инициализирован")
    return RECORDS_DIR


def _load_records() -> List[Dict[str, Any]]:
    records_dir = _get_records_dir()
    items: List[Dict[str, Any]] = []

    for path in sorted(records_dir.glob("act_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            
            # Форматируем данные для отображения в журнале
            from datetime import datetime
            
            # Извлекаем дату и время из finished_at или started_at
            timestamp = payload.get("finished_at") or payload.get("started_at") or 0
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
            else:
                date_str = "Неизвестная дата"
                time_str = ""
            
            # Извлекаем данные из payload
            stream_id = payload.get("stream_id", "unknown")
            seen_total = payload.get("seen_total", 0)
            peak_concurrent = payload.get("peak_concurrent", 0)
            flow = payload.get("flow", {})
            left_in = flow.get("left_in", 0)
            right_in = flow.get("right_in", 0)
            
            # Формируем запись для журнала
            items.append({
                "act_file": path.name,
                "name": stream_id,
                "date": date_str,
                "time": time_str,
                "weighing_section": stream_id,  # Используем stream_id как участок
                "group": f"Акт {path.stem}",
                "total_count": seen_total,
                "total_weight": 0.0,  # Вес не записывается в актах
                "timestamp": timestamp,
                **payload,
            })
        except Exception as exc:
            logger.warning("Не удалось прочитать акт %s: %s", path.name, exc)
            continue

    return sorted(items, key=lambda item: item.get("timestamp", 0), reverse=True)


def _load_record_details(act_name: str) -> Dict[str, Any]:
    if ".." in act_name or "/" in act_name or "\\" in act_name:
        raise HTTPException(status_code=400, detail="Некорректное имя акта")

    records_dir = _get_records_dir()
    target_path = records_dir / act_name

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Акт не найден")

    with open(target_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    svg_path = target_path.with_suffix(".svg")
    if svg_path.exists():
        svg_content = svg_path.read_text(encoding="utf-8")
        data["svg"] = "data:image/svg+xml;base64," + base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")

    return data


async def get_records_list() -> List[Dict[str, Any]]:
    """Возвращает список актов для переиспользования в других эндпоинтах."""
    return _load_records()


@router.get("/records")
async def list_records():
    try:
        records = await get_records_list()
        return {"records": records}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ошибка при чтении списка актов: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/records/{act_name}")
async def get_record(act_name: str):
    try:
        return _load_record_details(act_name)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ошибка при чтении акта %s: %s", act_name, exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
