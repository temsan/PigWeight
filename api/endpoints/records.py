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
            
            from datetime import datetime
            import re
            
            # Извлекаем данные из payload
            stream_id = payload.get("stream_id", "unknown")
            seen_total = payload.get("seen_total", 0)
            peak_concurrent = payload.get("peak_concurrent", 0)
            duration_sec = payload.get("duration_sec", 0)
            
            # КРИТИЧНО: Пропускаем нулевые записи
            if seen_total == 0:
                logger.debug(f"Пропускаем нулевую запись: {path.name}")
                continue
            
            # Извлекаем timestamp - ПРИОРИТЕТ: имя файла > finished_at > started_at > mtime
            timestamp = None
            
            # 1. Пытаемся извлечь из имени файла: act_stream_20251015-175651.json
            match = re.search(r'(\d{8})-(\d{6})', path.name)
            if match:
                try:
                    date_part = match.group(1)  # 20251015
                    time_part = match.group(2)  # 175651
                    dt_str = f"{date_part}{time_part}"
                    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                    timestamp = dt.timestamp()
                    logger.debug(f"Timestamp из имени файла: {timestamp}")
                except Exception as e:
                    logger.debug(f"Ошибка парсинга даты из имени: {e}")
            
            # 2. Если не получилось, пробуем finished_at или started_at
            if not timestamp or timestamp < 1000000000:
                ts = payload.get("finished_at") or payload.get("started_at")
                if ts and ts > 1000000000:
                    timestamp = ts
                    logger.debug(f"Timestamp из payload: {timestamp}")
            
            # 3. Если все еще нет, используем время модификации файла
            if not timestamp or timestamp < 1000000000:
                timestamp = path.stat().st_mtime
                logger.debug(f"Timestamp из mtime: {timestamp}")
            
            # Форматируем дату и время
            if timestamp and timestamp > 1000000000:
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M")
            else:
                # Если timestamp все еще некорректный, пропускаем запись
                logger.warning(f"Некорректный timestamp для {path.name}, пропускаем")
                continue
            
            # Формируем запись для журнала
            items.append({
                "act_file": path.name,
                "name": stream_id,
                "date": date_str,
                "time": time_str,
                "total_count": seen_total,
                "peak_concurrent": peak_concurrent,
                "duration_sec": duration_sec,
                "timestamp": timestamp,
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


@router.post("/records/cleanup")
async def cleanup_empty_records():
    """Удаляет нулевые записи из журнала"""
    try:
        records_dir = _get_records_dir()
        deleted_count = 0
        
        for path in records_dir.glob("act_*.json"):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                
                seen_total = payload.get("seen_total", 0)
                
                # Удаляем файлы с нулевым количеством
                if seen_total == 0:
                    path.unlink()
                    deleted_count += 1
                    logger.info(f"Удален нулевой акт: {path.name}")
                    
                    # Удаляем связанные файлы (svg, md)
                    svg_path = path.with_suffix(".svg")
                    if svg_path.exists():
                        svg_path.unlink()
                    
                    md_path = path.with_suffix(".md")
                    if md_path.exists():
                        md_path.unlink()
                        
            except Exception as e:
                logger.warning(f"Ошибка при проверке {path.name}: {e}")
                continue
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Удалено {deleted_count} нулевых записей"
        }
        
    except Exception as exc:
        logger.exception("Ошибка при очистке записей: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
