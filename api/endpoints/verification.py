"""
Эндпоинты для сверки актов и агрегированных отчётов.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.endpoints.records import get_records_list

router = APIRouter(prefix="/api", tags=["verification"])

logger = logging.getLogger(__name__)


async def _build_grouped_verification() -> Dict[str, Any]:
    acts = await get_records_list()

    grouped: Dict[str, Dict[str, Any]] = {}
    summary = {"total_acts": 0, "verified_acts": 0, "discrepancy_acts": 0}

    for act in acts:
        if act.get("status") != "success":
            continue

        timestamp = act.get("finished_at")
        date_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "unknown"

        bucket = grouped.setdefault(date_key, {
            "acts": [],
            "total_acts": 0,
            "verified_acts": 0,
            "discrepancy_acts": 0,
            "total_pigs": 0,
            "total_duration": 0,
        })

        verified = act.get("verification", {}).get("verified", False)
        act["verification"] = {"verified": verified}

        summary["total_acts"] += 1
        bucket["total_acts"] += 1
        bucket["total_pigs"] += act.get("seen_total", 0)
        bucket["total_duration"] += act.get("duration_sec", 0)

        if verified:
            summary["verified_acts"] += 1
            bucket["verified_acts"] += 1
        else:
            summary["discrepancy_acts"] += 1
            bucket["discrepancy_acts"] += 1

        bucket["acts"].append(act)

    sorted_dates = sorted(grouped.keys(), reverse=True)
    sorted_grouped = {date: grouped[date] for date in sorted_dates}

    for data in sorted_grouped.values():
        total = data["total_acts"]
        data["avg_duration"] = (data["total_duration"] / total) if total else 0

    return {"summary": summary, "grouped_by_date": sorted_grouped}


@router.get("/verification/grouped")
async def verification_grouped():
    try:
        return await _build_grouped_verification()
    except Exception as exc:
        logger.exception("Ошибка агрегации сверок: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/verification/report")
async def verification_report():
    try:
        data = await _build_grouped_verification()
    except Exception as exc:
        logger.exception("Ошибка формирования отчёта сверки: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    grouped = data["grouped_by_date"]

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_acts": data["summary"]["total_acts"],
            "verified_acts": data["summary"]["verified_acts"],
            "discrepancy_acts": data["summary"]["discrepancy_acts"],
            "total_pigs_counted": sum(bucket["total_pigs"] for bucket in grouped.values()),
            "verification_rate": (
                data["summary"]["verified_acts"] / data["summary"]["total_acts"] * 100
            ) if data["summary"]["total_acts"] else 0,
        },
        "issues": [],
    }

    for date_key, bucket in grouped.items():
        for act in bucket["acts"]:
            if act.get("verification", {}).get("verified"):
                continue
            flow = act.get("flow", {})
            report["issues"].append({
                "act_file": act.get("act_file"),
                "date": date_key,
                "left_count": flow.get("left_in", 0),
                "right_count": flow.get("right_in", 0),
                "difference": abs(flow.get("left_in", 0) - flow.get("right_in", 0)),
                "status": "несоответствие",
            })

    return report
