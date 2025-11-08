"""
Excel export endpoints
Экспорт актов взвешивания в Excel формат

СОЗДАНО: 8 ноября 2025
"""

import logging
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.dependencies import get_database_manager

router = APIRouter(prefix="/api/export", tags=["export"])
logger = logging.getLogger(__name__)


class ExportRequest(BaseModel):
    """Запрос на экспорт"""
    start_date: str
    end_date: str
    stream_id: Optional[str] = None
    section: str = "6B"  # Секция для Excel (по умолчанию 6B)


@router.post("/excel")
async def export_to_excel(request: ExportRequest = Body(...)):
    """
    Экспорт актов взвешивания в Excel
    
    Request body:
    {
        "start_date": "2025-11-01T00:00:00",
        "end_date": "2025-11-08T23:59:59",
        "stream_id": "cam101",  // опционально
        "section": "6B"  // опционально
    }
    
    Response:
    - Excel файл для скачивания
    
    Требования: 4.2, 4.3, 4.4, 4.5
    """
    try:
        # Парсим даты
        try:
            start = datetime.fromisoformat(request.start_date)
            end = datetime.fromisoformat(request.end_date)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Неверный формат даты: {e}"
            )
        
        # Получаем DatabaseManager
        db = get_database_manager()
        
        # Получаем акты за период
        acts = db.get_acts_by_period(
            start_date=start,
            end_date=end,
            stream_id=request.stream_id
        )
        
        if not acts:
            raise HTTPException(
                status_code=404,
                detail="Нет актов за указанный период"
            )
        
        # Импортируем ExcelExporter
        try:
            from pig_tracking.excel_exporter import ExcelExporter
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="ExcelExporter не установлен"
            )
        
        # Создаём временную директорию для экспорта
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        # Генерируем имя файла
        filename = f"acts_{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.xlsx"
        output_path = export_dir / filename
        
        # Экспортируем
        exporter = ExcelExporter()
        exporter.export_to_excel(
            acts=acts,
            output_path=str(output_path),
            section=request.section,
            group_by_date=True
        )
        
        logger.info(f"✅ Экспорт завершён: {filename}, актов: {len(acts)}")
        
        # Возвращаем файл
        return FileResponse(
            path=str(output_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка экспорта: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при экспорте: {str(e)}"
        )
