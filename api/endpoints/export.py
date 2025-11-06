"""
Export endpoints
Excel export and comparison functionality
"""

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_database_manager
from pig_tracking.excel_exporter import ExcelExporter
from pig_tracking.excel_comparator import ExcelComparator

router = APIRouter(prefix="/api/export", tags=["export"])
logger = logging.getLogger(__name__)


@router.post("/excel")
async def export_to_excel(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    output_path: Optional[str] = Query(None, description="Output file path")
):
    """
    Экспорт актов взвешивания в Excel
    
    Query params:
    - start_date: начальная дата (обязательно)
    - end_date: конечная дата (обязательно)
    - output_path: путь для сохранения (опционально)
    
    Response: Excel файл для скачивания
    """
    try:
        db = get_database_manager()
        
        # Парсим даты
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Генерируем имя файла
        if not output_path:
            filename = f"weighing_acts_{start_date}_to_{end_date}.xlsx"
            output_path = os.path.join("exports", filename)
            os.makedirs("exports", exist_ok=True)
        
        # Создаем экспортер
        exporter = ExcelExporter(db)
        
        # Экспортируем
        result_path = exporter.export_to_excel(
            start_date=start_dt,
            end_date=end_dt,
            output_path=output_path
        )
        
        logger.info(f"📊 Excel exported: {result_path}")
        
        # Возвращаем файл
        return FileResponse(
            path=result_path,
            filename=os.path.basename(result_path),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_with_excel(
    manual_excel_path: str = Query(..., description="Path to manual Excel file"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    output_path: Optional[str] = Query(None, description="Output comparison file path")
):
    """
    Сверка автоматических результатов с ручными записями из Excel
    
    Query params:
    - manual_excel_path: путь к файлу с ручными записями
    - start_date: начальная дата
    - end_date: конечная дата
    - output_path: путь для сохранения отчета (опционально)
    
    Response: Excel файл с результатами сверки
    """
    try:
        db = get_database_manager()
        
        # Проверяем существование файла
        if not os.path.exists(manual_excel_path):
            raise HTTPException(status_code=404, detail=f"File not found: {manual_excel_path}")
        
        # Парсим даты
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Генерируем имя файла для отчета
        if not output_path:
            filename = f"comparison_{start_date}_to_{end_date}.xlsx"
            output_path = os.path.join("exports", filename)
            os.makedirs("exports", exist_ok=True)
        
        # Создаем компаратор
        comparator = ExcelComparator(db)
        
        # Выполняем сверку
        result_path = comparator.compare_and_generate_report(
            manual_excel_path=manual_excel_path,
            start_date=start_dt,
            end_date=end_dt,
            output_path=output_path
        )
        
        logger.info(f"📊 Comparison report generated: {result_path}")
        
        # Возвращаем файл
        return FileResponse(
            path=result_path,
            filename=os.path.basename(result_path),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing with Excel: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
