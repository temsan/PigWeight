"""
Excel comparison endpoints
Сверка автоматических результатов с ручными записями

СОЗДАНО: 8 ноября 2025
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from api.dependencies import get_database_manager

router = APIRouter(prefix="/api/compare", tags=["compare"])
logger = logging.getLogger(__name__)


@router.post("/excel")
async def compare_with_excel(file: UploadFile = File(...)):
    """
    Сверка автоматических результатов с ручными записями из Excel
    
    Upload:
    - file: Excel файл с ручными записями
    
    Response:
    {
        "matches": 15,
        "discrepancies": 3,
        "missing_in_auto": 1,
        "missing_in_manual": 2,
        "accuracy": 0.85,
        "metrics": {
            "recall": 0.88,
            "precision": 0.90,
            "f1_score": 0.89,
            "mae": 2.5,
            "mape": 5.2,
            "correlation": 0.95
        },
        "report_url": "/exports/comparison_2025-11-08_10-30-00.xlsx"
    }
    
    Требования: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    try:
        # Проверяем тип файла
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Неверный формат файла. Требуется .xlsx или .xls"
            )
        
        # Сохраняем загруженный файл
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        temp_path = upload_dir / file.filename
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"📥 Загружен файл для сверки: {file.filename}")
        
        # Импортируем необходимые модули
        try:
            from pig_tracking.excel_comparator import ExcelComparator
            from pig_tracking.excel_analyzer import ExcelAnalyzer
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Модули сверки не установлены: {e}"
            )
        
        # Парсим Excel файл
        analyzer = ExcelAnalyzer()
        manual_data = analyzer.parse_data(str(temp_path))
        
        if not manual_data:
            raise HTTPException(
                status_code=400,
                detail="Не удалось распарсить Excel файл или файл пустой"
            )
        
        logger.info(f"📊 Распарсено {len(manual_data)} записей из Excel")
        
        # Определяем период из Excel данных
        dates = [record.get('date') for record in manual_data if record.get('date')]
        if not dates:
            raise HTTPException(
                status_code=400,
                detail="В Excel файле не найдены даты"
            )
        
        start_date = min(dates)
        end_date = max(dates)
        
        # Получаем автоматические акты из БД
        db = get_database_manager()
        auto_acts = db.get_acts_by_period(
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"📊 Получено {len(auto_acts)} автоматических актов из БД")
        
        # Выполняем сверку
        comparator = ExcelComparator(time_tolerance_minutes=5)
        comparison = comparator.compare(
            auto_acts=auto_acts,
            manual_data=manual_data
        )
        
        # Генерируем отчёт
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"comparison_{timestamp}.xlsx"
        report_path = export_dir / report_filename
        
        comparator.generate_report(
            comparison=comparison,
            output_path=str(report_path)
        )
        
        logger.info(f"✅ Сверка завершена: {report_filename}")
        
        # Формируем ответ
        response = {
            "matches": len(comparison.get("matches", [])),
            "discrepancies": len(comparison.get("discrepancies", [])),
            "missing_in_auto": len(comparison.get("missing_in_auto", [])),
            "missing_in_manual": len(comparison.get("missing_in_manual", [])),
            "accuracy": comparison.get("metrics", {}).get("accuracy", 0.0),
            "metrics": comparison.get("metrics", {}),
            "report_url": f"/exports/{report_filename}"
        }
        
        # Удаляем временный файл
        try:
            temp_path.unlink()
        except Exception as e:
            logger.warning(f"⚠️ Не удалось удалить временный файл: {e}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка сверки: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при сверке: {str(e)}"
        )


@router.get("/reports/{filename}")
async def get_comparison_report(filename: str):
    """
    Скачать отчёт о сверке
    
    Path params:
    - filename: имя файла отчёта
    
    Response:
    - Excel файл с отчётом
    """
    try:
        from fastapi.responses import FileResponse
        
        report_path = Path("exports") / filename
        
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Отчёт не найден"
            )
        
        return FileResponse(
            path=str(report_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка получения отчёта: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении отчёта: {str(e)}"
        )
