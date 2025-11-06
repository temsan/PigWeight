"""
API endpoints для экспорта и сверки данных
"""
from fastapi import APIRouter, Query, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime
from pathlib import Path
import logging
import os

from pig_tracking.database_manager import DatabaseManager
from pig_tracking.excel_exporter import ExcelExporter
from pig_tracking.excel_comparator import ExcelComparator
from pig_tracking.excel_analyzer import ExcelAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["export"])

# Глобальный экземпляр DatabaseManager
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Получить экземпляр DatabaseManager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_KEY")
        )
    return _db_manager


@router.post("/excel")
async def export_to_excel(
    start_date: str = Query(..., description="Дата начала (ISO format)"),
    end_date: str = Query(..., description="Дата окончания (ISO format)"),
    stream_id: Optional[str] = Query(None, description="ID потока")
):
    """
    Экспорт актов взвешивания в Excel
    
    Соответствует спецификации: Требования 4.1-4.5
    """
    try:
        db = get_db_manager()
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Получить акты
        acts = db.get_acts_by_period(start, end)
        
        if stream_id:
            acts = [a for a in acts if a.stream_id == stream_id]
        
        if not acts:
            raise HTTPException(
                status_code=404,
                detail="Нет актов за указанный период"
            )
        
        # Создать файл в uploads/
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        output_path = uploads_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Экспорт
        exporter = ExcelExporter()
        exporter.export_to_excel(acts, str(output_path), group_by_date=True)
        
        # Имя файла для скачивания
        filename = f"weighing_acts_{start_date}_{end_date}.xlsx"
        
        return FileResponse(
            path=str(output_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат даты: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_with_excel(
    file: UploadFile = File(..., description="Excel файл с ручными записями"),
    tolerance_minutes: int = Query(5, description="Допуск по времени в минутах")
):
    """
    Сверка автоматических результатов с ручными записями из Excel
    
    Соответствует спецификации: Требования 5.1-5.6
    """
    try:
        db = get_db_manager()
        
        # Проверить формат файла
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Неверный формат файла. Ожидается .xlsx или .xls"
            )
        
        # Сохранить загруженный файл в uploads/
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        input_path = uploads_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(input_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Парсить Excel
        analyzer = ExcelAnalyzer()
        manual_acts = analyzer.parse_data(str(input_path))
        
        if not manual_acts:
            raise HTTPException(
                status_code=400,
                detail="Не удалось извлечь данные из Excel файла"
            )
        
        # Определить период из Excel
        dates = [act.started_at for act in manual_acts]
        start = min(dates)
        end = max(dates)
        
        # Получить автоматические акты за тот же период
        auto_acts = db.get_acts_by_period(start, end)
        
        if not auto_acts:
            raise HTTPException(
                status_code=404,
                detail="Нет автоматических актов за период из Excel файла"
            )
        
        # Сверка
        comparator = ExcelComparator(time_tolerance_minutes=tolerance_minutes)
        comparison = comparator.compare(auto_acts, manual_acts)
        
        # Создать отчет в uploads/
        output_path = uploads_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        comparator.generate_report(comparison, str(output_path))
        
        # Вернуть метрики и ссылку на отчет
        return {
            "status": "success",
            "metrics": comparison.metrics,
            "summary": {
                "matches": len(comparison.matches),
                "discrepancies": len(comparison.discrepancies),
                "missing_in_auto": len(comparison.missing_in_auto),
                "missing_in_manual": len(comparison.missing_in_manual)
            },
            "report_file": str(output_path.name),
            "download_url": f"/api/export/download/{output_path.name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing with Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Скачать сгенерированный файл из uploads/
    """
    try:
        uploads_dir = Path("uploads")
        file_path = uploads_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Файл не найден")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
