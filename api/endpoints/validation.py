"""
API endpoints для сверки данных с Excel.
Парсинг, сравнение и генерация отчетов о расхождениях.
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Импорт валидатора и event logger
try:
    from services.excel_validator import get_excel_validator
    HAVE_VALIDATOR = True
except ImportError:
    HAVE_VALIDATOR = False
    logger.warning("ExcelValidator не доступен")

try:
    from services.event_logger import get_event_logger
    HAVE_EVENT_LOGGER = True
except ImportError:
    HAVE_EVENT_LOGGER = False


@router.get("/validation/excel/parse")
async def parse_excel_file(
    excel_path: Optional[str] = Query(None, description="Путь к Excel файлу")
):
    """Парсит Excel файл и возвращает данные"""
    try:
        if not HAVE_VALIDATOR:
            return JSONResponse(
                {"error": "ExcelValidator недоступен. Установите: pip install openpyxl"},
                status_code=503
            )
        
        validator = get_excel_validator(excel_path)
        if not validator:
            raise HTTPException(status_code=404, detail="Файл Excel не найден")
        
        # Парсим Excel
        raw_data = validator.parse_excel()
        normalized_data = validator.normalize_excel_data(raw_data)
        
        return {
            "success": True,
            "excel_path": str(validator.excel_path),
            "total_rows": len(raw_data),
            "normalized_rows": len(normalized_data),
            "data": normalized_data[:100],  # Ограничиваем для производительности
            "sample": normalized_data[0] if normalized_data else None
        }
        
    except FileNotFoundError as e:
        logger.error(f"Excel файл не найден: {e}")
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        logger.error(f"Ошибка парсинга Excel: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/validation/excel/compare")
async def compare_with_excel(
    stream_id: str = Query("cam1", description="ID потока для сверки"),
    excel_path: Optional[str] = Query(None, description="Путь к Excel файлу"),
    tolerance_minutes: int = Query(5, ge=1, le=60, description="Допустимая разница во времени (мин)")
):
    """Сравнивает данные Excel с журналом событий"""
    try:
        if not HAVE_VALIDATOR:
            return JSONResponse(
                {"error": "ExcelValidator недоступен. Установите: pip install openpyxl"},
                status_code=503
            )
        
        if not HAVE_EVENT_LOGGER:
            return JSONResponse(
                {"error": "EventLogger недоступен"},
                status_code=503
            )
        
        # Получаем валидатор и парсим Excel
        validator = get_excel_validator(excel_path)
        if not validator:
            raise HTTPException(status_code=404, detail="Файл Excel не найден")
        
        raw_excel = validator.parse_excel()
        excel_data = validator.normalize_excel_data(raw_excel)
        
        # Получаем события из журнала
        event_logger = get_event_logger()
        events = event_logger.get_events(stream_id=stream_id)
        events_data = [e.to_dict() for e in events]
        
        # Фильтруем только пиковые события для сверки
        peak_events = [e for e in events_data if e.get('event_type') == 'peak_count']
        
        # Сравниваем
        report = validator.compare_with_events(
            excel_data=excel_data,
            events_data=peak_events,
            tolerance_minutes=tolerance_minutes
        )
        
        # Добавляем метаданные
        report['meta'] = {
            'stream_id': stream_id,
            'excel_path': str(validator.excel_path),
            'tolerance_minutes': tolerance_minutes
        }
        
        return {
            "success": True,
            "report": report
        }
        
    except FileNotFoundError as e:
        logger.error(f"Excel файл не найден: {e}")
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        logger.error(f"Ошибка сверки с Excel: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/validation/excel/report")
async def generate_validation_report(
    stream_id: str = Query("cam1", description="ID потока"),
    excel_path: Optional[str] = Query(None, description="Путь к Excel файлу"),
    format: str = Query("json", regex="^(json|html)$", description="Формат отчета")
):
    """Генерирует детальный отчет сверки"""
    try:
        if not HAVE_VALIDATOR or not HAVE_EVENT_LOGGER:
            return JSONResponse(
                {"error": "Валидация недоступна"},
                status_code=503
            )
        
        # Получаем данные сверки
        validator = get_excel_validator(excel_path)
        if not validator:
            raise HTTPException(status_code=404, detail="Файл Excel не найден")
        
        raw_excel = validator.parse_excel()
        excel_data = validator.normalize_excel_data(raw_excel)
        
        event_logger = get_event_logger()
        events = event_logger.get_events(stream_id=stream_id)
        events_data = [e.to_dict() for e in events]
        peak_events = [e for e in events_data if e.get('event_type') == 'peak_count']
        
        report = validator.compare_with_events(excel_data, peak_events)
        
        if format == "html":
            # Генерируем HTML отчет
            html = generate_html_report(report, stream_id, str(validator.excel_path))
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=html)
        
        # JSON формат
        return {
            "success": True,
            "stream_id": stream_id,
            "excel_path": str(validator.excel_path),
            "report": report,
            "recommendations": generate_recommendations(report)
        }
        
    except Exception as e:
        logger.error(f"Ошибка генерации отчета: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def generate_html_report(report: dict, stream_id: str, excel_path: str) -> str:
    """Генерирует HTML отчет сверки"""
    
    summary = report.get('summary', {})
    matched = report.get('matched', [])
    unmatched_excel = report.get('unmatched_excel', [])
    unmatched_events = report.get('unmatched_events', [])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет сверки данных</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #666; margin-top: 30px; }}
            .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .stat-card {{ background: #f9f9f9; padding: 20px; border-radius: 6px; border-left: 4px solid #4CAF50; }}
            .stat-value {{ font-size: 32px; font-weight: bold; color: #333; }}
            .stat-label {{ color: #666; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background: #f5f5f5; }}
            .match-exact {{ background: #e8f5e9; }}
            .match-close {{ background: #fff9c4; }}
            .error {{ color: #d32f2f; }}
            .success {{ color: #388e3c; }}
            .warning {{ color: #f57c00; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Отчет сверки данных</h1>
            
            <p><strong>Поток:</strong> {stream_id}</p>
            <p><strong>Excel файл:</strong> {excel_path}</p>
            
            <div class="summary">
                <div class="stat-card">
                    <div class="stat-value">{summary.get('matched_count', 0)}</div>
                    <div class="stat-label">Совпадений</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('unmatched_excel_count', 0)}</div>
                    <div class="stat-label">Не найдено в журнале</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('unmatched_events_count', 0)}</div>
                    <div class="stat-label">Не найдено в Excel</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('match_rate', 0):.1f}%</div>
                    <div class="stat-label">Процент совпадений</div>
                </div>
            </div>
            
            <h2 class="success">✅ Совпадающие записи ({len(matched)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Дата</th>
                        <th>Время</th>
                        <th>Excel (кол-во)</th>
                        <th>Система (кол-во)</th>
                        <th>Excel (вес)</th>
                        <th>Качество</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'''
                    <tr class="match-{m.get('match_quality', 'exact')}">
                        <td>{m.get('date', 'N/A')}</td>
                        <td>{m.get('time', 'N/A')}</td>
                        <td>{m.get('excel_count', 0)}</td>
                        <td>{m.get('event_count', 0)}</td>
                        <td>{m.get('excel_weight', 0)}</td>
                        <td>{m.get('match_quality', 'exact')}</td>
                    </tr>
                    ''' for m in matched[:50]])}
                </tbody>
            </table>
            
            <h2 class="error">❌ Не найдено в журнале ({len(unmatched_excel)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Строка</th>
                        <th>Дата</th>
                        <th>Время</th>
                        <th>Количество</th>
                        <th>Причина</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'''
                    <tr>
                        <td>{u.get('row', 'N/A')}</td>
                        <td>{u.get('data', {}).get('date', 'N/A')}</td>
                        <td>{u.get('data', {}).get('time', 'N/A')}</td>
                        <td>{u.get('data', {}).get('count', 0)}</td>
                        <td>{u.get('reason', 'N/A')}</td>
                    </tr>
                    ''' for u in unmatched_excel[:50]])}
                </tbody>
            </table>
            
            <h2 class="warning">⚠️ Не найдено в Excel ({len(unmatched_events)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID события</th>
                        <th>Дата</th>
                        <th>Время</th>
                        <th>Количество</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'''
                    <tr>
                        <td>{u.get('event_id', 'N/A')}</td>
                        <td>{u.get('date', 'N/A')}</td>
                        <td>{u.get('time', 'N/A')}</td>
                        <td>{u.get('count', 0)}</td>
                    </tr>
                    ''' for u in unmatched_events[:50]])}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_recommendations(report: dict) -> List[str]:
    """Генерирует рекомендации на основе отчета"""
    recommendations = []
    summary = report.get('summary', {})
    
    match_rate = summary.get('match_rate', 0)
    
    if match_rate < 50:
        recommendations.append("Низкий процент совпадений. Проверьте корректность времени в Excel и системе.")
    elif match_rate < 80:
        recommendations.append("Средний процент совпадений. Возможны расхождения в методах подсчета.")
    else:
        recommendations.append("Высокий процент совпадений. Системаработает корректно.")
    
    if summary.get('unmatched_excel_count', 0) > 10:
        recommendations.append(f"Много записей из Excel не найдено в системе ({summary['unmatched_excel_count']}). Проверьте период журналирования.")
    
    if summary.get('unmatched_events_count', 0) > 10:
        recommendations.append(f"Много событий системы не найдено в Excel ({summary['unmatched_events_count']}). Возможно, не все измерения записаны.")
    
    return recommendations

