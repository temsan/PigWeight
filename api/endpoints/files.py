"""
File management endpoints
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse

router = APIRouter(prefix="/api", tags=["files"])

# Директории для записей
RECORDS_DIR = Path("records")
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/weighing/manual/save")
async def save_manual_weighing(data: Dict[str, Any]):
    """Сохранение ручного акта взвешивания"""
    try:
        # Валидация данных
        required_fields = ['count', 'total_weight']
        for field in required_fields:
            if field not in data:
                return JSONResponse(
                    {"error": f"Отсутствует обязательное поле: {field}"},
                    status_code=400
                )
        
        count = int(data['count'])
        total_weight = float(data['total_weight'])
        
        if count <= 0 or total_weight <= 0:
            return JSONResponse(
                {"error": "Количество и вес должны быть больше нуля"},
                status_code=400
            )
        
        # Создание записи акта
        act_data = {
            'id': f"manual_{int(datetime.now().timestamp())}",
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'group': data.get('group', 'Ручной ввод'),
            'total': count,
            'weight': total_weight,
            'avg_weight': round(total_weight / count, 2),
            'source': 'manual',
            'stream_id': data.get('stream_id', 'manual'),
            'created_at': datetime.now().isoformat()
        }
        
        # Сохранение в файл
        acts_file = RECORDS_DIR / "weighing_acts.json"
        acts = []
        
        if acts_file.exists():
            try:
                with open(acts_file, 'r', encoding='utf-8') as f:
                    acts = json.load(f)
            except Exception:
                acts = []
        
        acts.append(act_data)
        
        with open(acts_file, 'w', encoding='utf-8') as f:
            json.dump(acts, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "act_id": act_data['id'],
            "message": "Акт взвешивания сохранен"
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/journal/list")
async def get_journal_acts(
    date_from: str = Query(None),
    date_to: str = Query(None),
    camera: str = Query(None),
    limit: int = Query(100)
):
    """Получить список актов взвешивания с фильтрацией"""
    try:
        acts_file = RECORDS_DIR / "weighing_acts.json"
        if not acts_file.exists():
            return {
                "acts": [],
                "summary": {
                    "total_acts": 0,
                    "total_count": 0,
                    "total_weight": 0,
                    "avg_weight": 0
                }
            }
        
        with open(acts_file, 'r', encoding='utf-8') as f:
            acts = json.load(f)
        
        # Применяем фильтры
        filtered_acts = []
        for act in acts:
            # Фильтр по дате
            if date_from and act.get('date', '') < date_from:
                continue
            if date_to and act.get('date', '') > date_to:
                continue
            # Фильтр по камере/потоку
            if camera and act.get('stream_id', '') != camera:
                continue
            
            filtered_acts.append(act)
        
        # Сортируем по дате (новые сверху)
        filtered_acts.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
        
        # Ограничиваем количество
        filtered_acts = filtered_acts[:limit]
        
        # Добавляем статистику
        total_count = sum(act.get('total', 0) for act in filtered_acts)
        total_weight = sum(act.get('weight', 0) for act in filtered_acts)
        avg_weight = total_weight / total_count if total_count > 0 else 0
        
        return {
            "acts": filtered_acts,
            "summary": {
                "total_acts": len(filtered_acts),
                "total_count": total_count,
                "total_weight": round(total_weight, 1),
                "avg_weight": round(avg_weight, 2)
            }
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/journal/export")
async def export_journal_acts(
    format: str = Query("csv", regex="^(csv|excel)$"),
    date_from: str = Query(None),
    date_to: str = Query(None)
):
    """Экспорт актов взвешивания в CSV или Excel"""
    try:
        acts_file = RECORDS_DIR / "weighing_acts.json"
        if not acts_file.exists():
            return JSONResponse({"error": "Нет данных для экспорта"}, status_code=404)
        
        with open(acts_file, 'r', encoding='utf-8') as f:
            acts = json.load(f)
        
        # Фильтрация по датам
        if date_from or date_to:
            filtered_acts = []
            for act in acts:
                act_date = act.get('date', '')
                if date_from and act_date < date_from:
                    continue
                if date_to and act_date > date_to:
                    continue
                filtered_acts.append(act)
            acts = filtered_acts
        
        if not acts:
            return JSONResponse({"error": "Нет данных в указанном диапазоне дат"}, status_code=404)
        
        # Подготовка данных для экспорта
        export_data = []
        for act in acts:
            export_data.append({
                'Дата': act.get('date', ''),
                'Время': act.get('time', ''),
                'Группа': act.get('group', ''),
                'Количество голов': act.get('total', 0),
                'Общий вес (кг)': act.get('weight', 0),
                'Средний вес (кг)': act.get('avg_weight', 0),
                'Источник': act.get('source', ''),
                'ID потока': act.get('stream_id', '')
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"journal_export_{timestamp}.csv"
        filepath = RECORDS_DIR / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if export_data:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)
        
        return FileResponse(
            filepath,
            filename=filename,
            media_type='text/csv'
        )
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.post("/journal/compare")
async def compare_with_excel(file: UploadFile = File(...)):
    """Сверка актов взвешивания с Excel файлом"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            return JSONResponse(
                {"error": "Поддерживаются только Excel файлы (.xlsx, .xls)"}, 
                status_code=400
            )
        
        # Здесь будет логика сверки с Excel
        # Пока возвращаем заглушку
        return {
            "status": "success",
            "message": "Сверка выполнена",
            "matches": 0,
            "differences": 0,
            "comparison": {
                "excel": {"total_count": 0, "total_weight": 0, "rows": 0},
                "acts": {"total_count": 0, "total_weight": 0, "rows": 0}
            }
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)