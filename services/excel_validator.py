"""
Сервис для сверки данных журнала событий с Excel файлом.
Парсит Excel, сравнивает с событиями и генерирует отчет о расхождениях.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)

# Опциональный импорт openpyxl
try:
    import openpyxl
    from openpyxl.worksheet.worksheet import Worksheet
    HAVE_OPENPYXL = True
except ImportError:
    HAVE_OPENPYXL = False
    logger.warning("openpyxl не установлен. Функция сверки с Excel недоступна.")


class ExcelValidator:
    """
    Валидатор данных из Excel файла.
    Сравнивает данные взвешивания с журналом событий системы.
    """
    
    def __init__(self, excel_path: str = "docs/Замеры 20.07 по 03.09.xlsx"):
        self.excel_path = Path(excel_path)
        
        if not HAVE_OPENPYXL:
            raise ImportError("openpyxl не установлен. Установите: pip install openpyxl")
        
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel файл не найден: {excel_path}")
        
        logger.info(f"ExcelValidator инициализирован: {excel_path}")
    
    def parse_excel(self) -> List[Dict[str, Any]]:
        """Парсит Excel файл и возвращает список измерений"""
        try:
            workbook = openpyxl.load_workbook(self.excel_path, data_only=True)
            sheet = workbook.active
            
            measurements = []
            headers = []
            
            # Читаем заголовки (предполагаем, что они в первой строке)
            for col_idx, cell in enumerate(sheet[1], start=1):
                headers.append(cell.value if cell.value else f"col_{col_idx}")
            
            logger.info(f"Найдены заголовки: {headers}")
            
            # Читаем данные (начиная со второй строки)
            for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
                if not any(row):  # Пропускаем пустые строки
                    continue
                
                # Создаем словарь из строки
                measurement = {}
                for col_idx, value in enumerate(row):
                    if col_idx < len(headers):
                        key = headers[col_idx]
                        measurement[key] = value
                
                # Добавляем номер строки для отладки
                measurement['_row'] = row_idx
                measurements.append(measurement)
            
            logger.info(f"Загружено {len(measurements)} измерений из Excel")
            return measurements
            
        except Exception as e:
            logger.error(f"Ошибка парсинга Excel: {e}", exc_info=True)
            raise
    
    def normalize_excel_data(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Нормализует данные из Excel в единый формат"""
        normalized = []
        
        for m in measurements:
            try:
                # Пытаемся извлечь дату, время, количество, вес
                # Формат может варьироваться, поэтому проверяем разные варианты ключей
                
                date_val = self._extract_value(m, ['Дата', 'Date', 'дата', 'date'])
                time_val = self._extract_value(m, ['Время', 'Time', 'время', 'time'])
                count_val = self._extract_value(m, ['Количество', 'Count', 'количество', 'count', 'Кол-во'])
                weight_val = self._extract_value(m, ['Вес', 'Weight', 'вес', 'weight', 'Масса'])
                
                # Парсим дату
                if date_val:
                    if isinstance(date_val, datetime):
                        date_str = date_val.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date_val)
                else:
                    date_str = None
                
                # Парсим время
                if time_val:
                    if isinstance(time_val, datetime):
                        time_str = time_val.strftime('%H:%M:%S')
                    else:
                        time_str = str(time_val)
                else:
                    time_str = None
                
                # Парсим количество и вес
                count = int(count_val) if count_val else 0
                weight = float(weight_val) if weight_val else 0.0
                
                normalized.append({
                    'date': date_str,
                    'time': time_str,
                    'count': count,
                    'weight': weight,
                    '_row': m.get('_row'),
                    '_raw': m
                })
                
            except Exception as e:
                logger.warning(f"Ошибка нормализации строки {m.get('_row')}: {e}")
                continue
        
        logger.info(f"Нормализовано {len(normalized)} измерений")
        return normalized
    
    def _extract_value(self, data: Dict[str, Any], keys: List[str]) -> Any:
        """Извлекает значение по списку возможных ключей"""
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None
    
    def compare_with_events(self, 
                           excel_data: List[Dict[str, Any]], 
                           events_data: List[Dict[str, Any]],
                           tolerance_minutes: int = 5) -> Dict[str, Any]:
        """
        Сравнивает данные Excel с событиями системы.
        
        Args:
            excel_data: Нормализованные данные из Excel
            events_data: События из журнала системы
            tolerance_minutes: Допустимая разница во времени (минуты)
        
        Returns:
            Отчет о сверке с расхождениями
        """
        
        report = {
            'excel_total': len(excel_data),
            'events_total': len(events_data),
            'matched': [],
            'unmatched_excel': [],
            'unmatched_events': [],
            'discrepancies': []
        }
        
        # Создаем индекс событий по дате и времени
        events_by_datetime = {}
        for event in events_data:
            if event.get('timestamp'):
                dt = datetime.fromtimestamp(event['timestamp'])
                key = dt.strftime('%Y-%m-%d %H:%M')
                if key not in events_by_datetime:
                    events_by_datetime[key] = []
                events_by_datetime[key].append(event)
        
        # Сопоставляем данные Excel с событиями
        matched_events = set()
        
        for excel_row in excel_data:
            date = excel_row.get('date')
            time = excel_row.get('time')
            
            if not date or not time:
                report['unmatched_excel'].append({
                    'row': excel_row.get('_row'),
                    'reason': 'Отсутствует дата или время',
                    'data': excel_row
                })
                continue
            
            # Формируем ключ для поиска
            try:
                # Парсим время
                if ':' in str(time):
                    time_parts = str(time).split(':')
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                else:
                    hour = minute = 0
                
                search_key = f"{date} {hour:02d}:{minute:02d}"
                
                # Ищем совпадение
                found = False
                for event in events_by_datetime.get(search_key, []):
                    if event in matched_events:
                        continue
                    
                    # Сравниваем количество (с допуском ±1)
                    excel_count = excel_row.get('count', 0)
                    event_count = event.get('pig_count', 0)
                    
                    if abs(excel_count - event_count) <= 1:
                        matched_events.add(event)
                        report['matched'].append({
                            'excel_row': excel_row.get('_row'),
                            'event_id': event.get('event_id'),
                            'date': date,
                            'time': search_key,
                            'excel_count': excel_count,
                            'event_count': event_count,
                            'excel_weight': excel_row.get('weight'),
                            'match_quality': 'exact' if excel_count == event_count else 'close'
                        })
                        found = True
                        break
                
                if not found:
                    report['unmatched_excel'].append({
                        'row': excel_row.get('_row'),
                        'reason': 'Событие не найдено в журнале',
                        'data': {
                            'date': date,
                            'time': search_key,
                            'count': excel_row.get('count'),
                            'weight': excel_row.get('weight')
                        }
                    })
                    
            except Exception as e:
                logger.warning(f"Ошибка сопоставления строки {excel_row.get('_row')}: {e}")
                report['unmatched_excel'].append({
                    'row': excel_row.get('_row'),
                    'reason': f'Ошибка обработки: {e}',
                    'data': excel_row
                })
        
        # Находим события без соответствия в Excel
        for event in events_data:
            if event not in matched_events and event.get('event_type') == 'peak_count':
                dt = datetime.fromtimestamp(event.get('timestamp', 0))
                report['unmatched_events'].append({
                    'event_id': event.get('event_id'),
                    'date': dt.strftime('%Y-%m-%d'),
                    'time': dt.strftime('%H:%M:%S'),
                    'count': event.get('pig_count'),
                    'reason': 'Нет соответствия в Excel'
                })
        
        # Статистика
        report['summary'] = {
            'matched_count': len(report['matched']),
            'unmatched_excel_count': len(report['unmatched_excel']),
            'unmatched_events_count': len(report['unmatched_events']),
            'match_rate': len(report['matched']) / len(excel_data) * 100 if excel_data else 0
        }
        
        logger.info(f"Сверка завершена: {report['summary']}")
        return report


def get_excel_validator(excel_path: Optional[str] = None) -> Optional[ExcelValidator]:
    """Получает экземпляр валидатора Excel"""
    if not HAVE_OPENPYXL:
        logger.error("openpyxl не установлен")
        return None
    
    try:
        path = excel_path or "docs/Замеры 20.07 по 03.09.xlsx"
        return ExcelValidator(path)
    except Exception as e:
        logger.error(f"Ошибка создания ExcelValidator: {e}")
        return None

