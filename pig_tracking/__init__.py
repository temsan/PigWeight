"""
Модули для отслеживания и взвешивания свиней.
"""

from pig_tracking.crossing_counter import CrossingCounter, CrossingEvent
from pig_tracking.act_detector import ActDetector, WeighingAct
from pig_tracking.video_processor import IntegratedVideoProcessor
from pig_tracking.excel_analyzer import ExcelAnalyzer
from pig_tracking.excel_exporter import ExcelExporter
from pig_tracking.excel_comparator import ExcelComparator, ComparisonResult

__all__ = [
    'CrossingCounter',
    'CrossingEvent',
    'ActDetector',
    'WeighingAct',
    'IntegratedVideoProcessor',
    'ExcelAnalyzer',
    'ExcelExporter',
    'ExcelComparator',
    'ComparisonResult'
]
