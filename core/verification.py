import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False
    pd = None
    np = None

logger = logging.getLogger(__name__)

class WeighingVerification:
    """
    Система для верификации актов взвешивания свиней.
    Проверяет соответствие счетчиков слева/справа с учетом погрешности.
    """

    def __init__(self, records_dir: str = "records", tolerance: float = 0.1):
        """
        Инициализация системы верификации

        Args:
            records_dir: Директория с записями актов взвешивания
            tolerance: Допустимая погрешность (относительная, 0.1 = 10%)
        """
        self.records_dir = Path(records_dir)
        self.tolerance = tolerance
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def verify_weighing_act(self, act_file: str) -> Dict[str, Any]:
        """
        Простая проверка акта взвешивания на соответствие счетчиков

        Args:
            act_file: Имя файла акта взвешивания (без расширения)

        Returns:
            Словарь с результатами проверки
        """
        json_file = self.records_dir / f"{act_file}.json"

        if not json_file.exists():
            return {
                "status": "error",
                "message": f"Файл акта не найден: {json_file}",
                "verified": False
            }

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            left_in = data.get("flow", {}).get("left_in", 0)
            right_in = data.get("flow", {}).get("right_in", 0)

            # Простая проверка счетчиков
            verification_result = self._verify_counters(left_in, right_in)

            result = {
                "status": "success",
                "act_file": act_file,
                "stream_id": data.get("stream_id"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "duration_sec": data.get("duration_sec", 0),
                "seen_total": data.get("seen_total", 0),
                "flow": {
                    "left_in": left_in,
                    "right_in": right_in
                },
                "verification": verification_result
            }

            return result

        except Exception as e:
            logger.error(f"Ошибка при проверке акта {act_file}: {e}")
            return {
                "status": "error",
                "message": f"Ошибка при обработке файла: {str(e)}",
                "verified": False
            }

    def _verify_counters(self, left_count: int, right_count: int) -> Dict[str, Any]:
        """
        Проверяет соответствие счетчиков слева и справа

        Args:
            left_count: Количество входов слева
            right_count: Количество выходов справа

        Returns:
            Результаты проверки
        """
        total_count = left_count + right_count

        if total_count == 0:
            return {
                "verified": True,
                "status": "empty",
                "message": "Нет данных для проверки",
                "difference": 0,
                "tolerance": 0
            }

        # Вычисляем разницу с учетом погрешности
        diff = abs(left_count - right_count)
        max_count = max(left_count, right_count)

        if max_count == 0:
            relative_diff = 0.0
        else:
            relative_diff = diff / max_count

        verified = relative_diff <= self.tolerance

        if verified:
            status = "verified"
            message = f"Счетчики в пределах нормы (разница {relative_diff:.1%})"
        else:
            status = "discrepancy"
            message = f"Расхождение счетчиков: {relative_diff:.1%}"
            # Для больших расхождений - предупреждение
            if relative_diff > 0.5:
                status = "warning"
                message += " (значительное расхождение)"

        return {
            "verified": verified,
            "status": status,
            "message": message,
            "difference": diff,
            "relative_difference": relative_diff,
            "tolerance": self.tolerance,
            "expected_balance": f"{left_count} слева ≈ {right_count} справа"
        }

    def verify_all_acts(self) -> List[Dict[str, Any]]:
        """
        Проверяет все акты взвешивания в директории

        Returns:
            Список результатов проверки всех актов
        """
        results = []
        json_files = list(self.records_dir.glob("act_*.json"))

        for json_file in json_files:
            act_name = json_file.stem  # без расширения
            result = self.verify_weighing_act(act_name)
            results.append(result)

        # Сортируем по времени завершения (новые сверху)
        results.sort(key=lambda x: x.get("finished_at", 0), reverse=True)

        return results

    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Простая статистика по всем проверенным актам

        Returns:
            Основная статистика верификации
        """
        all_results = self.verify_all_acts()

        stats = {
            "total_acts": len(all_results),
            "verified_count": 0,
            "discrepancy_count": 0,
            "error_count": 0,
            "total_pigs": 0,
            "results": all_results
        }

        for result in all_results:
            if result["status"] == "success":
                verification = result.get("verification", {})
                if verification.get("verified"):
                    stats["verified_count"] += 1
                else:
                    stats["discrepancy_count"] += 1

                stats["total_pigs"] += result.get("seen_total", 0)
            else:
                stats["error_count"] += 1

        return stats


    def analyze_excel_measurements(self, excel_path: str) -> Dict[str, Any]:
        """
        Простой анализ файла с замерами Excel

        Args:
            excel_path: Путь к Excel файлу

        Returns:
            Основные результаты анализа файла
        """
        if not _HAVE_PANDAS:
            return {
                "status": "error",
                "message": "Библиотека pandas не установлена. Установите pandas для анализа Excel файлов."
            }

        try:
            # Читаем Excel файл
            df = pd.read_excel(excel_path)

            analysis = {
                "file": Path(excel_path).name,
                "total_rows": len(df),
                "columns": list(df.columns)
            }

            # Ищем числовые столбцы с весом и количеством
            weight_cols = [col for col in df.columns if any(word in col.lower() for word in ['вес', 'weight', 'кг'])]
            count_cols = [col for col in df.columns if any(word in col.lower() for word in ['количество', 'count', 'шт'])]
            
            if weight_cols:
                analysis["total_weight"] = float(df[weight_cols[0]].sum())
            if count_cols:
                analysis["total_count"] = int(df[count_cols[0]].sum())

            return {
                "status": "success",
                "analysis": analysis
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Ошибка при анализе Excel файла: {str(e)}"
            }





# Глобальный экземпляр для использования в API
verification_system = WeighingVerification()
