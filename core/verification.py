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
        Проверяет акт взвешивания на соответствие счетчиков

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

            # Проверяем соответствие счетчиков с учетом погрешности
            verification_result = self._verify_counters(left_in, right_in)

            # Анализируем прохождение через рамки
            frame_analysis = self._analyze_frame_crossings(data)

            # Анализируем всплески количества
            spike_analysis = self._analyze_spikes(data)

            result = {
                "status": "success",
                "act_file": act_file,
                "stream_id": data.get("stream_id"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "duration_sec": data.get("duration_sec", 0),
                "seen_total": data.get("seen_total", 0),
                "peak_concurrent": data.get("peak_concurrent", 0),
                "flow": {
                    "left_in": left_in,
                    "right_in": right_in
                },
                "verification": verification_result,
                "frame_analysis": frame_analysis,
                "spike_analysis": spike_analysis
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
            message = ".1f"        else:
            status = "discrepancy"
            message = ".1f"
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
        Получает статистику по всем проверенным актам

        Returns:
            Статистика верификации
        """
        all_results = self.verify_all_acts()

        stats = {
            "total_acts": len(all_results),
            "verified_count": 0,
            "discrepancy_count": 0,
            "error_count": 0,
            "total_pigs": 0,
            "avg_duration": 0,
            "results": [],
            "grouped_by_date": {}
        }

        total_duration = 0
        valid_acts = 0

        for result in all_results:
            if result["status"] == "success":
                verification = result.get("verification", {})
                if verification.get("verified"):
                    stats["verified_count"] += 1
                else:
                    stats["discrepancy_count"] += 1

                stats["total_pigs"] += result.get("seen_total", 0)
                duration = result.get("duration_sec", 0)
                if duration > 0:
                    total_duration += duration
                    valid_acts += 1
            else:
                stats["error_count"] += 1

            stats["results"].append(result)

        if valid_acts > 0:
            stats["avg_duration"] = total_duration / valid_acts

        # Группируем по датам
        stats["grouped_by_date"] = self._group_results_by_date(stats["results"])

        return stats

    def _group_results_by_date(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Группирует результаты по датам

        Args:
            results: Список результатов актов

        Returns:
            Сгруппированные результаты по датам
        """
        grouped = {}

        for result in results:
            if result["status"] != "success":
                continue

            # Извлекаем дату из имени файла или времени завершения
            act_file = result.get("act_file", "")
            finished_at = result.get("finished_at", 0)

            # Пытаемся извлечь дату из имени файла (формат: act_stream_timestamp)
            date_key = None
            if "_" in act_file:
                parts = act_file.split("_")
                if len(parts) >= 3:
                    # act_stream_timestamp_date-time
                    timestamp_part = parts[2] if len(parts) > 2 else ""
                    if len(timestamp_part) >= 8:  # YYYYMMDD
                        try:
                            year = int(timestamp_part[:4])
                            month = int(timestamp_part[4:6])
                            day = int(timestamp_part[6:8])
                            date_key = ".4d"
                        except (ValueError, IndexError):
                            pass

            # Если не удалось извлечь из имени файла, используем timestamp
            if date_key is None and finished_at > 0:
                from datetime import datetime
                dt = datetime.fromtimestamp(finished_at)
                date_key = dt.strftime("%Y-%m-%d")

            if date_key is None:
                date_key = "unknown"

            if date_key not in grouped:
                grouped[date_key] = {
                    "date": date_key,
                    "total_acts": 0,
                    "verified_acts": 0,
                    "discrepancy_acts": 0,
                    "total_pigs": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "acts": []
                }

            group = grouped[date_key]
            group["total_acts"] += 1
            group["total_pigs"] += result.get("seen_total", 0)
            group["total_duration"] += result.get("duration_sec", 0)

            verification = result.get("verification", {})
            if verification.get("verified"):
                group["verified_acts"] += 1
            else:
                group["discrepancy_acts"] += 1

            group["acts"].append(result)

        # Вычисляем среднюю длительность для каждой группы
        for group in grouped.values():
            if group["total_acts"] > 0:
                group["avg_duration"] = group["total_duration"] / group["total_acts"]

        # Сортируем группы по дате (новые сверху)
        sorted_groups = {}
        for date_key in sorted(grouped.keys(), reverse=True):
            sorted_groups[date_key] = grouped[date_key]

        return sorted_groups

    def analyze_excel_measurements(self, excel_path: str) -> Dict[str, Any]:
        """
        Анализирует файл с замерами Excel

        Args:
            excel_path: Путь к Excel файлу

        Returns:
            Результаты анализа файла
        """
        if not _HAVE_PANDAS:
            return {
                "status": "error",
                "message": "Библиотека pandas не установлена. Установите pandas для анализа Excel файлов."
            }

        try:
            # Попытка прочитать Excel файл
            df = pd.read_excel(excel_path)

            analysis = {
                "file": Path(excel_path).name,
                "total_rows": len(df),
                "columns": list(df.columns),
                "summary": {}
            }

            # Анализ числовых столбцов
            for col in df.select_dtypes(include=[np.number]).columns:
                analysis["summary"][col] = {
                    "count": int(df[col].count()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }

            return {
                "status": "success",
                "analysis": analysis
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Ошибка при анализе Excel файла: {str(e)}"
            }

    def _analyze_frame_crossings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует прохождение свиней через рамки (вертикальные полосы)

        Args:
            data: Данные акта взвешивания

        Returns:
            Анализ прохождения через рамки
        """
        crossings = data.get("crossings", [])

        if not crossings:
            return {
                "total_crossings": 0,
                "left_crossings": 0,
                "right_crossings": 0,
                "avg_crossing_time": 0,
                "crossings_per_minute": 0,
                "peak_crossing_periods": []
            }

        left_crossings = [c for c in crossings if c.get("side") == "left" and c.get("mode") == "enter"]
        right_crossings = [c for c in crossings if c.get("side") == "right" and c.get("mode") == "enter"]

        duration_sec = data.get("duration_sec", 1)

        # Вычисляем среднее время между пересечениями
        if len(crossings) > 1:
            times = sorted([c.get("t", 0) for c in crossings])
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_crossing_time = sum(intervals) / len(intervals) if intervals else 0
        else:
            avg_crossing_time = 0

        # Находим периоды с пиковым количеством пересечений
        peak_periods = self._find_peak_crossing_periods(crossings, duration_sec)

        return {
            "total_crossings": len(crossings),
            "left_crossings": len(left_crossings),
            "right_crossings": len(right_crossings),
            "avg_crossing_time": avg_crossing_time,
            "crossings_per_minute": (len(crossings) / duration_sec) * 60,
            "peak_crossing_periods": peak_periods
        }

    def _analyze_spikes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует всплески количества свиней

        Args:
            data: Данные акта взвешивания

        Returns:
            Анализ всплесков количества
        """
        timeline = data.get("timeline", [])

        if not timeline:
            return {
                "spike_count": 0,
                "avg_spike_intensity": 0,
                "max_spike_intensity": 0,
                "spike_periods": []
            }

        # Извлекаем значения количества из таймлайна
        counts = [point.get("count_est", 0) for point in timeline]

        if not counts:
            return {
                "spike_count": 0,
                "avg_spike_intensity": 0,
                "max_spike_intensity": 0,
                "spike_periods": []
            }

        # Вычисляем базовое среднее и стандартное отклонение
        mean_count = sum(counts) / len(counts)
        variance = sum((x - mean_count) ** 2 for x in counts) / len(counts)
        std_dev = variance ** 0.5

        # Определяем порог для всплеска (среднее + 2 стандартных отклонения)
        spike_threshold = mean_count + 2 * std_dev

        # Находим всплески
        spikes = []
        for i, count in enumerate(counts):
            if count > spike_threshold:
                time_point = timeline[i].get("t", 0)
                spikes.append({
                    "time": time_point,
                    "intensity": count,
                    "above_mean": count - mean_count
                })

        # Группируем близкие всплески в периоды
        spike_periods = self._group_spikes_into_periods(spikes)

        return {
            "spike_count": len(spikes),
            "avg_spike_intensity": sum(s["intensity"] for s in spikes) / len(spikes) if spikes else 0,
            "max_spike_intensity": max((s["intensity"] for s in spikes), default=0),
            "spike_threshold": spike_threshold,
            "base_mean": mean_count,
            "std_dev": std_dev,
            "spike_periods": spike_periods
        }

    def _find_peak_crossing_periods(self, crossings: List[Dict[str, Any]], duration_sec: float,
                                   window_size_sec: float = 30.0) -> List[Dict[str, Any]]:
        """
        Находит периоды с пиковым количеством пересечений

        Args:
            crossings: Список пересечений
            duration_sec: Общая длительность
            window_size_sec: Размер окна для анализа (секунды)

        Returns:
            Список периодов с пиковыми пересечениями
        """
        if not crossings:
            return []

        # Создаем временные окна
        windows = []
        start_time = 0

        while start_time < duration_sec:
            end_time = min(start_time + window_size_sec, duration_sec)

            # Считаем пересечения в этом окне
            window_crossings = [
                c for c in crossings
                if start_time <= c.get("t", 0) < end_time
            ]

            if window_crossings:
                windows.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "crossing_count": len(window_crossings),
                    "crossings_per_minute": (len(window_crossings) / (end_time - start_time)) * 60
                })

            start_time = end_time

        # Сортируем по интенсивности и возвращаем топ-5
        windows.sort(key=lambda x: x["crossing_count"], reverse=True)
        return windows[:5]

    def _group_spikes_into_periods(self, spikes: List[Dict[str, Any]],
                                  time_threshold_sec: float = 10.0) -> List[Dict[str, Any]]:
        """
        Группирует отдельные всплески в периоды

        Args:
            spikes: Список всплесков
            time_threshold_sec: Порог времени для группировки (секунды)

        Returns:
            Список периодов всплесков
        """
        if not spikes:
            return []

        # Сортируем всплески по времени
        sorted_spikes = sorted(spikes, key=lambda x: x["time"])

        periods = []
        current_period = {
            "start_time": sorted_spikes[0]["time"],
            "end_time": sorted_spikes[0]["time"],
            "spike_count": 1,
            "max_intensity": sorted_spikes[0]["intensity"],
            "total_intensity": sorted_spikes[0]["intensity"]
        }

        for spike in sorted_spikes[1:]:
            time_diff = spike["time"] - current_period["end_time"]

            if time_diff <= time_threshold_sec:
                # Продлеваем текущий период
                current_period["end_time"] = spike["time"]
                current_period["spike_count"] += 1
                current_period["max_intensity"] = max(current_period["max_intensity"], spike["intensity"])
                current_period["total_intensity"] += spike["intensity"]
            else:
                # Завершаем текущий период и начинаем новый
                current_period["duration"] = current_period["end_time"] - current_period["start_time"]
                current_period["avg_intensity"] = current_period["total_intensity"] / current_period["spike_count"]
                periods.append(current_period)

                current_period = {
                    "start_time": spike["time"],
                    "end_time": spike["time"],
                    "spike_count": 1,
                    "max_intensity": spike["intensity"],
                    "total_intensity": spike["intensity"]
                }

        # Добавляем последний период
        if current_period["spike_count"] > 0:
            current_period["duration"] = current_period["end_time"] - current_period["start_time"]
            current_period["avg_intensity"] = current_period["total_intensity"] / current_period["spike_count"]
            periods.append(current_period)

        return periods

# Глобальный экземпляр для использования в API
verification_system = WeighingVerification()
