#!/usr/bin/env python3
"""
Детальный анализ перформанс логов PigWeight.

Анализирует новые JSON логи с метриками по каждому батчу и создает
подробный отчет о производительности системы.
"""

import json
import statistics
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))


class PerformanceAnalyzer:
    def __init__(self):
        self.batch_data = []
        self.performance_summaries = []
        self.time_series = []
        self.stream_stats = defaultdict(list)

    def load_perf_log(self, log_file: str):
        """Загружаем и парсим perf.log"""
        print(f"📊 Загружаем логи из {log_file}")

        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Парсим JSON логи
                    if line.startswith('{'):
                        data = json.loads(line)

                        # Определяем тип записи
                        if 'batch_size' in data and 'inference_time_ms' in data:
                            self.batch_data.append(data)
                            self.time_series.append(data)
                            self.stream_stats[data.get('stream_id', 'unknown')].append(data)
                        elif data.get('type') == 'performance_summary':
                            self.performance_summaries.append(data)

                except json.JSONDecodeError:
                    continue

        print(f"✅ Загружено {len(self.batch_data)} batch записей")
        print(f"✅ Загружено {len(self.performance_summaries)} summary записей")
        print(f"📈 Потоков: {len(self.stream_stats)}")

    def analyze_batch_performance(self):
        """Анализ производительности батчей"""
        if not self.batch_data:
            return {}

        # Собираем метрики
        batch_times = [d['batch_time_ms'] for d in self.batch_data]
        inference_times = [d['inference_time_ms'] for d in self.batch_data]
        preprocess_times = [d['preprocess_time_ms'] for d in self.batch_data]
        postprocess_times = [d['postprocess_time_ms'] for d in self.batch_data]
        fps_values = [d['fps'] for d in self.batch_data]
        batch_sizes = [d['batch_size'] for d in self.batch_data]
        detections = [d['detections'] for d in self.batch_data]

        return {
            'total_batches': len(self.batch_data),
            'batch_time': {
                'mean': statistics.mean(batch_times),
                'median': statistics.median(batch_times),
                'min': min(batch_times),
                'max': max(batch_times),
                'p95': statistics.quantiles(batch_times, n=20)[18],
                'p99': statistics.quantiles(batch_times, n=100)[98]
            },
            'inference_time': {
                'mean': statistics.mean(inference_times),
                'median': statistics.median(inference_times),
                'min': min(inference_times),
                'max': max(inference_times),
                'p95': statistics.quantiles(inference_times, n=20)[18],
                'p99': statistics.quantiles(inference_times, n=100)[98]
            },
            'preprocess_time': {
                'mean': statistics.mean(preprocess_times),
                'median': statistics.median(preprocess_times),
                'min': min(preprocess_times),
                'max': max(preprocess_times)
            },
            'postprocess_time': {
                'mean': statistics.mean(postprocess_times),
                'median': statistics.median(postprocess_times),
                'min': min(postprocess_times),
                'max': max(postprocess_times)
            },
            'fps': {
                'mean': statistics.mean(fps_values),
                'median': statistics.median(fps_values),
                'min': min(fps_values),
                'max': max(fps_values),
                'p95': statistics.quantiles(fps_values, n=20)[18]
            },
            'batch_sizes': {
                'unique': list(set(batch_sizes)),
                'most_common': max(set(batch_sizes), key=batch_sizes.count),
                'distribution': {size: batch_sizes.count(size) for size in set(batch_sizes)}
            },
            'detections': {
                'mean': statistics.mean(detections),
                'median': statistics.median(detections),
                'total': sum(detections),
                'distribution': {det: detections.count(det) for det in set(detections)}
            }
        }

    def analyze_time_patterns(self):
        """Анализ временных паттернов"""
        if not self.time_series:
            return {}

        # Группировка по времени
        hour_stats = defaultdict(list)
        minute_stats = defaultdict(list)

        for entry in self.time_series:
            dt = datetime.fromtimestamp(entry['timestamp'])
            hour_stats[dt.hour].append(entry)
            minute_stats[dt.minute].append(entry)

        # Анализ по часам
        hour_performance = {}
        for hour, entries in hour_stats.items():
            fps_values = [e['fps'] for e in entries]
            batch_times = [e['batch_time_ms'] for e in entries]

            hour_performance[hour] = {
                'batches': len(entries),
                'avg_fps': statistics.mean(fps_values),
                'avg_batch_time': statistics.mean(batch_times),
                'min_fps': min(fps_values),
                'max_fps': max(fps_values)
            }

        return {
            'hourly_performance': hour_performance,
            'best_hour': max(hour_performance.items(), key=lambda x: x[1]['avg_fps']) if hour_performance else None,
            'worst_hour': min(hour_performance.items(), key=lambda x: x[1]['avg_fps']) if hour_performance else None,
            'total_duration_hours': (max(e['timestamp'] for e in self.time_series) -
                                   min(e['timestamp'] for e in self.time_series)) / 3600
        }

    def analyze_stream_performance(self):
        """Анализ производительности по потокам"""
        stream_analysis = {}

        for stream_id, entries in self.stream_stats.items():
            if not entries:
                continue

            fps_values = [e['fps'] for e in entries]
            batch_times = [e['batch_time_ms'] for e in entries]
            detections = [e['detections'] for e in entries]

            stream_analysis[stream_id] = {
                'batches': len(entries),
                'avg_fps': statistics.mean(fps_values),
                'avg_batch_time': statistics.mean(batch_times),
                'total_detections': sum(detections),
                'detection_rate': sum(detections) / len(entries),
                'first_timestamp': min(e['timestamp'] for e in entries),
                'last_timestamp': max(e['timestamp'] for e in entries),
                'duration_seconds': max(e['timestamp'] for e in entries) - min(e['timestamp'] for e in entries)
            }

        return stream_analysis

    def identify_bottlenecks(self, analysis):
        """Идентификация узких мест"""
        bottlenecks = []

        batch_time = analysis.get('batch_time', {})
        inference_time = analysis.get('inference_time', {})
        preprocess_time = analysis.get('preprocess_time', {})
        postprocess_time = analysis.get('postprocess_time', {})

        # Определяем bottleneck по процентному соотношению
        if batch_time.get('mean', 0) > 1000:  # > 1 секунда на батч
            bottlenecks.append({
                'stage': 'batch_processing',
                'severity': 'critical',
                'avg_time': batch_time['mean'],
                'recommendation': 'Увеличьте batch_size или оптимизируйте preprocessing'
            })

        if inference_time.get('mean', 0) > 100:  # > 100ms на inference
            bottlenecks.append({
                'stage': 'inference',
                'severity': 'high',
                'avg_time': inference_time['mean'],
                'recommendation': 'Включите GPU acceleration или используйте ONNX Runtime'
            })

        if preprocess_time.get('mean', 0) > 200:  # > 200ms на preprocessing
            bottlenecks.append({
                'stage': 'preprocessing',
                'severity': 'medium',
                'avg_time': preprocess_time['mean'],
                'recommendation': 'Оптимизируйте изображение preprocessing'
            })

        fps_stats = analysis.get('fps', {})
        if fps_stats.get('mean', 0) < 5:  # < 5 FPS
            bottlenecks.append({
                'stage': 'overall_fps',
                'severity': 'critical',
                'avg_fps': fps_stats['mean'],
                'recommendation': 'Критически низкая производительность'
            })

        return bottlenecks

    def generate_report(self):
        """Генерация полного отчета"""
        analysis = self.analyze_batch_performance()
        time_patterns = self.analyze_time_patterns()
        stream_analysis = self.analyze_stream_performance()
        bottlenecks = self.identify_bottlenecks(analysis)

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_batches_analyzed': len(self.batch_data),
                'date_range': {
                    'start': datetime.fromtimestamp(min(e['timestamp'] for e in self.batch_data)).isoformat() if self.batch_data else None,
                    'end': datetime.fromtimestamp(max(e['timestamp'] for e in self.batch_data)).isoformat() if self.batch_data else None
                },
                'overall_performance': {
                    'avg_fps': analysis.get('fps', {}).get('mean', 0),
                    'avg_batch_time_ms': analysis.get('batch_time', {}).get('mean', 0),
                    'total_detections': analysis.get('detections', {}).get('total', 0)
                }
            },
            'detailed_analysis': analysis,
            'time_patterns': time_patterns,
            'stream_analysis': stream_analysis,
            'bottlenecks': bottlenecks,
            'recommendations': self.generate_recommendations(analysis, bottlenecks)
        }

        return report

    def generate_recommendations(self, analysis, bottlenecks):
        """Генерация рекомендаций по оптимизации"""
        recommendations = []

        # FPS рекомендации
        fps = analysis.get('fps', {}).get('mean', 0)
        if fps < 10:
            recommendations.append({
                'priority': 'critical',
                'category': 'performance',
                'message': '.1f',
                'actions': [
                    'Включите CUDA acceleration',
                    'Увеличьте batch_size до 8-16',
                    'Используйте ONNX Runtime для CPU'
                ]
            })

        # Batch size рекомендации
        batch_sizes = analysis.get('batch_sizes', {})
        current_batch = batch_sizes.get('most_common', 1)
        if current_batch == 1:
            recommendations.append({
                'priority': 'high',
                'category': 'optimization',
                'message': 'Используется batch_size=1, что неэффективно',
                'actions': [
                    'Установите BATCH_SIZE=8 в .env',
                    'Перезапустите inference worker'
                ]
            })

        # Bottleneck рекомендации
        for bottleneck in bottlenecks:
            recommendations.append({
                'priority': bottleneck['severity'],
                'category': 'bottleneck',
                'stage': bottleneck['stage'],
                'message': f"Обнаружен bottleneck: {bottleneck['stage']}",
                'actions': [bottleneck['recommendation']]
            })

        return recommendations

    def print_report(self, report):
        """Вывод отчета в консоль"""
        print("\n" + "="*80)
        print("🔬 DETAILED PERFORMANCE ANALYSIS REPORT")
        print("="*80)

        summary = report['summary']
        print("\n📊 SUMMARY:")
        print(f"   Total batches analyzed: {summary['total_batches_analyzed']}")

        if summary['date_range']['start'] and summary['date_range']['end']:
            # Calculate duration from timestamps
            start_ts = min(e['timestamp'] for e in self.batch_data)
            end_ts = max(e['timestamp'] for e in self.batch_data)
            duration = end_ts - start_ts
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
            print(f"   Analysis period: {duration_str}")
            print(f"   Date range: {summary['date_range']['start']} - {summary['date_range']['end']}")

        overall = summary['overall_performance']
        print("\n🏃 OVERALL PERFORMANCE:")
        print(".1f")
        print(".1f")
        print(f"   Total detections: {overall['total_detections']}")

        # Детальный анализ
        detailed = report['detailed_analysis']
        if detailed:
            batch_time = detailed['batch_time']
            inference_time = detailed['inference_time']
            fps_stats = detailed['fps']

            print("\n⚡ DETAILED METRICS:")
            print("   Batch Time (ms):")
            print(".1f")
            print(".1f")
            print(".1f")
            print("   Inference Time (ms):")
            print(".1f")
            print(".1f")
            print(".1f")
            print("   FPS:")
            print(".1f")
            print(".1f")
            print(".1f")
            # Batch sizes
            batch_sizes = detailed['batch_sizes']
            print("   Batch Sizes:")
            print(f"     Most common: {batch_sizes['most_common']}")
            print(f"     Distribution: {batch_sizes['distribution']}")

        # Bottlenecks
        bottlenecks = report['bottlenecks']
        if bottlenecks:
            print("\n🚨 BOTTLENECKS DETECTED:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                icon = severity_icon.get(bottleneck['severity'], '⚪')
                print(f"   {i}. {icon} {bottleneck['stage']} - {bottleneck['severity'].upper()}")
                print(f"      Time: {bottleneck.get('avg_time', 'N/A')}")
                print(f"      Recommendation: {bottleneck['recommendation']}")

        # Рекомендации
        recommendations = report['recommendations']
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_recs = sorted(recommendations, key=lambda x: priority_order.get(x['priority'], 99))

            for i, rec in enumerate(sorted_recs, 1):
                priority_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                icon = priority_icon.get(rec['priority'], '⚪')
                print(f"   {i}. {icon} [{rec['priority'].upper()}] {rec['message']}")
                for action in rec.get('actions', []):
                    print(f"      • {action}")

        print("\n" + "="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deep performance analysis of PigWeight logs")
    parser.add_argument("--log_file", default="logs/perf.log", help="Path to performance log file")
    parser.add_argument("--output", default="deep_perf_analysis.json", help="Output file for detailed analysis")

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"❌ Log file not found: {args.log_file}")
        return 1

    analyzer = PerformanceAnalyzer()
    analyzer.load_perf_log(args.log_file)

    if not analyzer.batch_data:
        print("❌ No batch performance data found in log file")
        print("💡 Make sure the system has been running with the new logging enabled")
        return 1

    report = analyzer.generate_report()
    analyzer.print_report(report)

    # Сохраняем детальный отчет
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Detailed report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
