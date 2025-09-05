#!/usr/bin/env python3
"""
Анализатор полного пайплайна PigWeight на основе performance логов.

Разбирает весь поток обработки от захвата кадра до отображения результатов:
1. Frame Capture (захват кадра)
2. Preprocessing (предобработка)
3. Inference (инференс модели)
4. Postprocessing (постобработка)
5. Transmission (передача)
6. Rendering (отрисовка)

Использование:
python scripts/analyze_pipeline.py --log_file logs/perf.log
"""

import json
import re
import statistics
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PipelineAnalyzer:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.pipeline_stages = {
            'frame_capture': [],
            'preprocessing': [],
            'inference': [],
            'postprocessing': [],
            'transmission': [],
            'rendering': [],
            'model_loading': []
        }
        self.errors = []
        self.warnings = []

    def parse_ultra_file_ws_log(self, line: str) -> Dict[str, Any]:
        """Parse ultra_file_ws format: fps=X read=Xms proc=Xms enc=Xms send=Xms"""
        pattern = r'ultra_file_ws id=(\w+) fps=([\d.]+) read=([\d.]+)ms proc=([\d.]+)ms enc=([\d.]+)ms send=([\d.]+)ms'
        match = re.search(pattern, line)

        if match:
            stream_id, fps, read_ms, proc_ms, enc_ms, send_ms = match.groups()
            return {
                'timestamp': self.extract_timestamp(line),
                'stream_id': stream_id,
                'fps': float(fps),
                'read_ms': float(read_ms),
                'proc_ms': float(proc_ms),
                'enc_ms': float(enc_ms),
                'send_ms': float(send_ms),
                'total_latency_ms': float(read_ms) + float(proc_ms) + float(enc_ms) + float(send_ms)
            }
        return None

    def parse_json_log(self, line: str) -> Dict[str, Any]:
        """Parse JSON format logs"""
        try:
            # Find JSON part in line
            json_start = line.find('{')
            if json_start != -1:
                json_str = line[json_start:]
                data = json.loads(json_str)

                # Use existing timestamp if present, otherwise extract from line
                if 'timestamp' not in data:
                    data['timestamp'] = self.extract_timestamp(line)

                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def extract_timestamp(self, line: str) -> str:
        """Extract timestamp from log line"""
        # Format: 2025-08-14 21:57:33,190
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
        return timestamp_match.group(1) if timestamp_match else "unknown"

    def analyze_log_file(self, log_file: str):
        """Analyze entire log file"""
        print(f"Analyzing log file: {log_file}")

        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse ultra_file_ws format
                ultra_data = self.parse_ultra_file_ws_log(line)
                if ultra_data:
                    self._categorize_ultra_metrics(ultra_data)
                    continue

                # Parse JSON format
                json_data = self.parse_json_log(line)
                if json_data:
                    self._categorize_json_metrics(json_data)
                    continue

                # Check for errors/warnings
                self._check_for_errors_warnings(line)

        print(f"Analysis complete. Processed {line_num} lines.")

    def _categorize_ultra_metrics(self, data: Dict[str, Any]):
        """Categorize ultra_file_ws metrics into pipeline stages"""
        # Frame capture (read)
        if 'read_ms' in data:
            self.pipeline_stages['frame_capture'].append(data['read_ms'])

        # Processing (inference + preprocessing)
        if 'proc_ms' in data:
            self.pipeline_stages['inference'].append(data['proc_ms'])

        # Encoding (postprocessing)
        if 'enc_ms' in data:
            self.pipeline_stages['postprocessing'].append(data['enc_ms'])

        # Transmission (send)
        if 'send_ms' in data:
            self.pipeline_stages['transmission'].append(data['send_ms'])

        # Store raw data
        self.metrics['ultra_file_ws'].append(data)

    def _categorize_json_metrics(self, data: Dict[str, Any]):
        """Categorize JSON metrics"""
        entry_type = data.get('type', '')

        if entry_type == 'performance_summary':
            # Handle performance summary entries
            self.metrics['performance_summaries'].append(data)
            return

        # Handle batch performance entries (new format)
        if 'batch_size' in data and 'inference_time_ms' in data:
            batch_time = data.get('batch_time_ms', 0) / 1000  # Convert to seconds
            inference_time = data.get('inference_time_ms', 0) / 1000
            preprocess_time = data.get('preprocess_time_ms', 0) / 1000
            postprocess_time = data.get('postprocess_time_ms', 0) / 1000

            # Categorize into pipeline stages
            if batch_time > 0:
                self.pipeline_stages['frame_capture'].append(batch_time * 0.1)  # Estimate
            if preprocess_time > 0:
                self.pipeline_stages['preprocessing'].append(preprocess_time)
            if inference_time > 0:
                self.pipeline_stages['inference'].append(inference_time)
            if postprocess_time > 0:
                self.pipeline_stages['postprocessing'].append(postprocess_time)

            self.metrics['batch_performance'].append(data)
            return

        # Handle legacy JSON format
        phase = data.get('phase', '')
        if phase == 'model_load':
            self.pipeline_stages['model_loading'].append(data.get('load_ms', 0))
            self.metrics['model_loading'].append(data)
        elif phase == 'inference':
            self.pipeline_stages['inference'].append(data.get('inference_ms', 0))
            self.metrics['inference'].append(data)
        elif phase == 'preprocessing':
            self.pipeline_stages['preprocessing'].append(data.get('preprocess_ms', 0))
            self.metrics['preprocessing'].append(data)

    def _check_for_errors_warnings(self, line: str):
        """Check for errors and warnings in log lines"""
        if 'ERROR' in line or 'error' in line.lower():
            self.errors.append(line)
        elif 'WARNING' in line or 'warning' in line.lower():
            self.warnings.append(line)

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        stats = {}

        # Pipeline stage statistics
        for stage, values in self.pipeline_stages.items():
            if values:
                stats[stage] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                    'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0
                }

        # Overall pipeline statistics
        if self.metrics['ultra_file_ws']:
            total_latencies = [d['total_latency_ms'] for d in self.metrics['ultra_file_ws']]
            fps_values = [d['fps'] for d in self.metrics['ultra_file_ws']]

            stats['overall'] = {
                'total_measurements': len(self.metrics['ultra_file_ws']),
                'mean_total_latency': statistics.mean(total_latencies),
                'mean_fps': statistics.mean(fps_values),
                'bottleneck_stage': self._identify_bottleneck(),
                'efficiency_score': self._calculate_efficiency_score()
            }

        # Error analysis
        stats['errors'] = {
            'count': len(self.errors),
            'warnings_count': len(self.warnings)
        }

        return stats

    def _identify_bottleneck(self) -> str:
        """Identify the bottleneck stage"""
        max_mean_time = 0
        bottleneck = "unknown"

        for stage, values in self.pipeline_stages.items():
            if values and statistics.mean(values) > max_mean_time:
                max_mean_time = statistics.mean(values)
                bottleneck = stage

        return bottleneck

    def _calculate_efficiency_score(self) -> float:
        """Calculate overall pipeline efficiency (0-100)"""
        if not self.metrics['ultra_file_ws']:
            return 0.0

        # Ideal target: 30 FPS with <50ms latency
        target_fps = 30.0
        target_latency = 50.0

        avg_fps = statistics.mean([d['fps'] for d in self.metrics['ultra_file_ws']])
        avg_latency = statistics.mean([d['total_latency_ms'] for d in self.metrics['ultra_file_ws']])

        fps_score = min(100, (avg_fps / target_fps) * 100)
        latency_score = min(100, (target_latency / avg_latency) * 100)

        return (fps_score + latency_score) / 2

    def print_analysis_report(self):
        """Print comprehensive analysis report"""
        stats = self.calculate_statistics()

        print("\n" + "="*80)
        print("🐷 PIGWEIGHT PIPELINE ANALYSIS REPORT")
        print("="*80)

        # Overall performance
        if 'overall' in stats:
            overall = stats['overall']
            print("\n📊 OVERALL PERFORMANCE:")
            print(".2f")
            print(".2f")
            print(f"🎯 Bottleneck Stage: {overall['bottleneck_stage']}")
            print(".1f")

        # Show recent entries with human-readable dates
        print("\n📅 RECENT PERFORMANCE ENTRIES:")
        if self.metrics.get('batch_performance'):
            recent_entries = self.metrics['batch_performance'][-5:]  # Last 5 entries
            for entry in recent_entries:
                dt = entry.get('datetime', 'unknown')
                stream = entry.get('stream_id', 'unknown')
                fps = entry.get('fps', 0)
                batch_size = entry.get('batch_size', 1)
                detections = entry.get('detections', 0)
                inference_ms = entry.get('inference_time_ms', 0)
                print(f"  {dt} | {stream} | Batch:{batch_size} | FPS:{fps:.1f} | Detections:{detections} | Inference:{inference_ms:.1f}ms")

        if self.metrics.get('performance_summaries'):
            print("\n📈 PERFORMANCE SUMMARIES:")
            for summary in self.metrics['performance_summaries'][-3:]:  # Last 3 summaries
                dt = summary.get('datetime', 'unknown')
                stream = summary.get('stream_id', 'unknown')
                throughput = summary.get('throughput_fps', 0)
                batches = summary.get('total_batches', 0)
                print(f"  {dt} | {stream} | Throughput:{throughput:.1f} FPS | Batches:{batches}")

        # Pipeline breakdown
        print("\n🔧 PIPELINE BREAKDOWN:")
        print("-" * 60)

        stage_names = {
            'model_loading': '🤖 Model Loading',
            'frame_capture': '📹 Frame Capture',
            'preprocessing': '🎨 Preprocessing',
            'inference': '🧠 Inference',
            'postprocessing': '✂️  Postprocessing',
            'transmission': '📡 Transmission',
            'rendering': '🖥️  Rendering'
        }

        for stage, stage_stats in stats.items():
            if stage in ['overall', 'errors']:
                continue

            name = stage_names.get(stage, stage.title())
            if isinstance(stage_stats, dict) and 'count' in stage_stats:
                count = stage_stats['count']
                mean_time = stage_stats['mean']
                p95 = stage_stats.get('p95', stage_stats['max'])

                print("8d")
                print("6.1f")
                print("6.1f")

        # Error summary
        if 'errors' in stats:
            err_stats = stats['errors']
            print("\n⚠️  ISSUES SUMMARY:")
            print(f"🚨 Errors: {err_stats['count']}")
            print(f"⚡ Warnings: {err_stats['warnings_count']}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        self._print_recommendations(stats)

        print("\n" + "="*80)

    def _print_recommendations(self, stats: Dict[str, Any]):
        """Print optimization recommendations"""
        if 'overall' not in stats:
            return

        overall = stats['overall']
        bottleneck = overall['bottleneck_stage']

        if bottleneck == 'inference':
            print("🔥 INFERENCE BOTTLENECK:")
            print("   • Enable CUDA acceleration if available")
            print("   • Reduce model size (use YOLOv11n instead of v11)")
            print("   • Implement model quantization (FP16)")
            print("   • Increase batch size for GPU utilization")

        elif bottleneck == 'transmission':
            print("📡 TRANSMISSION BOTTLENECK:")
            print("   • Switch to WebRTC for lower latency")
            print("   • Reduce JPEG quality for faster encoding")
            print("   • Implement frame skipping for high-motion scenes")
            print("   • Use H.264 encoding instead of JPEG")

        elif bottleneck == 'frame_capture':
            print("📹 CAPTURE BOTTLENECK:")
            print("   • Check camera frame rate settings")
            print("   • Reduce capture resolution if possible")
            print("   • Use hardware-accelerated decoding")
            print("   • Optimize RTSP stream settings")

        if overall['mean_fps'] < 10:
            print("🐌 LOW FPS DETECTED:")
            print("   • Consider MJPEG fallback for debugging")
            print("   • Check GPU/CPU utilization")
            print("   • Profile individual pipeline stages")

        if overall.get('efficiency_score', 0) < 50:
            print("⚡ LOW EFFICIENCY:")
            print("   • Enable all optimizations (.env settings)")
            print("   • Update PyTorch to latest version")
            print("   • Consider hardware upgrade")

    def save_detailed_report(self, output_file: str):
        """Save detailed analysis to JSON file"""
        stats = self.calculate_statistics()

        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'pipeline_stages': stats,
            'raw_metrics': dict(self.metrics),
            'errors': self.errors[:100],  # Limit to first 100 errors
            'warnings': self.warnings[:100]  # Limit to first 100 warnings
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Detailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PigWeight pipeline performance")
    parser.add_argument("--log_file", default="logs/perf.log",
                       help="Path to performance log file")
    parser.add_argument("--output", default="pipeline_analysis.json",
                       help="Output file for detailed analysis")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"❌ Log file not found: {args.log_file}")
        return 1

    analyzer = PipelineAnalyzer()
    analyzer.analyze_log_file(args.log_file)
    analyzer.print_analysis_report()
    analyzer.save_detailed_report(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
