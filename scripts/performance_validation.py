"""
Скрипт для валидации производительности оптимизированной системы
Проверяет достижение целевых метрик: FPS 60+, латентность 50-100ms
"""

import asyncio
import time
import statistics
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import concurrent.futures
import threading
from pathlib import Path

import numpy as np
import requests
import websockets

# Простая настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceTarget:
    """Целевые показатели производительности"""
    min_fps: float = 60.0
    max_latency_ms: float = 100.0
    min_latency_ms: float = 50.0
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 85.0
    max_gpu_usage: float = 95.0
    min_concurrent_streams: int = 4
    test_duration_seconds: int = 120

@dataclass
class TestResults:
    """Результаты тестирования"""
    timestamp: float
    test_name: str
    passed: bool
    actual_fps: float
    actual_latency_ms: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    concurrent_streams: int
    details: Dict[str, Any]

class PerformanceValidator:
    """Валидатор производительности оптимизированной системы"""
    
    def __init__(self, base_url: str = "http://localhost:8000", targets: Optional[PerformanceTarget] = None):
        self.base_url = base_url
        self.targets = targets or PerformanceTarget()
        
        # Результаты тестирования
        self.test_results: List[TestResults] = []
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Состояние тестирования
        self.testing_active = False
        self.current_test = None
        
        logger.info(f"Инициализирован валидатор для {base_url}")
        logger.info(f"Целевые показатели: FPS≥{targets.min_fps}, Latency≤{targets.max_latency_ms}ms")
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Запуск полной валидации производительности"""
        logger.info("🚀 Начало полной валидации производительности")
        
        validation_start = time.time()
        
        try:
            # 1. Проверка готовности системы
            await self._check_system_readiness()
            
            # 2. Базовый тест производительности
            base_results = await self._run_baseline_test()
            
            # 3. Нагрузочное тестирование
            load_results = await self._run_load_test()
            
            # 4. Тест стабильности
            stability_results = await self._run_stability_test()
            
            # 5. Тест адаптивности
            adaptive_results = await self._run_adaptive_test()
            
            # 6. Генерация финального отчета
            final_report = self._generate_final_report([
                base_results, load_results, stability_results, adaptive_results
            ])
            
            validation_duration = time.time() - validation_start
            final_report['total_duration_seconds'] = validation_duration
            
            # Сохранение результатов
            await self._save_results(final_report)
            
            logger.info(f"✅ Валидация завершена за {validation_duration:.1f} секунд")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации: {e}")
            raise
            
    async def _check_system_readiness(self):
        """Проверка готовности системы"""
        logger.info("🔍 Проверка готовности системы...")
        
        try:
            # Проверка API доступности
            response = requests.get(f"{self.base_url}/api/v2/status", timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"API недоступен: {response.status_code}")
                
            # Проверка оптимизированных компонентов
            system_info = requests.get(f"{self.base_url}/api/v2/system/info", timeout=10)
            if not system_info.json().get('components_ready', False):
                raise RuntimeError("Система не подтвердила готовность компонентов")
                
            logger.info("✅ Система готова к тестированию")
            
        except Exception as e:
            logger.error(f"❌ Система не готова: {e}")
            raise
            
    async def _run_baseline_test(self) -> TestResults:
        """Базовое тестирование производительности"""
        logger.info("📊 Запуск базового теста производительности...")
        
        self.current_test = "baseline"
        
        # Сбор метрик в течение 30 секунд
        metrics = await self._collect_metrics_over_time(30)
        
        if not metrics:
            raise RuntimeError("Не удалось собрать метрики")
            
        # Анализ результатов
        avg_fps = statistics.mean([m.get('fps', 0) for m in metrics])
        avg_latency = statistics.mean([m.get('latency_ms', 0) for m in metrics if m.get('latency_ms', 0) > 0])
        avg_cpu = statistics.mean([m.get('cpu_usage', 0) for m in metrics])
        avg_memory = statistics.mean([m.get('memory_usage', 0) for m in metrics])
        avg_gpu = statistics.mean([m.get('gpu_usage', 0) for m in metrics if m.get('gpu_usage')])
        
        # Определение успешности теста
        passed = (
            avg_fps >= self.targets.min_fps and
            avg_latency <= self.targets.max_latency_ms and
            avg_cpu <= self.targets.max_cpu_usage and
            avg_memory <= self.targets.max_memory_usage
        )
        
        results = TestResults(
            timestamp=time.time(),
            test_name="baseline",
            passed=passed,
            actual_fps=avg_fps,
            actual_latency_ms=avg_latency,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu,
            concurrent_streams=1,
            details={
                'metrics_count': len(metrics),
                'fps_range': [min([m.get('fps', 0) for m in metrics]), max([m.get('fps', 0) for m in metrics])],
                'latency_p95': np.percentile([m.get('latency_ms', 0) for m in metrics if m.get('latency_ms', 0) > 0], 95) if metrics else 0
            }
        )
        
        self.test_results.append(results)
        logger.info(f"📊 Базовый тест: {'✅ ПРОЙДЕН' if passed else '❌ ПРОВАЛЕН'} - FPS: {avg_fps:.1f}, Latency: {avg_latency:.1f}ms")
        
        return results
        
    async def _run_load_test(self) -> TestResults:
        """Нагрузочное тестирование"""
        logger.info("🔥 Запуск нагрузочного тестирования...")
        
        self.current_test = "load"
        
        # Создание нагрузки через множественные запросы
        async def generate_load():
            tasks = []
            for i in range(10):  # 10 параллельных потоков
                task = asyncio.create_task(self._simulate_stream_requests())
                tasks.append(task)
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # Запуск нагрузки в фоне и сбор метрик
        load_task = asyncio.create_task(generate_load())
        
        try:
            metrics = await self._collect_metrics_over_time(60)  # 60 секунд под нагрузкой
        finally:
            load_task.cancel()
            try:
                await load_task
            except asyncio.CancelledError:
                pass
                
        # Анализ под нагрузкой
        if not metrics:
            raise RuntimeError("Не удалось собрать метрики под нагрузкой")
            
        avg_fps = statistics.mean([m.get('fps', 0) for m in metrics])
        p95_latency = np.percentile([m.get('latency_ms', 0) for m in metrics if m.get('latency_ms', 0) > 0], 95)
        max_cpu = max([m.get('cpu_usage', 0) for m in metrics])
        max_memory = max([m.get('memory_usage', 0) for m in metrics])
        
        # Под нагрузкой допустимы более мягкие требования
        load_targets = PerformanceTarget(
            min_fps=self.targets.min_fps * 0.8,  # 80% от целевого FPS
            max_latency_ms=self.targets.max_latency_ms * 1.5,  # 150% от целевой латентности
            max_cpu_usage=95.0,  # Повышенное использование CPU приемлемо
            max_memory_usage=90.0  # Повышенное использование памяти приемлемо
        )
        
        passed = (
            avg_fps >= load_targets.min_fps and
            p95_latency <= load_targets.max_latency_ms and
            max_cpu <= load_targets.max_cpu_usage and
            max_memory <= load_targets.max_memory_usage
        )
        
        results = TestResults(
            timestamp=time.time(),
            test_name="load",
            passed=passed,
            actual_fps=avg_fps,
            actual_latency_ms=p95_latency,
            cpu_usage=max_cpu,
            memory_usage=max_memory,
            gpu_usage=max([m.get('gpu_usage', 0) for m in metrics if m.get('gpu_usage')]),
            concurrent_streams=10,
            details={
                'load_duration': 60,
                'parallel_streams': 10,
                'latency_p99': np.percentile([m.get('latency_ms', 0) for m in metrics if m.get('latency_ms', 0) > 0], 99)
            }
        )
        
        self.test_results.append(results)
        logger.info(f"🔥 Нагрузочный тест: {'✅ ПРОЙДЕН' if passed else '❌ ПРОВАЛЕН'} - FPS: {avg_fps:.1f}, P95 Latency: {p95_latency:.1f}ms")
        
        return results
        
    async def _run_stability_test(self) -> TestResults:
        """Тест стабильности системы"""
        logger.info("⏱️ Запуск теста стабильности...")
        
        self.current_test = "stability"
        
        # Длительный сбор метрик (2 минуты)
        metrics = await self._collect_metrics_over_time(self.targets.test_duration_seconds)
        
        if not metrics:
            raise RuntimeError("Не удалось собрать метрики стабильности")
            
        # Анализ стабильности
        fps_values = [m.get('fps', 0) for m in metrics]
        latency_values = [m.get('latency_ms', 0) for m in metrics if m.get('latency_ms', 0) > 0]
        
        fps_std = statistics.stdev(fps_values) if len(fps_values) > 1 else 0
        latency_std = statistics.stdev(latency_values) if len(latency_values) > 1 else 0
        
        avg_fps = statistics.mean(fps_values)
        avg_latency = statistics.mean(latency_values) if latency_values else 0
        
        # Критерии стабильности: низкая вариабельность
        fps_stability = fps_std / avg_fps if avg_fps > 0 else 1.0  # Коэффициент вариации
        latency_stability = latency_std / avg_latency if avg_latency > 0 else 1.0
        
        passed = (
            avg_fps >= self.targets.min_fps and
            avg_latency <= self.targets.max_latency_ms and
            fps_stability <= 0.1 and  # Вариация FPS не более 10%
            latency_stability <= 0.2   # Вариация латентности не более 20%
        )
        
        results = TestResults(
            timestamp=time.time(),
            test_name="stability", 
            passed=passed,
            actual_fps=avg_fps,
            actual_latency_ms=avg_latency,
            cpu_usage=statistics.mean([m.get('cpu_usage', 0) for m in metrics]),
            memory_usage=statistics.mean([m.get('memory_usage', 0) for m in metrics]),
            gpu_usage=statistics.mean([m.get('gpu_usage', 0) for m in metrics if m.get('gpu_usage')]),
            concurrent_streams=1,
            details={
                'test_duration': self.targets.test_duration_seconds,
                'fps_std': fps_std,
                'latency_std': latency_std,
                'fps_stability_coefficient': fps_stability,
                'latency_stability_coefficient': latency_stability,
                'metrics_count': len(metrics)
            }
        )
        
        self.test_results.append(results)
        logger.info(f"⏱️ Тест стабильности: {'✅ ПРОЙДЕН' if passed else '❌ ПРОВАЛЕН'} - Стабильность FPS: {fps_stability:.3f}")
        
        return results
        
    async def _run_adaptive_test(self) -> TestResults:
        """Тест адаптивности качества"""
        logger.info("🎛️ Запуск теста адаптивности...")
        
        self.current_test = "adaptive"
        
        # Тест переключения уровней качества
        quality_levels = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "ULTRA"]
        adaptation_results = []
        
        for level in quality_levels:
            logger.info(f"   Тестирование уровня {level}...")
            
            # Установка уровня качества
            try:
                response = requests.post(
                    f"{self.base_url}/api/v2/quality/set",
                    json={"level": level, "force": True},
                    timeout=10
                )
                if response.status_code != 200:
                    logger.warning(f"Не удалось установить уровень {level}")
                    continue
            except Exception as e:
                logger.warning(f"Ошибка установки уровня {level}: {e}")
                continue
                
            # Ждем адаптации
            await asyncio.sleep(5)
            
            # Сбор метрик на этом уровне
            level_metrics = await self._collect_metrics_over_time(20)
            
            if level_metrics:
                avg_fps = statistics.mean([m.get('fps', 0) for m in level_metrics])
                avg_latency = statistics.mean([m.get('latency_ms', 0) for m in level_metrics if m.get('latency_ms', 0) > 0])
                
                adaptation_results.append({
                    'level': level,
                    'fps': avg_fps,
                    'latency': avg_latency,
                    'metrics_count': len(level_metrics)
                })
                
        # Анализ адаптивности
        if len(adaptation_results) < 3:
            passed = False
            avg_fps = 0
            avg_latency = 999
        else:
            # Проверяем что система адаптируется (FPS/качество меняется между уровнями)
            fps_range = max([r['fps'] for r in adaptation_results]) - min([r['fps'] for r in adaptation_results])
            adaptation_working = fps_range > 5.0  # Разница в FPS между уровнями > 5
            
            avg_fps = statistics.mean([r['fps'] for r in adaptation_results])
            avg_latency = statistics.mean([r['latency'] for r in adaptation_results if r['latency'] > 0])
            
            passed = adaptation_working and avg_fps >= self.targets.min_fps * 0.7
            
        results = TestResults(
            timestamp=time.time(),
            test_name="adaptive",
            passed=passed,
            actual_fps=avg_fps,
            actual_latency_ms=avg_latency,
            cpu_usage=0,  # Не измеряем в этом тесте
            memory_usage=0,
            gpu_usage=None,
            concurrent_streams=1,
            details={
                'tested_levels': len(adaptation_results),
                'adaptation_results': adaptation_results,
                'fps_range': max([r['fps'] for r in adaptation_results]) - min([r['fps'] for r in adaptation_results]) if adaptation_results else 0
            }
        )
        
        self.test_results.append(results)
        logger.info(f"🎛️ Тест адаптивности: {'✅ ПРОЙДЕН' if passed else '❌ ПРОВАЛЕН'} - Уровней: {len(adaptation_results)}")
        
        return results
        
    async def _collect_metrics_over_time(self, duration_seconds: int) -> List[Dict[str, Any]]:
        """Сбор метрик за указанный период"""
        metrics = []
        start_time = time.time()
        
        logger.info(f"   Сбор метрик в течение {duration_seconds} секунд...")
        
        while (time.time() - start_time) < duration_seconds:
            try:
                # Получение текущих метрик
                response = requests.get(f"{self.base_url}/api/v2/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    metric = {
                        'timestamp': time.time(),
                        'fps': data.get('current_fps', 0),
                        'latency_ms': 0,  # Будет заполнено из performance endpoint
                        'cpu_usage': data.get('cpu_usage', 0),
                        'memory_usage': data.get('memory_usage', 0),
                        'gpu_usage': data.get('gpu_usage'),
                        'active_streams': data.get('active_streams', 0)
                    }
                    
                    # Дополнительные метрики производительности
                    try:
                        perf_response = requests.get(f"{self.base_url}/api/v2/performance", timeout=5)
                        if perf_response.status_code == 200:
                            perf_data = perf_response.json()
                            metric['latency_ms'] = perf_data.get('avg_latency_ms', 0)
                    except:
                        pass
                        
                    metrics.append(metric)
                    
            except Exception as e:
                logger.debug(f"Ошибка сбора метрики: {e}")
                
            await asyncio.sleep(1.0)  # Сбор каждую секунду
            
        logger.info(f"   Собрано {len(metrics)} метрик")
        return metrics
        
    async def _simulate_stream_requests(self):
        """Симуляция запросов потока для создания нагрузки"""
        for i in range(30):  # 30 запросов за минуту
            try:
                requests.get(f"{self.base_url}/api/v2/status", timeout=2)
                await asyncio.sleep(2.0)
            except:
                pass  # Игнорируем ошибки в нагрузочном тесте
                
    def _generate_final_report(self, test_results: List[TestResults]) -> Dict[str, Any]:
        """Генерация финального отчета"""
        passed_tests = sum(1 for result in test_results if result.passed)
        total_tests = len(test_results)
        
        overall_passed = passed_tests == total_tests
        
        # Агрегированные метрики
        all_fps = [r.actual_fps for r in test_results if r.actual_fps > 0]
        all_latencies = [r.actual_latency_ms for r in test_results if r.actual_latency_ms > 0]
        
        report = {
            'validation_timestamp': time.time(),
            'overall_result': 'PASSED' if overall_passed else 'FAILED',
            'tests_passed': passed_tests,
            'tests_total': total_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            
            'performance_summary': {
                'avg_fps': statistics.mean(all_fps) if all_fps else 0,
                'min_fps': min(all_fps) if all_fps else 0,
                'max_fps': max(all_fps) if all_fps else 0,
                'avg_latency_ms': statistics.mean(all_latencies) if all_latencies else 0,
                'p95_latency_ms': np.percentile(all_latencies, 95) if all_latencies else 0,
                'fps_target_met': min(all_fps) >= self.targets.min_fps if all_fps else False,
                'latency_target_met': max(all_latencies) <= self.targets.max_latency_ms if all_latencies else False
            },
            
            'targets': asdict(self.targets),
            'detailed_results': [asdict(result) for result in test_results],
            
            'recommendations': self._generate_recommendations(test_results)
        }
        
        return report
        
    def _generate_recommendations(self, test_results: List[TestResults]) -> List[str]:
        """Генерация рекомендаций по результатам тестирования"""
        recommendations = []
        
        # Анализ FPS
        fps_values = [r.actual_fps for r in test_results if r.actual_fps > 0]
        if fps_values and min(fps_values) < self.targets.min_fps:
            recommendations.append(f"FPS ниже целевого ({min(fps_values):.1f} < {self.targets.min_fps}). Рекомендуется увеличить batch_size или снизить качество.")
            
        # Анализ латентности
        latency_values = [r.actual_latency_ms for r in test_results if r.actual_latency_ms > 0]
        if latency_values and max(latency_values) > self.targets.max_latency_ms:
            recommendations.append(f"Латентность превышает цель ({max(latency_values):.1f}ms > {self.targets.max_latency_ms}ms). Рекомендуется уменьшить batch_size или включить H.264 Direct.")
            
        # Анализ стабильности
        stability_test = next((r for r in test_results if r.test_name == 'stability'), None)
        if stability_test and not stability_test.passed:
            recommendations.append("Обнаружены проблемы стабильности. Проверьте память и GC настройки.")
            
        # Анализ адаптивности
        adaptive_test = next((r for r in test_results if r.test_name == 'adaptive'), None)
        if adaptive_test and not adaptive_test.passed:
            recommendations.append("Система адаптивного качества работает неправильно. Проверьте AdaptiveQualityController.")
            
        if not recommendations:
            recommendations.append("Все тесты пройдены успешно! Система работает в пределах целевых показателей.")
            
        return recommendations
        
    async def _save_results(self, report: Dict[str, Any]):
        """Сохранение результатов в файл"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"performance_validation_{timestamp}.json"
        
        results_dir = Path("logs/validation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / filename
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📄 Результаты сохранены в {results_file}")

async def main():
    """Главная функция для запуска валидации"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Валидация производительности PigWeight")
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL сервера')
    parser.add_argument('--min-fps', type=float, default=60.0, help='Минимальный целевой FPS')
    parser.add_argument('--max-latency', type=float, default=100.0, help='Максимальная целевая латентность (мс)')
    parser.add_argument('--duration', type=int, default=120, help='Длительность теста стабильности (сек)')
    
    args = parser.parse_args()
    
    # Создание целевых показателей
    targets = PerformanceTarget(
        min_fps=args.min_fps,
        max_latency_ms=args.max_latency,
        test_duration_seconds=args.duration
    )
    
    # Запуск валидации
    validator = PerformanceValidator(args.url, targets)
    
    try:
        results = await validator.run_full_validation()
        
        # Вывод результатов
        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*60)
        print(f"Общий результат: {'✅ УСПЕХ' if results['overall_result'] == 'PASSED' else '❌ НЕУДАЧА'}")
        print(f"Пройдено тестов: {results['tests_passed']}/{results['tests_total']} ({results['success_rate']:.1f}%)")
        print(f"Средний FPS: {results['performance_summary']['avg_fps']:.1f}")
        print(f"Средняя латентность: {results['performance_summary']['avg_latency_ms']:.1f} мс")
        print(f"P95 латентность: {results['performance_summary']['p95_latency_ms']:.1f} мс")
        
        print("\n📋 Рекомендации:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
            
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Ошибка валидации: {e}")
        return 1
        
    return 0 if results['overall_result'] == 'PASSED' else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

