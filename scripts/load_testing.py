"""
Скрипт для нагрузочного тестирования оптимизированной системы PigWeight
Проверяет производительность под различными типами нагрузки
"""

import asyncio
import time
import random
import statistics
import json
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

import requests
import websockets
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Конфигурация нагрузочного тестирования"""
    base_url: str = "http://localhost:8000"
    
    # Параметры нагрузки
    concurrent_users: int = 16
    requests_per_user: int = 100
    ramp_up_duration: int = 60  # Время наращивания нагрузки
    test_duration: int = 300   # Общее время теста
    
    # Типы запросов и их веса
    request_weights: Dict[str, float] = None
    
    # Целевые показатели
    target_rps: float = 50.0  # Запросов в секунду
    max_response_time: float = 5000.0  # мс
    max_error_rate: float = 5.0  # %
    
    def __post_init__(self):
        if self.request_weights is None:
            self.request_weights = {
                'status': 0.4,
                'performance': 0.2,
                'quality': 0.1,
                'system_info': 0.1,
                'queue_stats': 0.1,
                'batcher_stats': 0.1
            }

@dataclass
class RequestResult:
    """Результат отдельного запроса"""
    timestamp: float
    endpoint: str
    response_time_ms: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    payload_size: int = 0

@dataclass
class LoadTestResults:
    """Результаты нагрузочного тестирования"""
    test_name: str
    config: LoadTestConfig
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Статистика времени ответа
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    
    # Throughput
    requests_per_second: float
    bytes_per_second: float
    
    # Ошибки
    error_rate: float
    error_distribution: Dict[str, int]
    
    # Результат
    passed: bool
    details: Dict[str, Any]

class LoadTestClient:
    """Клиент для генерации нагрузки"""
    
    def __init__(self, client_id: int, config: LoadTestConfig):
        self.client_id = client_id
        self.config = config
        self.results: List[RequestResult] = []
        
        # Подготовка эндпоинтов
        self.endpoints = {
            'status': '/api/v2/status',
            'performance': '/api/v2/performance',
            'quality': '/api/v2/quality/current',
            'system_info': '/api/v2/system/info',
            'queue_stats': '/api/v2/queue/stats',
            'batcher_stats': '/api/v2/batcher/stats'
        }
        
    async def run_load_test(self) -> List[RequestResult]:
        """Запуск нагрузочного тестирования от клиента"""
        logger.info(f"🔥 Клиент {self.client_id}: начало нагрузочного теста")
        
        start_time = time.time()
        
        for request_num in range(self.config.requests_per_user):
            if (time.time() - start_time) > self.config.test_duration:
                break
                
            # Выбор эндпоинта по весам
            endpoint_name = self._select_endpoint()
            endpoint_path = self.endpoints[endpoint_name]
            
            # Выполнение запроса
            result = await self._make_request(endpoint_name, endpoint_path)
            self.results.append(result)
            
            # Случайная задержка между запросами
            await asyncio.sleep(random.uniform(0.1, 2.0))
            
        logger.info(f"🏁 Клиент {self.client_id}: завершил {len(self.results)} запросов")
        return self.results
        
    def _select_endpoint(self) -> str:
        """Выбор эндпоинта по весовым коэффициентам"""
        weights = list(self.config.request_weights.values())
        endpoints = list(self.config.request_weights.keys())
        return random.choices(endpoints, weights=weights)[0]
        
    async def _make_request(self, endpoint_name: str, endpoint_path: str) -> RequestResult:
        """Выполнение HTTP запроса"""
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}{endpoint_path}"
            
            # Выполнение запроса (синхронный, но в executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(url, timeout=10)
            )
            
            response_time = (time.time() - start_time) * 1000  # в мс
            payload_size = len(response.content) if response.content else 0
            
            return RequestResult(
                timestamp=time.time(),
                endpoint=endpoint_name,
                response_time_ms=response_time,
                status_code=response.status_code,
                success=200 <= response.status_code < 300,
                payload_size=payload_size
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return RequestResult(
                timestamp=time.time(),
                endpoint=endpoint_name,
                response_time_ms=response_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )

class SystemMonitor:
    """Мониторинг системных ресурсов во время нагрузочного тестирования"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.metrics: List[Dict[str, Any]] = []
        self.running = False
        
    async def start_monitoring(self):
        """Запуск мониторинга"""
        self.running = True
        
        while self.running:
            try:
                # Сбор метрик системы
                response = requests.get(f"{self.base_url}/api/v2/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    metric = {
                        'timestamp': time.time(),
                        'cpu_usage': data.get('cpu_usage', 0),
                        'memory_usage': data.get('memory_usage', 0),
                        'gpu_usage': data.get('gpu_usage'),
                        'current_fps': data.get('current_fps', 0),
                        'active_streams': data.get('active_streams', 0)
                    }
                    
                    self.metrics.append(metric)
                    
            except Exception as e:
                logger.debug(f"Ошибка мониторинга: {e}")
                
            await asyncio.sleep(2.0)  # Каждые 2 секунды
            
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.running = False
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик"""
        if not self.metrics:
            return {}
            
        return {
            'avg_cpu_usage': statistics.mean([m['cpu_usage'] for m in self.metrics]),
            'max_cpu_usage': max([m['cpu_usage'] for m in self.metrics]),
            'avg_memory_usage': statistics.mean([m['memory_usage'] for m in self.metrics]),
            'max_memory_usage': max([m['memory_usage'] for m in self.metrics]),
            'avg_fps': statistics.mean([m['current_fps'] for m in self.metrics if m['current_fps'] > 0]),
            'metrics_count': len(self.metrics)
        }

class LoadTester:
    """Главный класс для нагрузочного тестирования"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.system_monitor = SystemMonitor(config.base_url)
        
    async def run_load_test(self) -> LoadTestResults:
        """Запуск полного нагрузочного тестирования"""
        logger.info(f"🚀 Начало нагрузочного тестирования: {self.config.concurrent_users} клиентов")
        
        start_time = time.time()
        
        # Запуск системного мониторинга
        monitoring_task = asyncio.create_task(self.system_monitor.start_monitoring())
        
        try:
            # Создание клиентов
            clients = [
                LoadTestClient(i, self.config) 
                for i in range(self.config.concurrent_users)
            ]
            
            # Запуск клиентов с постепенным наращиванием нагрузки
            all_results = await self._run_clients_with_rampup(clients)
            
        finally:
            # Остановка мониторинга
            self.system_monitor.stop_monitoring()
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
                
        end_time = time.time()
        
        # Анализ результатов
        results = self._analyze_results(all_results, start_time, end_time)
        
        # Сохранение результатов
        await self._save_results(results)
        
        return results
        
    async def _run_clients_with_rampup(self, clients: List[LoadTestClient]) -> List[RequestResult]:
        """Запуск клиентов с постепенным наращиванием нагрузки"""
        all_results = []
        
        # Разделение клиентов на группы для ramp-up
        groups = self._split_into_groups(clients, 4)  # 4 группы
        rampup_interval = self.config.ramp_up_duration / len(groups)
        
        active_tasks = []
        
        for group_idx, group in enumerate(groups):
            # Задержка между группами
            if group_idx > 0:
                await asyncio.sleep(rampup_interval)
                
            logger.info(f"📈 Запуск группы {group_idx + 1}/{len(groups)} ({len(group)} клиентов)")
            
            # Запуск клиентов группы
            group_tasks = [
                asyncio.create_task(client.run_load_test())
                for client in group
            ]
            
            active_tasks.extend(group_tasks)
            
        # Ожидание завершения всех клиентов
        logger.info(f"⏳ Ожидание завершения {len(active_tasks)} клиентов...")
        
        completed_results = await asyncio.gather(*active_tasks, return_exceptions=True)
        
        # Сбор всех результатов
        for result in completed_results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Ошибка клиента: {result}")
                
        logger.info(f"✅ Собрано {len(all_results)} результатов запросов")
        return all_results
        
    def _split_into_groups(self, items: List, group_count: int) -> List[List]:
        """Разделение списка на группы"""
        group_size = len(items) // group_count
        groups = []
        
        for i in range(0, len(items), group_size):
            groups.append(items[i:i + group_size])
            
        return groups
        
    def _analyze_results(self, all_results: List[RequestResult], start_time: float, end_time: float) -> LoadTestResults:
        """Анализ результатов нагрузочного тестирования"""
        
        if not all_results:
            logger.error("Нет результатов для анализа")
            return LoadTestResults(
                test_name="load_test",
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                max_response_time=0,
                requests_per_second=0,
                bytes_per_second=0,
                error_rate=100.0,
                error_distribution={},
                passed=False,
                details={}
            )
            
        # Базовая статистика
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r.success)
        failed_requests = total_requests - successful_requests
        
        # Статистика времени ответа
        response_times = [r.response_time_ms for r in all_results]
        avg_response_time = statistics.mean(response_times)
        p50_response_time = np.percentile(response_times, 50)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        max_response_time = max(response_times)
        
        # Throughput
        test_duration = end_time - start_time
        requests_per_second = total_requests / test_duration
        total_bytes = sum(r.payload_size for r in all_results)
        bytes_per_second = total_bytes / test_duration
        
        # Ошибки
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        error_distribution = {}
        for result in all_results:
            if not result.success:
                error_key = f"{result.status_code}_{result.error_message or 'unknown'}"
                error_distribution[error_key] = error_distribution.get(error_key, 0) + 1
                
        # Определение успешности теста
        passed = (
            requests_per_second >= self.config.target_rps * 0.8 and  # 80% от целевого RPS
            p95_response_time <= self.config.max_response_time and
            error_rate <= self.config.max_error_rate
        )
        
        # Системные метрики
        system_metrics = self.system_monitor.get_metrics_summary()
        
        results = LoadTestResults(
            test_name="load_test",
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            bytes_per_second=bytes_per_second,
            error_rate=error_rate,
            error_distribution=error_distribution,
            passed=passed,
            details={
                'test_duration': test_duration,
                'system_metrics': system_metrics,
                'endpoint_distribution': self._get_endpoint_distribution(all_results),
                'timeline_stats': self._get_timeline_stats(all_results)
            }
        )
        
        return results
        
    def _get_endpoint_distribution(self, results: List[RequestResult]) -> Dict[str, Dict[str, Any]]:
        """Статистика по эндпоинтам"""
        endpoint_stats = {}
        
        for result in results:
            if result.endpoint not in endpoint_stats:
                endpoint_stats[result.endpoint] = {
                    'total': 0,
                    'successful': 0,
                    'response_times': []
                }
                
            stats = endpoint_stats[result.endpoint]
            stats['total'] += 1
            if result.success:
                stats['successful'] += 1
            stats['response_times'].append(result.response_time_ms)
            
        # Вычисление статистики для каждого эндпоинта
        for endpoint, stats in endpoint_stats.items():
            response_times = stats['response_times']
            stats.update({
                'success_rate': (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'p95_response_time': np.percentile(response_times, 95) if len(response_times) > 1 else (response_times[0] if response_times else 0)
            })
            del stats['response_times']  # Удаляем сырые данные для экономии места
            
        return endpoint_stats
        
    def _get_timeline_stats(self, results: List[RequestResult]) -> Dict[str, Any]:
        """Статистика по временным интервалам"""
        if not results:
            return {}
            
        start_time = min(r.timestamp for r in results)
        end_time = max(r.timestamp for r in results)
        
        # Разделение на 10 интервалов
        interval_duration = (end_time - start_time) / 10
        intervals = []
        
        for i in range(10):
            interval_start = start_time + i * interval_duration
            interval_end = interval_start + interval_duration
            
            interval_results = [
                r for r in results 
                if interval_start <= r.timestamp < interval_end
            ]
            
            if interval_results:
                intervals.append({
                    'interval': i + 1,
                    'start_time': interval_start,
                    'requests': len(interval_results),
                    'rps': len(interval_results) / interval_duration,
                    'avg_response_time': statistics.mean([r.response_time_ms for r in interval_results]),
                    'success_rate': sum(1 for r in interval_results if r.success) / len(interval_results) * 100
                })
                
        return {
            'intervals': intervals,
            'total_duration': end_time - start_time
        }
        
    async def _save_results(self, results: LoadTestResults):
        """Сохранение результатов"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results_{timestamp}.json"
        
        results_dir = Path("logs/load_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / filename
        
        # Преобразование в JSON-сериализуемый формат
        results_dict = asdict(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
            
        logger.info(f"💾 Результаты сохранены в {results_file}")

async def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Нагрузочное тестирование PigWeight")
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL сервера')
    parser.add_argument('--users', type=int, default=16, help='Количество одновременных пользователей')
    parser.add_argument('--requests', type=int, default=100, help='Запросов на пользователя')
    parser.add_argument('--duration', type=int, default=300, help='Длительность теста (сек)')
    parser.add_argument('--ramp-up', type=int, default=60, help='Время наращивания нагрузки (сек)')
    parser.add_argument('--target-rps', type=float, default=50.0, help='Целевой RPS')
    
    args = parser.parse_args()
    
    # Создание конфигурации
    config = LoadTestConfig(
        base_url=args.url,
        concurrent_users=args.users,
        requests_per_user=args.requests,
        ramp_up_duration=args.ramp_up,
        test_duration=args.duration,
        target_rps=args.target_rps
    )
    
    # Запуск тестирования
    load_tester = LoadTester(config)
    
    try:
        results = await load_tester.run_load_test()
        
        # Вывод результатов
        print("\n" + "="*60)
        print("🔥 РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТИРОВАНИЯ")
        print("="*60)
        print(f"Общий результат: {'✅ УСПЕХ' if results.passed else '❌ НЕУДАЧА'}")
        print(f"Всего запросов: {results.total_requests}")
        print(f"Успешных: {results.successful_requests} ({((results.successful_requests/results.total_requests)*100):.1f}%)")
        print(f"Ошибок: {results.failed_requests} ({results.error_rate:.1f}%)")
        print(f"RPS: {results.requests_per_second:.1f} (цель: {config.target_rps})")
        print(f"Среднее время ответа: {results.avg_response_time:.1f} мс")
        print(f"P95 время ответа: {results.p95_response_time:.1f} мс")
        print(f"Максимальное время: {results.max_response_time:.1f} мс")
        
        if results.details.get('system_metrics'):
            sys_metrics = results.details['system_metrics']
            print(f"\n💻 Системные ресурсы:")
            print(f"Средний CPU: {sys_metrics.get('avg_cpu_usage', 0):.1f}%")
            print(f"Максимальный CPU: {sys_metrics.get('max_cpu_usage', 0):.1f}%")
            print(f"Средняя память: {sys_metrics.get('avg_memory_usage', 0):.1f}%")
            print(f"Максимальная память: {sys_metrics.get('max_memory_usage', 0):.1f}%")
            
        print("="*60)
        
        return 0 if results.passed else 1
        
    except Exception as e:
        logger.error(f"❌ Ошибка нагрузочного тестирования: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)