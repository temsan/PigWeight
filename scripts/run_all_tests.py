#!/usr/bin/env python3
"""
Мастер-скрипт для запуска всех тестов производительности и валидации
"""

import asyncio
import sys
import time
import json
import logging
from pathlib import Path
import subprocess
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSuite:
    """Набор тестов для комплексной валидации"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
    async def run_all_tests(self) -> dict:
        """Запуск всех тестов"""
        logger.info("🎯 Начало комплексного тестирования PigWeight")
        
        suite_start = time.time()
        
        # 1. Проверка готовности системы
        if not await self._check_system_ready():
            logger.error("❌ Система не готова к тестированию")
            return {"success": False, "error": "System not ready"}
            
        # 2. Валидация производительности
        logger.info("📊 Запуск валидации производительности...")
        perf_result = await self._run_performance_validation()
        self.results['performance_validation'] = perf_result
        
        # 3. Нагрузочное тестирование
        logger.info("🔥 Запуск нагрузочного тестирования...")
        load_result = await self._run_load_testing()
        self.results['load_testing'] = load_result
        
        # 4. Тест стресс-сценариев
        logger.info("💥 Запуск стресс-тестирования...")
        stress_result = await self._run_stress_testing()
        self.results['stress_testing'] = stress_result
        
        # 5. Финальный отчет
        suite_duration = time.time() - suite_start
        final_report = self._generate_comprehensive_report(suite_duration)
        
        # 6. Сохранение результатов
        await self._save_comprehensive_results(final_report)
        
        logger.info(f"🏁 Комплексное тестирование завершено за {suite_duration:.1f} секунд")
        
        return final_report
        
    async def _check_system_ready(self) -> bool:
        """Проверка готовности системы"""
        try:
            import requests
            
            # Проверка базового API
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"API health check failed: {response.status_code}")
                return False
                
            # Проверка оптимизированных endpoints
            response = requests.get(f"{self.base_url}/api/v2/status", timeout=10)
            if response.status_code != 200:
                logger.error(f"Optimized API not ready: {response.status_code}")
                return False
                
            logger.info("✅ Система готова к тестированию")
            return True
            
        except Exception as e:
            logger.error(f"System readiness check failed: {e}")
            return False
            
    async def _run_performance_validation(self) -> dict:
        """Запуск валидации производительности"""
        try:
            # Импорт и запуск валидатора
            from performance_validation import PerformanceValidator, PerformanceTarget
            
            targets = PerformanceTarget(
                min_fps=60.0,
                max_latency_ms=100.0,
                test_duration_seconds=120
            )
            
            validator = PerformanceValidator(self.base_url, targets)
            result = await validator.run_full_validation()
            
            return {
                "success": result['overall_result'] == 'PASSED',
                "details": result
            }
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _run_load_testing(self) -> dict:
        """Запуск нагрузочного тестирования"""
        try:
            from load_testing import LoadTester, LoadTestConfig
            
            config = LoadTestConfig(
                base_url=self.base_url,
                concurrent_users=16,
                requests_per_user=50,
                test_duration=180,
                ramp_up_duration=30
            )
            
            load_tester = LoadTester(config)
            result = await load_tester.run_load_test()
            
            return {
                "success": result.passed,
                "requests_per_second": result.requests_per_second,
                "error_rate": result.error_rate,
                "avg_response_time": result.avg_response_time,
                "details": result
            }
            
        except Exception as e:
            logger.error(f"Load testing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _run_stress_testing(self) -> dict:
        """Запуск стресс-тестирования"""
        try:
            # Стресс-тест с экстремальными параметрами
            from load_testing import LoadTester, LoadTestConfig
            
            stress_config = LoadTestConfig(
                base_url=self.base_url,
                concurrent_users=50,  # Больше пользователей
                requests_per_user=200,  # Больше запросов
                test_duration=300,  # Дольше
                ramp_up_duration=60,
                target_rps=100.0,  # Выше RPS
                max_error_rate=10.0  # Более мягкие требования к ошибкам
            )
            
            stress_tester = LoadTester(stress_config)
            result = await stress_tester.run_load_test()
            
            return {
                "success": result.passed,
                "max_concurrent_users": stress_config.concurrent_users,
                "peak_rps": result.requests_per_second,
                "system_stable": result.error_rate < 15.0,  # Система стабильна если < 15% ошибок
                "details": result
            }
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _generate_comprehensive_report(self, suite_duration: float) -> dict:
        """Генерация комплексного отчета"""
        
        # Подсчет общих результатов
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('success', False))
        
        overall_success = passed_tests == total_tests
        
        # Агрегированные метрики
        metrics = {
            "performance": self._extract_performance_metrics(),
            "load": self._extract_load_metrics(),
            "stress": self._extract_stress_metrics()
        }
        
        # Рекомендации
        recommendations = self._generate_recommendations()
        
        report = {
            "test_suite_version": "1.0.0",
            "timestamp": time.time(),
            "duration_seconds": suite_duration,
            
            # Общие результаты
            "overall_result": "PASSED" if overall_success else "FAILED",
            "tests_total": total_tests,
            "tests_passed": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            
            # Детальные результаты
            "test_results": self.results,
            
            # Агрегированные метрики
            "performance_summary": metrics,
            
            # Рекомендации
            "recommendations": recommendations,
            
            # Сертификация
            "certification": self._generate_certification(overall_success, metrics)
        }
        
        return report
        
    def _extract_performance_metrics(self) -> dict:
        """Извлечение метрик производительности"""
        perf_result = self.results.get('performance_validation', {})
        
        if not perf_result.get('success'):
            return {"available": False}
            
        details = perf_result.get('details', {})
        summary = details.get('performance_summary', {})
        
        return {
            "available": True,
            "avg_fps": summary.get('avg_fps', 0),
            "min_fps": summary.get('min_fps', 0),
            "avg_latency_ms": summary.get('avg_latency_ms', 0),
            "p95_latency_ms": summary.get('p95_latency_ms', 0),
            "fps_target_met": summary.get('fps_target_met', False),
            "latency_target_met": summary.get('latency_target_met', False)
        }
        
    def _extract_load_metrics(self) -> dict:
        """Извлечение метрик нагрузки"""
        load_result = self.results.get('load_testing', {})
        
        if not load_result.get('success'):
            return {"available": False}
            
        return {
            "available": True,
            "requests_per_second": load_result.get('requests_per_second', 0),
            "error_rate": load_result.get('error_rate', 0),
            "avg_response_time": load_result.get('avg_response_time', 0)
        }
        
    def _extract_stress_metrics(self) -> dict:
        """Извлечение метрик стресса"""
        stress_result = self.results.get('stress_testing', {})
        
        if not stress_result.get('success'):
            return {"available": False}
            
        return {
            "available": True,
            "max_concurrent_users": stress_result.get('max_concurrent_users', 0),
            "peak_rps": stress_result.get('peak_rps', 0),
            "system_stable": stress_result.get('system_stable', False)
        }
        
    def _generate_recommendations(self) -> list:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Анализ производительности
        perf_metrics = self._extract_performance_metrics()
        if perf_metrics.get('available'):
            if not perf_metrics.get('fps_target_met'):
                recommendations.append(
                    "❗ FPS ниже целевого. Рекомендации: увеличить batch_size, "
                    "включить H.264 Direct, проверить GPU утилизацию."
                )
                
            if not perf_metrics.get('latency_target_met'):
                recommendations.append(
                    "❗ Латентность выше целевой. Рекомендации: уменьшить batch_size, "
                    "оптимизировать очередь кадров, включить адаптивное качество."
                )
                
        # Анализ нагрузки
        load_metrics = self._extract_load_metrics()
        if load_metrics.get('available'):
            if load_metrics.get('error_rate', 0) > 5.0:
                recommendations.append(
                    "❗ Высокий процент ошибок под нагрузкой. Проверьте стабильность "
                    "системы и масштабируемость компонентов."
                )
                
        # Анализ стресса
        stress_metrics = self._extract_stress_metrics()
        if stress_metrics.get('available'):
            if not stress_metrics.get('system_stable'):
                recommendations.append(
                    "❗ Система нестабильна под стрессом. Рекомендуется "
                    "горизонтальное масштабирование или увеличение ресурсов."
                )
                
        if not recommendations:
            recommendations.append(
                "✅ Все тесты пройдены успешно! Система демонстрирует отличную "
                "производительность и стабильность."
            )
            
        return recommendations
        
    def _generate_certification(self, overall_success: bool, metrics: dict) -> dict:
        """Генерация сертификата производительности"""
        
        # Определение уровня сертификации
        if not overall_success:
            level = "FAILED"
            score = 0
        else:
            score = 0
            max_score = 100
            
            # FPS оценка (25 баллов)
            perf = metrics.get('performance', {})
            if perf.get('available'):
                fps = perf.get('avg_fps', 0)
                if fps >= 60:
                    score += 25
                elif fps >= 40:
                    score += 20
                elif fps >= 30:
                    score += 15
                elif fps >= 20:
                    score += 10
                    
            # Латентность оценка (25 баллов)
            if perf.get('available'):
                latency = perf.get('avg_latency_ms', 999)
                if latency <= 50:
                    score += 25
                elif latency <= 75:
                    score += 20
                elif latency <= 100:
                    score += 15
                elif latency <= 150:
                    score += 10
                    
            # Нагрузочная стабильность (25 баллов)
            load = metrics.get('load', {})
            if load.get('available'):
                error_rate = load.get('error_rate', 100)
                if error_rate <= 1:
                    score += 25
                elif error_rate <= 3:
                    score += 20
                elif error_rate <= 5:
                    score += 15
                elif error_rate <= 10:
                    score += 10
                    
            # Стресс-устойчивость (25 баллов)
            stress = metrics.get('stress', {})
            if stress.get('available'):
                if stress.get('system_stable'):
                    score += 25
                    peak_rps = stress.get('peak_rps', 0)
                    if peak_rps >= 100:
                        score += 5  # Бонус за высокую пропускную способность
                        
            # Определение уровня
            if score >= 95:
                level = "PLATINUM"
            elif score >= 85:
                level = "GOLD"
            elif score >= 75:
                level = "SILVER"
            elif score >= 60:
                level = "BRONZE"
            else:
                level = "BASIC"
                
        return {
            "level": level,
            "score": score,
            "max_score": 100,
            "percentage": score,
            "issued_at": time.time(),
            "valid_until": time.time() + (30 * 24 * 3600),  # 30 дней
            "certificate_id": f"PW-{int(time.time())}-{level}"
        }
        
    async def _save_comprehensive_results(self, report: dict):
        """Сохранение комплексного отчета"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Создание директории результатов
        results_dir = Path("logs/comprehensive_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Полный отчет
        full_report_file = results_dir / f"comprehensive_test_report_{timestamp}.json"
        with open(full_report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
        # Краткий отчет
        summary_report = {
            "timestamp": report["timestamp"],
            "overall_result": report["overall_result"],
            "success_rate": report["success_rate"],
            "performance_summary": report["performance_summary"],
            "certification": report["certification"],
            "recommendations": report["recommendations"]
        }
        
        summary_file = results_dir / f"test_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False, default=str)
            
        logger.info(f"📋 Полный отчет: {full_report_file}")
        logger.info(f"📋 Краткий отчет: {summary_file}")

async def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Комплексное тестирование PigWeight")
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL сервера')
    parser.add_argument('--quick', action='store_true', help='Быстрое тестирование (сокращенные тесты)')
    
    args = parser.parse_args()
    
    # Создание и запуск набора тестов
    test_suite = TestSuite(args.url)
    
    try:
        report = await test_suite.run_all_tests()
        
        # Вывод результатов
        print("\n" + "="*80)
        print("🎯 КОМПЛЕКСНЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ PIGWEIGHT")
        print("="*80)
        
        cert = report["certification"]
        result_emoji = "✅" if report["overall_result"] == "PASSED" else "❌"
        
        print(f"{result_emoji} Общий результат: {report['overall_result']}")
        print(f"📊 Успешность: {report['success_rate']:.1f}% ({report['tests_passed']}/{report['tests_total']} тестов)")
        print(f"🏆 Сертификация: {cert['level']} ({cert['score']}/100 баллов)")
        print(f"⏱️ Длительность: {report['duration_seconds']:.1f} секунд")
        
        # Производительность
        perf = report["performance_summary"]["performance"]
        if perf.get("available"):
            print(f"\n📈 Производительность:")
            print(f"   FPS: {perf['avg_fps']:.1f} (мин: {perf['min_fps']:.1f})")
            print(f"   Латентность: {perf['avg_latency_ms']:.1f}ms (P95: {perf['p95_latency_ms']:.1f}ms)")
            
        # Нагрузка
        load = report["performance_summary"]["load"] 
        if load.get("available"):
            print(f"\n🔥 Нагрузочное тестирование:")
            print(f"   RPS: {load['requests_per_second']:.1f}")
            print(f"   Ошибки: {load['error_rate']:.1f}%")
            print(f"   Отклик: {load['avg_response_time']:.1f}ms")
            
        # Стресс
        stress = report["performance_summary"]["stress"]
        if stress.get("available"):
            print(f"\n💥 Стресс-тестирование:")
            print(f"   Макс. пользователей: {stress['max_concurrent_users']}")
            print(f"   Пиковый RPS: {stress['peak_rps']:.1f}")
            print(f"   Стабильность: {'✅' if stress['system_stable'] else '❌'}")
            
        print(f"\n💡 Рекомендации:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
            
        print("\n🏅 Сертификат производительности:")
        print(f"   ID: {cert['certificate_id']}")
        print(f"   Уровень: {cert['level']}")
        print(f"   Оценка: {cert['score']}/100")
        
        print("="*80)
        
        return 0 if report["overall_result"] == "PASSED" else 1
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка тестирования: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)