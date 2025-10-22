"""
Полное тестирование всех API эндпоинтов системы отслеживания свиней
ВКЛАДКА 2: Запуск и тестирование API
"""

import requests
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

BASE_URL = "http://localhost:8080"

def print_section(title: str):
    """Печатает заголовок секции"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_endpoint(name: str, method: str, url: str, **kwargs) -> tuple[bool, Any]:
    """Универсальная функция для тестирования эндпоинта"""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=5, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, timeout=5, **kwargs)
        else:
            return False, f"Неподдерживаемый метод: {method}"
        
        if response.status_code == 200:
            try:
                data = response.json()
                return True, data
            except:
                return True, response.text
        else:
            return False, f"Статус {response.status_code}: {response.text[:200]}"
    except requests.exceptions.ConnectionError:
        return False, "Не удалось подключиться к серверу"
    except Exception as e:
        return False, str(e)

# ============================================================================
# БАЗОВЫЕ ЭНДПОИНТЫ
# ============================================================================

def test_health():
    """Health check"""
    print_section("1. Health Check")
    success, data = test_endpoint("Health", "GET", f"{BASE_URL}/health")
    if success:
        print(f"✅ Сервер работает")
        print(f"   Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
    else:
        print(f"❌ Ошибка: {data}")
    return success

def test_swagger():
    """Swagger UI"""
    print_section("2. Swagger UI")
    success, data = test_endpoint("Swagger", "GET", f"{BASE_URL}/docs")
    if success:
        print(f"✅ Swagger UI доступен: {BASE_URL}/docs")
    else:
        print(f"❌ Ошибка: {data}")
    return success

def test_cameras():
    """Список камер"""
    print_section("3. Список камер")
    success, data = test_endpoint("Cameras", "GET", f"{BASE_URL}/api/cameras")
    if success:
        print(f"✅ Эндпоинт работает")
        if isinstance(data, dict):
            print(f"   Камер найдено: {len(data)}")
            for cam_id, cam_url in list(data.items())[:3]:
                print(f"   - {cam_id}: {cam_url[:50]}...")
        else:
            print(f"   Ответ: {data}")
    else:
        print(f"❌ Ошибка: {data}")
    return success

# ============================================================================
# ЖУРНАЛ АКТОВ ВЗВЕШИВАНИЯ
# ============================================================================

def test_journal_acts():
    """Получение актов из журнала"""
    print_section("4. Журнал актов взвешивания")
    
    # Пробуем разные эндпоинты
    endpoints = [
        ("/api/journal/acts", "GET"),
        ("/api/journal/list", "GET"),
        ("/api/weighing/logs", "GET"),
    ]
    
    for endpoint, method in endpoints:
        print(f"\nПроверка: {method} {endpoint}")
        success, data = test_endpoint("Journal Acts", method, f"{BASE_URL}{endpoint}")
        if success:
            print(f"✅ Эндпоинт работает")
            if isinstance(data, list):
                print(f"   Актов найдено: {len(data)}")
                if data:
                    print(f"\n   Пример первого акта:")
                    print("   " + json.dumps(data[0], indent=2, ensure_ascii=False).replace("\n", "\n   "))
                    return True
            elif isinstance(data, dict) and 'acts' in data:
                print(f"   Актов найдено: {len(data['acts'])}")
                return True
            else:
                print(f"   Ответ: {str(data)[:200]}")
        else:
            print(f"⚠️  {data}")
    
    print("\nℹ️  Актов пока нет. Обработайте видео через console_app.py")
    return False

def test_weighing_stats():
    """Статистика взвешиваний"""
    print_section("5. Статистика взвешиваний")
    success, data = test_endpoint("Weighing Stats", "GET", f"{BASE_URL}/api/weighing/stats")
    if success:
        print(f"✅ Эндпоинт работает")
        print(f"   Статистика: {json.dumps(data, indent=2, ensure_ascii=False)}")
    else:
        print(f"⚠️  {data}")
    return success

# ============================================================================
# ЗАПИСИ (RECORDS)
# ============================================================================

def test_records():
    """Список записей актов"""
    print_section("6. Записи актов (Records)")
    success, data = test_endpoint("Records", "GET", f"{BASE_URL}/api/records")
    if success:
        print(f"✅ Эндпоинт работает")
        if isinstance(data, list):
            print(f"   Записей найдено: {len(data)}")
            for record in data[:3]:
                if isinstance(record, dict):
                    print(f"   - {record.get('name', 'N/A')}: {record.get('acts_count', 0)} актов")
        else:
            print(f"   Ответ: {str(data)[:200]}")
    else:
        print(f"⚠️  {data}")
    return success

# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ЭНДПОИНТЫ
# ============================================================================

def test_dashboard():
    """Дашборд"""
    print_section("7. Дашборд")
    success, data = test_endpoint("Dashboard", "GET", f"{BASE_URL}/dashboard")
    if success:
        print(f"✅ Дашборд доступен: {BASE_URL}/dashboard")
    else:
        print(f"⚠️  {data}")
    return success

def test_monitoring():
    """Мониторинг"""
    print_section("8. Мониторинг")
    success, data = test_endpoint("Monitoring", "GET", f"{BASE_URL}/monitoring")
    if success:
        print(f"✅ Мониторинг доступен: {BASE_URL}/monitoring")
    else:
        print(f"⚠️  {data}")
    return success

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция тестирования"""
    print("\n" + "🚀 " * 25)
    print("  ПОЛНОЕ ТЕСТИРОВАНИЕ API ЭНДПОИНТОВ")
    print("  Система отслеживания свиней - ВКЛАДКА 2")
    print("🚀 " * 25)
    
    print(f"\nБазовый URL: {BASE_URL}")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Запускаем все тесты
    tests = [
        ("Health Check", test_health),
        ("Swagger UI", test_swagger),
        ("Cameras API", test_cameras),
        ("Journal Acts", test_journal_acts),
        ("Weighing Stats", test_weighing_stats),
        ("Records API", test_records),
        ("Dashboard", test_dashboard),
        ("Monitoring", test_monitoring),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Критическая ошибка в тесте '{test_name}': {e}")
            results[test_name] = False
    
    # Итоговый отчет
    print_section("ИТОГОВЫЙ ОТЧЕТ")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\n{'='*70}")
    print(f"Всего тестов: {total}")
    print(f"Пройдено: {passed} ({passed/total*100:.1f}%)")
    print(f"Провалено: {total - passed}")
    print(f"{'='*70}")
    
    # Рекомендации
    if passed < total:
        print("\n📋 РЕКОМЕНДАЦИИ:")
        if not results.get("Health Check"):
            print("  1. Запустите API сервер:")
            print("     python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload")
        if not results.get("Journal Acts"):
            print("  2. Обработайте тестовое видео:")
            print("     python console_app.py")
        print()
    else:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n📊 Следующие шаги:")
        print("  1. Откройте Swagger UI: http://localhost:8080/docs")
        print("  2. Откройте Dashboard: http://localhost:8080/dashboard")
        print("  3. Откройте Monitoring: http://localhost:8080/monitoring")
        print()
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
