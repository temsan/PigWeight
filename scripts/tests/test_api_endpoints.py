"""
Тестирование API эндпоинтов системы отслеживания свиней
"""

import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8080"

def print_section(title: str):
    """Печатает заголовок секции"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health() -> bool:
    """Тест health check эндпоинта"""
    print_section("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ Ошибка: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к API серверу")
        print(f"   Убедитесь, что сервер запущен: python -m uvicorn api.app:app --port 8080")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_statistics() -> bool:
    """Тест эндпоинта статистики"""
    print_section("Статистика")
    try:
        response = requests.get(f"{BASE_URL}/api/statistics", timeout=5)
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"⚠️ Статус {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_weighing_acts() -> bool:
    """Тест эндпоинта актов взвешивания"""
    print_section("Акты взвешивания")
    try:
        response = requests.get(f"{BASE_URL}/api/weighing-acts", timeout=5)
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            acts = response.json()
            print(f"✅ Найдено актов: {len(acts)}")
            if acts:
                print("\nПример первого акта:")
                print(json.dumps(acts[0], indent=2, ensure_ascii=False))
            else:
                print("ℹ️  Актов пока нет (обработайте видео через console_app.py)")
            return True
        else:
            print(f"⚠️ Статус {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_crossings() -> bool:
    """Тест эндпоинта пересечений линий"""
    print_section("Пересечения линий")
    try:
        response = requests.get(f"{BASE_URL}/api/line-crossings?limit=10", timeout=5)
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            crossings = response.json()
            print(f"✅ Найдено пересечений: {len(crossings)}")
            if crossings:
                print("\nПример первого пересечения:")
                print(json.dumps(crossings[0], indent=2, ensure_ascii=False))
            else:
                print("ℹ️  Пересечений пока нет (обработайте видео через console_app.py)")
            return True
        else:
            print(f"⚠️ Статус {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_swagger_ui() -> bool:
    """Проверка доступности Swagger UI"""
    print_section("Swagger UI")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Swagger UI доступен: {BASE_URL}/docs")
            return True
        else:
            print(f"⚠️ Swagger UI недоступен")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("\n" + "🚀 " * 20)
    print("  ТЕСТИРОВАНИЕ API ЭНДПОИНТОВ")
    print("  Система отслеживания свиней")
    print("🚀 " * 20)
    
    results = {
        "Health Check": test_health(),
        "Swagger UI": test_swagger_ui(),
        "Statistics": test_statistics(),
        "Weighing Acts": test_weighing_acts(),
        "Line Crossings": test_crossings()
    }
    
    # Итоговый отчет
    print_section("ИТОГОВЫЙ ОТЧЕТ")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nВсего тестов: {total}")
    print(f"Пройдено: {passed}")
    print(f"Провалено: {total - passed}")
    
    if passed == total:
        print("\n🎉 Все тесты пройдены успешно!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} тест(ов) провалено")
        return 1

if __name__ == '__main__':
    sys.exit(main())
