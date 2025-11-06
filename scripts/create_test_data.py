"""
Скрипт для создания тестовых данных в БД
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Добавляем корневую папку в путь
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pig_tracking.database_manager import DatabaseManager
from dotenv import load_dotenv

# Загрузить .env
load_dotenv()

def create_test_data():
    """Создать тестовые акты взвешивания"""
    
    # Подключение к БД
    db = DatabaseManager(
        supabase_url=os.getenv("SUPABASE_URL", "http://localhost:54321"),
        supabase_key=os.getenv("SUPABASE_KEY")
    )
    
    print("🔄 Создание тестовых данных...")
    
    # Создаем 5 тестовых актов за последние 3 дня
    base_time = datetime.now() - timedelta(days=2)
    
    test_acts = []
    for i in range(5):
        started_at = base_time + timedelta(hours=i*3)
        ended_at = started_at + timedelta(minutes=15)
        duration_sec = (ended_at - started_at).total_seconds()
        
        act_data = {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_sec": int(duration_sec),
            "left_count": 20 + i * 5,
            "right_count": 18 + i * 4,
            "peak_count": 12 + i * 2,
            "total_weight": 1500.0 + i * 200,
            "avg_weight": 35.0 + i * 2,
            "stream_id": "test_stream"
        }
        
        try:
            # Прямая вставка через Supabase client
            response = db.client.table("weighing_acts").insert(act_data).execute()
            if response.data:
                act_id = response.data[0]['id']
                test_acts.append(act_id)
                print(f"✅ Создан акт #{i+1}: ID={act_id}, время={started_at.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"❌ Ошибка создания акта #{i+1}: нет данных в ответе")
        except Exception as e:
            print(f"❌ Ошибка создания акта #{i+1}: {e}")
    
    print(f"\n✅ Создано {len(test_acts)} тестовых актов")
    print(f"📊 IDs: {test_acts}")
    
    # Проверяем созданные данные
    print("\n🔍 Проверка данных...")
    acts = db.get_acts_by_period(
        start_date=base_time,
        end_date=datetime.now()
    )
    print(f"✅ Найдено актов в БД: {len(acts)}")
    
    return test_acts

if __name__ == "__main__":
    try:
        test_acts = create_test_data()
        print("\n🎉 Тестовые данные успешно созданы!")
        print("\n📝 Теперь можно тестировать API:")
        print("   curl http://localhost:8000/api/weighing/acts")
        print("   curl http://localhost:8000/api/weighing/stats")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
