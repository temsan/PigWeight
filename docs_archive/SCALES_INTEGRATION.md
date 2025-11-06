# 📡 Интеграция с IP весами (Scales Integration)

## Обзор

PigWeight получает данные о **весе каждого животного** от IP весов, которые рассылают данные по сети. На основе этих данных система вычисляет:
- **Средний вес** акта взвешивания
- **Общий вес** (сумма всех животных)
- **Распределение весов** (макс, мин, медиана)

---

## 🏗️ Архитектура интеграции

```
IP ВЕСЫ 📡                 PIGWEIGHT СИСТЕМА              ДАШБОРД/EXCEL
┌──────────────┐          ┌─────────────────┐            ┌────────────┐
│ Весы рассылают│          │  WebSocket      │            │ Мобильный  │
│ вес 125.5 кг │ ──TCP/UDP──► Listener      │ ──REST───► │ дашборд    │
│              │          │                 │            │ (100 кг)   │
│ Весы рассылают│          │ Store weights   │            │            │
│ вес 98.2 кг  │ ──TCP/UDP──► in DB         │ ──JSON───► │ Excel      │
│              │          │                 │            │ экспорт    │
│ Весы рассылают│          │ Calculate       │            │            │
│ вес 115.0 кг │ ──TCP/UDP──► Average:      │ ──Socket─► │ Консоль    │
└──────────────┘          │ 112.9 кг        │            └────────────┘
                          └─────────────────┘
```

---

## 📥 Получение данных

### Протоколы передачи

#### Опция 1: TCP Socket (рекомендуется)

```python
# Пример: Весы рассылают вес по TCP
import socket
import json

def receive_weights():
    """Получить вес с IP весов по TCP"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 9000))
    sock.listen(1)
    
    while True:
        conn, addr = sock.accept()
        data = conn.recv(1024).decode('utf-8')
        weight_data = json.loads(data)
        
        print(f"Получен вес: {weight_data['weight']} кг от {addr}")
        # {"weight": 125.5, "timestamp": "2025-11-06T10:30:45", "pig_id": "pig_001"}
        
        process_weight(weight_data)
        conn.close()
```

#### Опция 2: UDP Broadcast

```python
import socket
import json

def receive_weights_udp():
    """Получить вес по UDP (broadcast)"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', 9001))
    
    while True:
        data, addr = sock.recvfrom(1024)
        weight_data = json.loads(data.decode('utf-8'))
        print(f"UDP вес: {weight_data['weight']} кг")
```

#### Опция 3: HTTP POST

```python
# Весы делают POST запрос на сервер
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/api/scales/register-weight")
async def register_weight(weight: float, timestamp: str = None):
    """
    Весы отправляют вес через HTTP POST
    
    Пример: 
    POST http://localhost:8000/api/scales/register-weight
    {"weight": 125.5, "timestamp": "2025-11-06T10:30:45"}
    """
    try:
        weight_data = {
            "weight": weight,
            "timestamp": timestamp or datetime.now().isoformat(),
            "source": "ip_scales"
        }
        
        # Сохранить в БД
        store_weight(weight_data)
        
        # Пересчитать средний вес
        avg_weight = calculate_average()
        
        return {
            "status": "success",
            "weight": weight,
            "average": avg_weight
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## 💾 Хранение и обработка

### Модель данных в БД

```sql
-- Таблица для хранения весов
CREATE TABLE weights (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    weight_value FLOAT NOT NULL,
    source VARCHAR(50) DEFAULT 'ip_scales',  -- ip_scales, manual, estimated
    crossing_id INTEGER REFERENCES crossings(id),
    act_id INTEGER REFERENCES weighing_acts(id),
    metadata JSONB  -- доп. информация
);

-- Индекс для быстрого поиска
CREATE INDEX idx_weights_timestamp ON weights(timestamp);
CREATE INDEX idx_weights_act ON weights(act_id);
```

### Сохранение веса в Python

```python
from pig_tracking.database import DatabaseManager

async def store_weight(weight_data: dict):
    """Сохранить вес в БД"""
    db = DatabaseManager()
    
    # Вес = факт с IP весов
    await db.add_weight(
        weight_value=weight_data['weight'],
        timestamp=weight_data['timestamp'],
        source='ip_scales',
        crossing_id=weight_data.get('crossing_id'),
        metadata={
            'raw_data': weight_data,
            'origin': 'scales_broadcast'
        }
    )
```

---

## 🧮 Вычисление среднего веса

### Простой расчёт

```python
def calculate_average_weight(act_id: int) -> float:
    """
    Вычислить средний вес для акта взвешивания
    
    Средний вес = Сумма всех весов / Количество животных
    """
    db = DatabaseManager()
    weights = db.get_weights_by_act(act_id)
    
    if not weights:
        return 0.0
    
    total_weight = sum(w['weight_value'] for w in weights)
    average = total_weight / len(weights)
    
    return round(average, 2)  # Округлить до 2 знаков
```

### С фильтрацией выбросов (outliers)

```python
import statistics

def calculate_average_weight_filtered(act_id: int, exclude_outliers=True) -> dict:
    """
    Вычислить средний вес с опциональным исключением выбросов
    
    Используется IQR (Interquartile Range) метод
    """
    db = DatabaseManager()
    weights = [w['weight_value'] for w in db.get_weights_by_act(act_id)]
    
    if not weights:
        return {'average': 0.0, 'count': 0, 'removed_outliers': 0}
    
    removed = 0
    if exclude_outliers and len(weights) > 4:
        # Исключить выбросы (значения > 1.5 * IQR)
        q1 = statistics.quantiles(weights, n=4)[0]
        q3 = statistics.quantiles(weights, n=4)[2]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_weights = [w for w in weights if lower_bound <= w <= upper_bound]
        removed = len(weights) - len(filtered_weights)
        weights = filtered_weights
    
    average = sum(weights) / len(weights) if weights else 0.0
    
    return {
        'average': round(average, 2),
        'count': len(weights),
        'min': min(weights) if weights else 0,
        'max': max(weights) if weights else 0,
        'median': statistics.median(weights) if weights else 0,
        'removed_outliers': removed
    }
```

### Динамическое обновление при новом весе

```python
class WeightCalculator:
    """Калькулятор среднего веса в реальном времени"""
    
    def __init__(self):
        self.weights = []
        self.sum_weight = 0.0
    
    def add_weight(self, weight: float) -> float:
        """
        Добавить новый вес и вернуть обновленный средний вес
        
        Используется online алгоритм (не нужно пересчитывать всё заново)
        """
        self.weights.append(weight)
        self.sum_weight += weight
        
        avg = self.sum_weight / len(self.weights)
        return round(avg, 2)
    
    def get_average(self) -> float:
        """Получить текущий средний вес"""
        if not self.weights:
            return 0.0
        return round(self.sum_weight / len(self.weights), 2)
    
    def reset(self):
        """Сброс для нового акта"""
        self.weights = []
        self.sum_weight = 0.0
```

---

## 🎯 Использование в мобильном дашборде

### Получение среднего веса через API

```javascript
// static/mobile-dashboard.html

async function fetchMetrics() {
    const response = await fetch('/api/metrics/current?stream_id=cam101');
    const data = await response.json();
    
    // Обновить показатель среднего веса
    document.getElementById('avg-weight').textContent = 
        `${data.avg_weight.toFixed(2)} кг`;
    
    // Цвет в зависимости от значения
    const avgWeightEl = document.getElementById('avg-weight');
    if (data.avg_weight < 80) {
        avgWeightEl.style.color = '#3498db';  // Синий - лёгкие
    } else if (data.avg_weight > 130) {
        avgWeightEl.style.color = '#e74c3c';  // Красный - тяжёлые
    } else {
        avgWeightEl.style.color = '#2ecc71';  // Зелёный - норма
    }
}

// Обновление каждую секунду
setInterval(fetchMetrics, 1000);
```

### WebSocket для live-обновления весов

```javascript
// Подписаться на обновления весов в реальном времени
const socket = new WebSocket('ws://localhost:8000/ws/scales');

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'weight_received') {
        console.log(`📡 Получен вес: ${data.weight} кг`);
        
        // Обновить средний вес
        updateAverageWeight(data.average_weight);
        
        // Показать анимацию
        animateWeightUpdate(data.weight);
    }
    
    if (data.type === 'average_updated') {
        updateAverageWeight(data.average);
    }
};

function animateWeightUpdate(weight) {
    const el = document.getElementById('last-weight');
    el.textContent = `Последний: ${weight} кг`;
    el.classList.add('pulse');
    setTimeout(() => el.classList.remove('pulse'), 500);
}
```

---

## 🧪 Тестирование с random весами

### Генератор случайных весов (для demo)

```python
import random
from datetime import datetime

class MockScalesGenerator:
    """Генератор random весов для тестирования"""
    
    def __init__(self, min_weight=70, max_weight=150):
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def generate_weight(self) -> dict:
        """Генерировать случайный вес"""
        weight = round(random.uniform(self.min_weight, self.max_weight), 1)
        
        return {
            'weight': weight,
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_scales'
        }
    
    def simulate_weighing_act(self, num_pigs=5) -> list:
        """Симулировать акт взвешивания (несколько животных)"""
        weights = []
        for i in range(num_pigs):
            # Добавить разброс весов (некоторые лёгче, некоторые тяжелее)
            base_weight = random.uniform(self.min_weight, self.max_weight)
            weight_data = {
                'weight': round(base_weight, 1),
                'timestamp': datetime.now().isoformat(),
                'pig_sequence': i + 1
            }
            weights.append(weight_data)
        
        return weights

# Использование
generator = MockScalesGenerator()

# Для тестирования
test_weights = generator.simulate_weighing_act(num_pigs=8)
average = sum(w['weight'] for w in test_weights) / len(test_weights)
print(f"Симулирован акт: {len(test_weights)} животных, средний вес {average:.1f} кг")
```

### Запуск в тестовом режиме

```bash
# Запустить с mock весами
python console_app.py --mode test --mock-scales

# Со своим range весов
python console_app.py --mode test --mock-scales --min-weight 60 --max-weight 160
```

---

## 📊 Excel экспорт с реальными весами

### Структура Excel с весами

```python
def export_with_weights(output_file='report.xlsx'):
    """Экспортировать в Excel с реальными весами"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Взвешивание"
    
    # Заголовки
    headers = [
        'Дата', 'Время начала', 'Кол-во свиней', 
        'Вес минимальный', 'Вес максимальный', 'Вес средний', 'Вес общий'
    ]
    ws.append(headers)
    
    # Форматирование заголовков
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    
    # Заполнение данными
    db = DatabaseManager()
    acts = db.get_weighing_acts()
    
    for act in acts:
        weights = db.get_weights_by_act(act['id'])
        weight_values = [w['weight_value'] for w in weights]
        
        row = [
            act['started_at'].date(),
            act['started_at'].time(),
            len(weight_values),
            min(weight_values) if weight_values else 0,
            max(weight_values) if weight_values else 0,
            sum(weight_values) / len(weight_values) if weight_values else 0,
            sum(weight_values)
        ]
        ws.append(row)
    
    wb.save(output_file)
    print(f"✅ Экспортировано в {output_file}")
```

---

## 🔧 Конфигурация

### .env переменные

```env
# IP ВЕСЫ
SCALES_PROTOCOL=tcp              # tcp, udp, http
SCALES_HOST=192.168.1.200        # IP адрес весов
SCALES_PORT=9000                 # Порт слушания

# Калибровка
SCALES_MIN_WEIGHT=50             # Минимум (кг)
SCALES_MAX_WEIGHT=200            # Максимум (кг)
SCALES_CALIBRATION_FACTOR=1.0    # Коэффициент калибровки

# Тестирование
USE_MOCK_SCALES=false            # Использовать mock при отсутствии реальных
MOCK_SCALES_RANGE_MIN=70         # Диапазон mock весов
MOCK_SCALES_RANGE_MAX=150
```

---

## 🔍 Отладка

### Проверка подключения к весам

```python
import socket

def test_scales_connection():
    """Проверить подключение к IP весам"""
    host = os.getenv('SCALES_HOST', 'localhost')
    port = int(os.getenv('SCALES_PORT', 9000))
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        
        if result == 0:
            print(f"✅ Весы доступны: {host}:{port}")
        else:
            print(f"❌ Весы недоступны: {host}:{port}")
        
        sock.close()
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")

test_scales_connection()
```

### Логирование весов

```python
import logging

logger = logging.getLogger(__name__)

def log_weight_received(weight: float, source: str):
    """Логировать получение веса"""
    logger.info(f"📡 Вес получен: {weight} кг от {source}")

def log_average_calculated(average: float, count: int):
    """Логировать расчёт среднего"""
    logger.info(f"🧮 Средний вес: {average} кг ({count} животных)")
```

---

## 📱 Примеры интеграции

### С системой автоматических весов (Corpwight)

```python
# Corpwight рассылает HTTP POST

from fastapi import FastAPI

app = FastAPI()

@app.post("/api/scales/corpwight")
async def handle_corpwight_weight(data: dict):
    """Интеграция с Corpwight весами"""
    weight = data.get('weight_kg')
    timestamp = data.get('timestamp')
    
    await store_weight({
        'weight': weight,
        'timestamp': timestamp,
        'source': 'corpwight_scales'
    })
    
    return {'status': 'received', 'weight': weight}
```

### С системой pneumatic scales

```python
# Pneumatic весы отправляют TCP

async def handle_pneumatic_scales():
    """Интеграция с пневматическими весами"""
    reader, writer = await asyncio.open_connection('scales_host', 9000)
    
    while True:
        data = await reader.read(1024)
        weight = float(data.decode().strip())
        
        await store_weight({
            'weight': weight,
            'timestamp': datetime.now().isoformat(),
            'source': 'pneumatic_scales'
        })
```

---

## 📞 Поддержка

Если возникают проблемы с весами:

1. Проверьте подключение: `test_scales_connection()`
2. Посмотрите логи: `tail -f logs/daemon.log | grep weight`
3. Включите debug режим: `USE_MOCK_SCALES=true` для тестирования
4. Проверьте формат данных: должен быть JSON с полем `weight`

---

**Версия:** 1.0  
**Дата:** Ноябрь 2025  
**Статус:** Production Ready ✅

