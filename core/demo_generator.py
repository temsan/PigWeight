"""
Генератор демо видео для тестирования
"""

import cv2
import numpy as np
import time
import random
from typing import Generator, Tuple
import logging

logger = logging.getLogger(__name__)

class DemoVideoGenerator:
    """Генератор демо видео с движущимися объектами"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 24):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        
        # Параметры движущихся объектов
        self.objects = []
        for i in range(3):  # 3 демо "свиньи"
            self.objects.append({
                'x': random.randint(50, width - 50),
                'y': random.randint(50, height - 50),
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-2, 2),
                'size': random.randint(20, 40),
                'color': (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            })
    
    def generate_frame(self) -> np.ndarray:
        """Генерация одного кадра"""
        # Создаем фон
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
        
        # Добавляем "шум" фона
        noise = np.random.randint(0, 30, (self.height, self.width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Обновляем и рисуем объекты
        for obj in self.objects:
            # Обновляем позицию
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # Отскок от границ
            if obj['x'] <= obj['size'] or obj['x'] >= self.width - obj['size']:
                obj['vx'] *= -1
            if obj['y'] <= obj['size'] or obj['y'] >= self.height - obj['size']:
                obj['vy'] *= -1
            
            # Ограничиваем позицию
            obj['x'] = max(obj['size'], min(self.width - obj['size'], obj['x']))
            obj['y'] = max(obj['size'], min(self.height - obj['size'], obj['y']))
            
            # Рисуем объект
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), obj['size'], obj['color'], -1)
            
            # Добавляем "ноги" для имитации свиньи
            leg_offset = obj['size'] // 2
            cv2.circle(frame, (int(obj['x'] - leg_offset//2), int(obj['y'] + leg_offset)), 
                      obj['size']//4, obj['color'], -1)
            cv2.circle(frame, (int(obj['x'] + leg_offset//2), int(obj['y'] + leg_offset)), 
                      obj['size']//4, obj['color'], -1)
        
        # Добавляем таймстамп
        timestamp = f"Demo Frame {self.frame_count} - {time.strftime('%H:%M:%S')}"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Добавляем счетчик объектов
        count_text = f"Objects: {len(self.objects)}"
        cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        self.frame_count += 1
        return frame
    
    def generate_frames(self) -> Generator[np.ndarray, None, None]:
        """Генератор кадров"""
        while True:
            yield self.generate_frame()
            time.sleep(1.0 / self.fps)

def create_demo_stream() -> DemoVideoGenerator:
    """Создание демо потока"""
    logger.info("🎬 Creating demo video stream")
    return DemoVideoGenerator()