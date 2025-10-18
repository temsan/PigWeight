#!/usr/bin/env python3
"""
Создание тестового видео для демонстрации системы
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_video():
    """Создает простое тестовое видео с движущимися объектами"""
    
    # Параметры видео
    width, height = 1280, 720
    fps = 30
    duration_sec = 10
    total_frames = fps * duration_sec
    
    # Создаем папку uploads если её нет
    uploads_dir = Path('uploads')
    uploads_dir.mkdir(exist_ok=True)
    
    output_path = uploads_dir / 'test_video.mp4'
    
    # Создаем видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"🎬 Создание тестового видео: {output_path}")
    print(f"📐 Размер: {width}x{height}, FPS: {fps}, Длительность: {duration_sec}s")
    
    # Позиции линий детекции
    line_left_x = int(width * 0.25)
    line_right_x = int(width * 0.75)
    
    for frame_num in range(total_frames):
        # Создаем черный фон
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Рисуем линии детекции
        cv2.line(frame, (line_left_x, 0), (line_left_x, height), (0, 255, 0), 2)
        cv2.line(frame, (line_right_x, 0), (line_right_x, height), (0, 255, 0), 2)
        
        # Добавляем текст
        cv2.putText(frame, 'Test Video for Pig Tracking', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {frame_num}/{total_frames}', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Создаем движущиеся "свиньи" (белые эллипсы)
        time_factor = frame_num / total_frames
        
        # Свинья 1: движется слева направо
        pig1_x = int(50 + (width - 100) * time_factor)
        pig1_y = int(height * 0.3)
        cv2.ellipse(frame, (pig1_x, pig1_y), (40, 25), 0, 0, 360, (255, 255, 255), -1)
        
        # Свинья 2: движется справа налево (с задержкой)
        if frame_num > total_frames // 3:
            pig2_progress = (frame_num - total_frames // 3) / (total_frames * 2 // 3)
            pig2_x = int(width - 50 - (width - 100) * pig2_progress)
            pig2_y = int(height * 0.7)
            cv2.ellipse(frame, (pig2_x, pig2_y), (40, 25), 0, 0, 360, (255, 255, 255), -1)
        
        # Свинья 3: движется по синусоиде
        if frame_num > total_frames // 2:
            pig3_progress = (frame_num - total_frames // 2) / (total_frames // 2)
            pig3_x = int(100 + (width - 200) * pig3_progress)
            pig3_y = int(height * 0.5 + 100 * np.sin(pig3_progress * 4 * np.pi))
            cv2.ellipse(frame, (pig3_x, pig3_y), (40, 25), 0, 0, 360, (255, 255, 255), -1)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"⏳ Прогресс: {frame_num}/{total_frames} кадров")
    
    out.release()
    print(f"✅ Видео создано: {output_path}")
    print(f"📊 Размер файла: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == '__main__':
    create_test_video()
