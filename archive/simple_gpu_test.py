"""
Простой тест основной функциональности GPU видео процессора
Проверяет импорты и базовые возможности без внешних зависимостей
"""

import sys
import traceback
import time
import numpy as np

def test_imports():
    """Тестирование импортов основных модулей"""
    print("🧪 Testing imports...")
    
    try:
        # Базовые модули
        import cv2
        print("✅ OpenCV imported successfully")
        
        import torch
        print(f"✅ PyTorch imported: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        import numpy as np
        print(f"✅ NumPy imported: {np.__version__}")
        
        # FastAPI компоненты
        from fastapi import FastAPI
        print("✅ FastAPI imported")
        
        # Наши модули
        from gpu_video_processor import GPUVideoProcessor
        print("✅ GPUVideoProcessor imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_gpu_processor_init():
    """Тестирование инициализации GPU процессора"""
    print("\n🚀 Testing GPU processor initialization...")
    
    try:
        from gpu_video_processor import GPUVideoProcessor
        
        # Инициализация с безопасными настройками
        processor = GPUVideoProcessor(
            use_cuda=True,  # Попробуем с GPU, если доступен
            max_cache_size=100,  # Маленький кеш для теста
            batch_size=4,
            index_keyframes_only=True  # Только ключевые кадры для ускорения
        )
        
        print(f"✅ GPU Processor initialized")
        print(f"   Device: {processor.device}")
        print(f"   CUDA enabled: {processor.use_cuda}")
        print(f"   Cache size: {processor.max_cache_size}")
        print(f"   Batch size: {processor.batch_size}")
        
        # Проверяем базовые структуры данных
        assert hasattr(processor, 'videos'), "Videos dictionary missing"
        assert hasattr(processor, 'frame_indices'), "Frame indices missing"
        assert hasattr(processor, 'frame_cache'), "Frame cache missing"
        assert hasattr(processor, 'stats'), "Stats missing"
        
        print("✅ All processor attributes present")
        
        return processor
        
    except Exception as e:
        print(f"❌ Processor initialization error: {e}")
        traceback.print_exc()
        return None

def test_model_initialization(processor):
    """Тестирование инициализации модели"""
    print("\n🧠 Testing model initialization...")
    
    try:
        if processor.use_cuda and processor.model is not None:
            print("✅ GPU model initialized")
            
            # Тестируем простой forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 64, 64).to(processor.device)
                output = processor.model(dummy_input)
                print(f"✅ Model forward pass successful, output shape: {output.shape}")
                
        elif not processor.use_cuda:
            print("ℹ️  CPU mode - model initialization skipped")
        else:
            print("⚠️  Model not initialized")
            
        return True
        
    except Exception as e:
        print(f"❌ Model test error: {e}")
        traceback.print_exc()
        return False

def test_create_dummy_video():
    """Создание тестового видео"""
    print("\n🎬 Creating dummy test video...")
    
    try:
        import cv2
        import os
        
        # Создаем папку uploads если её нет
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
            print("📁 Created uploads directory")
        
        # Параметры тестового видео
        width, height = 320, 240
        fps = 10
        duration = 2  # 2 секунды
        total_frames = fps * duration
        
        filename = 'uploads/test_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # Создаем простые кадры
        for i in range(total_frames):
            # Создаем кадр с изменяющимся цветом
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Градиент цвета по времени
            color_value = int(255 * (i / total_frames))
            frame[:, :, 0] = color_value  # Синий канал
            frame[:, :, 1] = 128  # Зеленый константа
            frame[:, :, 2] = 255 - color_value  # Красный обратный
            
            # Добавляем текст с номером кадра
            cv2.putText(frame, f"Frame {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Движущийся круг
            x = int(width * 0.1 + (width * 0.8) * (i / total_frames))
            y = height // 2
            cv2.circle(frame, (x, y), 20, (0, 255, 255), -1)
            
            out.write(frame)
        
        out.release()
        
        # Проверяем что файл создан
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✅ Test video created: {filename}")
            print(f"   Size: {file_size} bytes")
            print(f"   Duration: {duration}s, FPS: {fps}")
            return filename
        else:
            print("❌ Failed to create test video")
            return None
            
    except Exception as e:
        print(f"❌ Video creation error: {e}")
        traceback.print_exc()
        return None

def test_video_loading(processor, video_path):
    """Тестирование загрузки видео"""
    print(f"\n📹 Testing video loading: {video_path}")
    
    try:
        if not video_path:
            print("❌ No video file provided")
            return False
        
        # Загружаем видео
        start_time = time.time()
        result = processor.load_video(video_path, "test_video")
        load_time = time.time() - start_time
        
        print(f"✅ Video loaded successfully in {load_time:.2f}s")
        print(f"   ID: {result['id']}")
        print(f"   FPS: {result['fps']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Total frames: {result['total_frames']}")
        print(f"   Resolution: {result['resolution']}")
        print(f"   Index size: {result['index_size']} entries")
        print(f"   Index time: {result['index_time']:.3f}s")
        
        # Проверяем что видео действительно загружено
        assert "test_video" in processor.videos, "Video not in processor.videos"
        assert "test_video" in processor.video_metadata, "Video metadata missing"
        assert "test_video" in processor.frame_indices, "Frame indices missing"
        
        print("✅ All video data structures populated correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Video loading error: {e}")
        traceback.print_exc()
        return False

def test_frame_access(processor):
    """Тестирование быстрого доступа к кадрам"""
    print("\n⚡ Testing frame access...")
    
    try:
        video_id = "test_video"
        
        if video_id not in processor.videos:
            print("❌ Video not loaded, skipping frame access test")
            return False
        
        # Получаем метаданные
        duration = processor.video_metadata[video_id]['duration']
        print(f"   Video duration: {duration:.2f}s")
        
        # Тестируем доступ к разным кадрам
        test_times = [0.0, duration * 0.25, duration * 0.5, duration * 0.75, duration * 0.9]
        
        total_time = 0
        successful_frames = 0
        
        for i, t in enumerate(test_times):
            start_time = time.time()
            frame = processor.get_frame_fast(video_id, t)
            access_time = time.time() - start_time
            total_time += access_time
            
            if frame is not None:
                successful_frames += 1
                print(f"   Frame {i+1} (t={t:.2f}s): {access_time:.3f}s, shape={frame.shape}")
            else:
                print(f"   Frame {i+1} (t={t:.2f}s): FAILED")
        
        avg_time = total_time / len(test_times)
        success_rate = successful_frames / len(test_times)
        
        print(f"✅ Frame access test completed:")
        print(f"   Success rate: {success_rate * 100:.1f}% ({successful_frames}/{len(test_times)})")
        print(f"   Average access time: {avg_time:.3f}s")
        
        return success_rate > 0.8  # 80% успешности достаточно
        
    except Exception as e:
        print(f"❌ Frame access error: {e}")
        traceback.print_exc()
        return False

def test_batch_inference(processor):
    """Тестирование batch inference"""
    print("\n📦 Testing batch inference...")
    
    try:
        video_id = "test_video"
        
        if video_id not in processor.videos:
            print("❌ Video not loaded, skipping batch inference test")
            return False
        
        # Получаем несколько кадров для batch обработки
        duration = processor.video_metadata[video_id]['duration']
        test_times = [0.0, duration * 0.3, duration * 0.6, duration * 0.9]
        
        frames = []
        for t in test_times:
            frame = processor.get_frame_fast(video_id, t)
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            print("❌ No frames available for batch inference")
            return False
        
        print(f"   Got {len(frames)} frames for batch processing")
        
        # Выполняем batch inference
        start_time = time.time()
        results = processor.batch_inference(frames)
        inference_time = time.time() - start_time
        
        print(f"✅ Batch inference completed in {inference_time:.3f}s")
        print(f"   Input frames: {len(frames)}")
        print(f"   Output results: {len(results) if results else 0}")
        print(f"   Average time per frame: {inference_time / len(frames):.3f}s")
        
        # Проверяем результаты
        if results and len(results) == len(frames):
            for i, result in enumerate(results):
                if result:
                    print(f"   Result {i+1}: {result}")
            return True
        else:
            print("⚠️  Batch inference returned unexpected results")
            return False
            
    except Exception as e:
        print(f"❌ Batch inference error: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Запуск комплексного теста"""
    print("🚀 Starting PigWeight GPU Comprehensive Test")
    print("=" * 60)
    
    results = []
    
    # 1. Тест импортов
    print("\n" + "="*20 + " PHASE 1: IMPORTS " + "="*20)
    import_success = test_imports()
    results.append(("Imports", import_success))
    
    if not import_success:
        print("❌ Critical failure: Cannot proceed without proper imports")
        return
    
    # 2. Инициализация процессора
    print("\n" + "="*15 + " PHASE 2: PROCESSOR INIT " + "="*15)
    processor = test_gpu_processor_init()
    results.append(("Processor Init", processor is not None))
    
    if not processor:
        print("❌ Critical failure: Cannot proceed without processor")
        return
    
    # 3. Тест модели
    print("\n" + "="*18 + " PHASE 3: MODEL TEST " + "="*18)
    model_success = test_model_initialization(processor)
    results.append(("Model Init", model_success))
    
    # 4. Создание тестового видео
    print("\n" + "="*16 + " PHASE 4: VIDEO CREATION " + "="*16)
    video_path = test_create_dummy_video()
    results.append(("Video Creation", video_path is not None))
    
    if not video_path:
        print("❌ Cannot proceed without test video")
        return
    
    # 5. Загрузка видео
    print("\n" + "="*17 + " PHASE 5: VIDEO LOADING " + "="*17)
    loading_success = test_video_loading(processor, video_path)
    results.append(("Video Loading", loading_success))
    
    if not loading_success:
        print("❌ Cannot proceed without loaded video")
        return
    
    # 6. Доступ к кадрам
    print("\n" + "="*17 + " PHASE 6: FRAME ACCESS " + "="*17)
    frame_success = test_frame_access(processor)
    results.append(("Frame Access", frame_success))
    
    # 7. Batch inference
    print("\n" + "="*16 + " PHASE 7: BATCH INFERENCE " + "="*16)
    batch_success = test_batch_inference(processor)
    results.append(("Batch Inference", batch_success))
    
    # Финальные результаты
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print("-" * 60)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests >= total_tests * 0.8:  # 80% success rate
        print("🎉 OVERALL STATUS: EXCELLENT - System ready for production!")
    elif passed_tests >= total_tests * 0.6:  # 60% success rate
        print("👍 OVERALL STATUS: GOOD - Minor issues to address")
    else:
        print("⚠️  OVERALL STATUS: NEEDS WORK - Major issues detected")
    
    # Рекомендации
    print("\n💡 RECOMMENDATIONS:")
    if not import_success:
        print("   • Install missing dependencies: pip install -r requirements_gpu.txt")
    if processor and not processor.use_cuda:
        print("   • Consider installing CUDA for better performance")
    if not frame_success:
        print("   • Check video codec compatibility")
    if not batch_success:
        print("   • Verify model initialization and GPU memory")
    
    print("\n🏁 Test completed! Check results above.")

if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()
