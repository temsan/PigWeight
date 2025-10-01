#!/usr/bin/env python3
"""
Быстрый тест производительности профилей для PigWeight
"""

import os
import time
import psutil
from core.config import PERFORMANCE_PROFILES, detect_optimal_runtime, apply_runtime_optimizations

def test_gpu_memory():
    """Проверяет доступную память GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_free = gpu_memory - torch.cuda.memory_allocated(0)
            return gpu_memory // (1024**3), gpu_free // (1024**3)
    except:
        pass
    return 0, 0

def benchmark_profile(profile_name):
    """Тестирует производительность профиля"""
    settings = PERFORMANCE_PROFILES[profile_name]
    
    print(f"\n🧪 Тестирование профиля: {profile_name}")
    print("─" * 50)
    
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    # Симуляция нагрузки
    target_fps = int(settings.get('TARGET_FPS', '25'))
    batch_size = int(settings.get('BATCH_MAX_SIZE', '4'))
    
    print(f"\n📊 Расчетная производительность:")
    print(f"  • Целевой FPS: {target_fps}")
    print(f"  • Размер батча: {batch_size}")
    print(f"  • Пропускная способность: ~{target_fps * batch_size} объектов/сек")
    
    return target_fps * batch_size

def main():
    print("🐷 PigWeight - Анализ производительности профилей")
    print("=" * 60)
    
    # Информация о системе
    print("\n💻 Информация о системе:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram, free_vram = test_gpu_memory()
            print(f"  🔥 GPU: {gpu_name}")
            print(f"  💾 VRAM: {total_vram}GB общая, {free_vram}GB свободна")
        else:
            print("  💻 Только CPU (CUDA недоступен)")
    except:
        print("  ⚠️ PyTorch недоступен")
    
    cpu_count = psutil.cpu_count(logical=False)
    memory_gb = psutil.virtual_memory().total // (1024**3)
    print(f"  🧮 CPU: {cpu_count} ядер")
    print(f"  🧠 RAM: {memory_gb}GB")
    
    # Автоматический выбор
    print("\n🤖 Автоматический выбор:")
    runtime_info = detect_optimal_runtime()
    print(f"  ✅ Рекомендуемый профиль: {runtime_info['profile']}")
    print(f"  🎯 Рантайм: {runtime_info['runtime']}")
    print(f"  📋 Причина: {runtime_info['reasons'][0] if runtime_info['reasons'] else 'Не указана'}")
    
    # Тестирование всех профилей
    print("\n🏁 Сравнение профилей:")
    results = {}
    
    for profile_name in PERFORMANCE_PROFILES.keys():
        score = benchmark_profile(profile_name)
        results[profile_name] = score
    
    # Рейтинг по производительности
    print("\n🏆 Рейтинг по производительности:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for i, (profile, score) in enumerate(sorted_results, 1):
        marker = "👑" if profile == runtime_info['profile'] else f"{i}."
        recommended = " (РЕКОМЕНДУЕТСЯ)" if profile == runtime_info['profile'] else ""
        print(f"  {marker} {profile}: {score} объектов/сек{recommended}")
    
    print(f"\n💡 Рекомендация: Используйте профиль {runtime_info['profile']} для оптимальной работы на вашем железе!")
    print(f"📝 Запуск: python main_optimized.py --profile {runtime_info['profile']}")

if __name__ == '__main__':
    main()