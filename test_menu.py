#!/usr/bin/env python3
"""
Тест интерактивного меню со стрелками
"""

from console_app import VideoSelector, HAVE_QUESTIONARY, HAVE_RICH

print("=" * 60)
print("📊 ТЕСТ ИНТЕРАКТИВНОГО МЕНЮ PigWeight")
print("=" * 60)

print("\n✅ Проверка флагов:")
print(f"  HAVE_QUESTIONARY: {HAVE_QUESTIONARY} (со стрелками)")
print(f"  HAVE_RICH: {HAVE_RICH} (красивая таблица)")

print("\n✅ Инициализация селектора источников:")
selector = VideoSelector()

print("\n✅ Поиск видеофайлов:")
video_files = selector.get_video_files()
print(f"  Найдено: {len(video_files)} видеофайлов")
for i, vf in enumerate(video_files[:3], 1):
    info = selector.get_file_info(vf)
    print(f"    {i}. {vf.name} ({info['size']} • {info['duration']})")

print("\n✅ Поиск камер:")
cameras = selector.cameras
print(f"  Найдено: {len(cameras)} камер(ы)")
for cam_id, cam_info in cameras.items():
    print(f"    - {cam_info['name']}: {cam_info['url']}")

print("\n✅ Проверка методов:")
print(f"  _select_source_questionary: {'есть' if hasattr(selector, '_select_source_questionary') else 'нет'}")
print(f"  _select_source_rich: {'есть' if hasattr(selector, '_select_source_rich') else 'нет'}")
print(f"  _select_source_simple: {'есть' if hasattr(selector, '_select_source_simple') else 'нет'}")

print("\n🎯 Путь выполнения:")
if HAVE_QUESTIONARY:
    print("  ➜ questionary (интерактивное меню со стрелками) ⭐")
elif HAVE_RICH:
    print("  ➜ Rich (таблица с вводом номера)")
else:
    print("  ➜ Simple (простой текстовый список)")

print("\n" + "=" * 60)
print("✅ Тест завершен успешно!")
print("=" * 60)

print("\n💡 Для тестирования интерактивного меню запустите:")
print("   python console_app.py")
