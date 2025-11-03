#!/usr/bin/env python3
from console_app import VideoSelector, HAVE_QUESTIONARY, HAVE_RICH

print("=" * 60)
print("TEST: Interactive Menu System")
print("=" * 60)

print("\nChecking flags:")
print(f"  HAVE_QUESTIONARY: {HAVE_QUESTIONARY}")
print(f"  HAVE_RICH: {HAVE_RICH}")

selector = VideoSelector()

print("\nLoading components:")
videos = selector.get_video_files()
cameras = selector.cameras
print(f"  Video files: {len(videos)}")
print(f"  Cameras: {len(cameras)}")

print("\nMethods available:")
has_q = hasattr(selector, '_select_source_questionary')
has_r = hasattr(selector, '_select_source_rich')
has_s = hasattr(selector, '_select_source_simple')
print(f"  _select_source_questionary: {has_q}")
print(f"  _select_source_rich: {has_r}")
print(f"  _select_source_simple: {has_s}")

print("\nActive menu system:")
if HAVE_QUESTIONARY:
    print("  [*] questionary (arrow keys menu) - BEST")
elif HAVE_RICH:
    print("  [*] rich (numbered menu)")
else:
    print("  [*] simple (text menu)")

print("\n" + "=" * 60)
if HAVE_QUESTIONARY and has_q and HAVE_RICH:
    print("TEST PASSED: All systems operational!")
else:
    print("TEST WARNING: Some components missing")
print("=" * 60)
