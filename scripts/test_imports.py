#!/usr/bin/env python3
print("=== Testing imports ===")

try:
    from core.frame_broker import FRAME_BROKER
    print("✅ FRAME_BROKER imported:", type(FRAME_BROKER))
except Exception as e:
    print("❌ FRAME_BROKER error:", e)

try:
    from services.inference_worker import start_global_worker_for
    print("✅ start_global_worker_for imported:", type(start_global_worker_for))
except Exception as e:
    print("❌ start_global_worker_for error:", e)

try:
    from core.results_store import RESULTS_STORE
    print("✅ RESULTS_STORE imported:", type(RESULTS_STORE))
except Exception as e:
    print("❌ RESULTS_STORE error:", e)

print("\n=== Checking globals ===")
import sys
if 'FRAME_BROKER' in globals():
    print("✅ FRAME_BROKER in globals")
else:
    print("❌ FRAME_BROKER not in globals")

if 'FRAME_BROKER' in sys.modules:
    print("✅ FRAME_BROKER in sys.modules")
else:
    print("❌ FRAME_BROKER not in sys.modules")
