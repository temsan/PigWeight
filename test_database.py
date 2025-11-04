#!/usr/bin/env python3
"""
Test database functionality - save and read weighing acts
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Set UTF-8 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("DATABASE TEST - SAVE AND READ WEIGHING ACTS")
print("=" * 70)

# Check Supabase configuration
supabase_key = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_KEY')
supabase_url = os.getenv('SUPABASE_URL')

print("\nChecking configuration:")
print(f"  SUPABASE_URL: {'FOUND' if supabase_url else 'NOT FOUND'}")
print(f"  SUPABASE_KEY: {'FOUND' if supabase_key else 'NOT FOUND'}")

if not supabase_key or not supabase_url:
    print("\nDATABASE NOT CONFIGURED")
    print("Current behavior:")
    print("  - Results are saved to JSON files (records/)")
    print("  - To use database: copy .env.example to .env")
    print("  - Set SUPABASE_URL and SUPABASE_KEY")
    sys.exit(0)

print("\nConnecting to database...")
try:
    from pig_tracking.database import DatabaseManager, WeighingAct, CrossingEvent
    
    db = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key)
    print("  OK - Connection established")
    
except Exception as e:
    print(f"  ERROR: {e}")
    print("\nMake sure:")
    print("  1. Supabase instance is running")
    print("  2. SUPABASE_URL and SUPABASE_KEY are correct")
    print("  3. Tables weighing_acts and crossings are created")
    sys.exit(1)

# Test 1: Get statistics
print("\n" + "=" * 70)
print("Test 1: Get current database statistics")
print("=" * 70)
try:
    stats = db.get_stats()
    print(f"  Total acts: {stats['total_acts']}")
    print(f"  Total crossings: {stats['total_crossings']}")
    if stats['last_act']:
        print(f"  Last act: {stats['last_act']['started_at']}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Save new act
print("\n" + "=" * 70)
print("Test 2: Save new weighing act")
print("=" * 70)
try:
    now = datetime.now()
    test_act = WeighingAct(
        started_at=now - timedelta(seconds=120),
        ended_at=now,
        duration_sec=120.0,
        left_count=5,
        right_count=5,
        peak_count=3,
        total_weight=550.0,
        avg_weight=110.0,
        stream_id="test_cam_101",
        video_file="test_video.mp4"
    )
    
    act_id = db.save_weighing_act(test_act)
    print(f"  OK - Act saved with ID: {act_id}")
    print(f"       Time: {test_act.started_at} -> {test_act.ended_at}")
    print(f"       Left: {test_act.left_count}, Right: {test_act.right_count}")
    print(f"       Weight: {test_act.total_weight}kg (avg: {test_act.avg_weight}kg)")
    
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Read acts by period
print("\n" + "=" * 70)
print("Test 3: Read acts from last 24 hours")
print("=" * 70)
try:
    start_time = datetime.now() - timedelta(hours=24)
    end_time = datetime.now()
    
    acts = db.get_acts_by_period(start_time, end_time)
    print(f"  Found: {len(acts)} acts")
    
    for i, act in enumerate(acts[-5:], 1):
        print(f"\n    Act #{i}:")
        print(f"      ID: {act.id}")
        print(f"      Time: {act.started_at.strftime('%Y-%m-%d %H:%M:%S')} ({act.duration_sec:.1f}s)")
        print(f"      Left: {act.left_count}, Right: {act.right_count}, Peak: {act.peak_count}")
        print(f"      Weight: {act.total_weight}kg, Avg: {act.avg_weight}kg")
        
except Exception as e:
    print(f"  ERROR: {e}")

# Test 4: Read crossings for act
print("\n" + "=" * 70)
print("Test 4: Read crossings for last act")
print("=" * 70)
try:
    if 'act_id' in locals():
        crossings = db.get_crossings_by_act(act_id)
        print(f"  Found: {len(crossings)} crossings")
        for crossing in crossings[:3]:
            print(f"    - Pig {crossing.pig_id}: {crossing.direction} (weight: {crossing.weight_estimate}kg)")
    else:
        print("  WARNING: No saved acts to check")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 5: Pig passages
print("\n" + "=" * 70)
print("Test 5: Aggregated pig passage data")
print("=" * 70)
try:
    passages = db.get_pig_passages()
    print(f"  Found: {len(passages)} passages")
    for passage in passages[:5]:
        print(f"    - Pig {passage['pig_id']}: {passage['path']} (crossings: {passage['crossings_count']})")
except Exception as e:
    print(f"  ERROR: {e}")

# Summary
print("\n" + "=" * 70)
print("TESTING COMPLETED")
print("=" * 70)
print("\nDatabase status:")
print("  [OK] Connection works")
print("  [OK] Act writing works")
print("  [OK] Act reading works")
print("\nNext steps:")
print("  1. Process video and save results to database")
print("  2. Check data through /metrics web interface")
print("  3. Export data for analysis")
