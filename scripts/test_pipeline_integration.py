#!/usr/bin/env python3
"""
Test script for VideoPipeline integration (PHASE 3)
Verifies that all components work together
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import (
    create_pipeline,
    VideoPipeline,
    WeighingAct,
    CrossingEvent,
    CrossingDirection
)
from datetime import datetime


def test_pipeline_creation():
    """Test that pipeline can be created"""
    print("[TEST] Pipeline Creation")
    
    try:
        pipeline = create_pipeline(
            stream_id="cam101",
            source_uri="test://demo",
            model_path="models/pig_yolo11-seg.pt"
        )
        
        assert pipeline is not None
        assert pipeline.stream_id == "cam101"
        assert pipeline.source_uri == "test://demo"
        
        print("  [OK] Pipeline created successfully")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_weighing_act_creation():
    """Test WeighingAct data model"""
    print("[TEST] WeighingAct Data Model")
    
    try:
        now = datetime.now()
        
        # Create act
        act = WeighingAct(
            started_at=now,
            ended_at=now,
            duration_sec=10.0,
            left_count=3,
            right_count=2,
            peak_count=5
        )
        
        # Add crossings
        for i in range(5):
            crossing = CrossingEvent(
                pig_id=i,
                direction=CrossingDirection.LEFT if i < 3 else CrossingDirection.RIGHT,
                timestamp=now,
                line_x=0.25 if i < 3 else 0.75,
                line_y=0.5,
                weight_estimate=80.0 + i * 5
            )
            act.crossings.append(crossing)
        
        # Verify calculations
        total = act.get_total_weight()
        avg = act.get_avg_weight()
        
        assert total > 0, "Total weight should be > 0"
        assert avg > 0, "Average weight should be > 0"
        assert len(act.crossings) == 5, "Should have 5 crossings"
        
        print(f"  [OK] Act created: {len(act.crossings)} crossings, "
              f"total={total:.1f}, avg={avg:.1f}")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_pipeline_statistics():
    """Test pipeline statistics collection"""
    print("[TEST] Pipeline Statistics")
    
    try:
        pipeline = create_pipeline(
            stream_id="cam102",
            source_uri="test://demo"
        )
        
        # Add some acts
        act1 = WeighingAct(
            started_at=datetime.now(),
            ended_at=datetime.now(),
            duration_sec=5.0,
            left_count=2,
            right_count=1,
            peak_count=3
        )
        pipeline.detected_acts.append(act1)
        
        # Get stats
        stats = pipeline.get_statistics()
        
        assert stats["stream_id"] == "cam102"
        assert stats["detected_acts"] == 1
        assert "processed_frames" in stats
        
        print(f"  [OK] Statistics: {stats['detected_acts']} acts detected")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_imports():
    """Test that all components can be imported"""
    print("[TEST] Module Imports")
    
    try:
        from core import (
            VideoPipeline,
            VideoCapture,
            LineAnalyzer,
            ActDetector,
            WeighingAct,
            CrossingEvent,
            create_pipeline
        )
        
        print("  [OK] All components imported successfully")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 3: VideoPipeline Integration Tests")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_pipeline_creation,
        test_weighing_act_creation,
        test_pipeline_statistics,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [FATAL] {e}")
            results.append(False)
        print()
    
    # Summary
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    if passed == total:
        print("[SUCCESS] PHASE 3 integration verified!")
        return 0
    else:
        print("[FAILED] Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())

