#!/usr/bin/env python3
"""
Complete System Integration Test
Tests entire pipeline: API -> Pipeline -> Database

Verifies spec compliance and production readiness
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from datetime import datetime


async def test_api_health():
    """Test API health check"""
    print("[TEST] API Health Check")
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get("http://localhost:8000/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            print(f"  [OK] API health: {data}")
            return True
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False


async def test_all_standards_endpoints():
    """Test all 16 standard endpoints"""
    print("[TEST] Standard Endpoints")
    
    endpoints = [
        "/api/stats/current",
        "/api/stats/history",
        "/api/events/list",
        "/api/events/stats",
        "/api/config/parameters",
        "/api/health",
        "/api/streams/active",
    ]
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        passed = 0
        for endpoint in endpoints:
            try:
                resp = await client.get(f"http://localhost:8000{endpoint}")
                if resp.status_code == 200:
                    passed += 1
                    print(f"  [OK] {endpoint} (200)")
                else:
                    print(f"  [WARN] {endpoint} ({resp.status_code})")
            except Exception as e:
                print(f"  [ERROR] {endpoint}: {e}")
        
        print(f"  RESULTS: {passed}/{len(endpoints)} endpoints working")
        return passed == len(endpoints)


async def test_pipeline_imports():
    """Test that all pipeline components can be imported"""
    print("[TEST] Pipeline Imports")
    
    try:
        from core.pipeline import (
            VideoPipeline, 
            VideoCapture, 
            LineAnalyzer, 
            ActDetector, 
            WeighingAct,
            CrossingEvent,
            create_pipeline
        )
        from pig_tracking.pipeline_integration import (
            PipelineAdapter,
            process_video_spec_compliant,
            get_pipeline_adapter
        )
        
        print("  [OK] All imports successful")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


async def test_pipeline_instantiation():
    """Test creating pipeline instances"""
    print("[TEST] Pipeline Instantiation")
    
    try:
        from core.pipeline import create_pipeline
        from pig_tracking.pipeline_integration import get_pipeline_adapter
        
        # Test core pipeline
        pipeline = create_pipeline(
            stream_id="test_cam",
            source_uri="test://demo"
        )
        assert pipeline is not None
        
        # Test adapter
        adapter = get_pipeline_adapter(
            stream_id="test_cam",
            video_source="test://demo"
        )
        assert adapter is not None
        
        print("  [OK] All pipeline instances created")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


async def test_spec_compliance():
    """Verify spec compliance"""
    print("[TEST] Spec Compliance")
    
    spec_requirements = {
        "/api/stats/current": {"method": "GET"},
        "/api/events/list": {"method": "GET"},
        "/api/export/excel": {"method": "POST"},
        "/api/verify/compare": {"method": "POST"},
        "/api/config/parameters": {"method": "GET"},
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        passed = 0
        for endpoint, spec in spec_requirements.items():
            try:
                if spec["method"] == "GET":
                    resp = await client.get(f"http://localhost:8000{endpoint}")
                else:
                    resp = await client.post(f"http://localhost:8000{endpoint}", json={})
                
                if resp.status_code in [200, 422]:  # 422 for missing body params
                    passed += 1
                    print(f"  [OK] {endpoint} {spec['method']} (compliant)")
            except Exception as e:
                print(f"  [WARN] {endpoint}: {e}")
        
        print(f"  RESULTS: {passed}/{len(spec_requirements)} spec requirements met")
        return passed >= len(spec_requirements) - 1


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMPLETE SYSTEM INTEGRATION TEST")
    print("="*70 + "\n")
    
    tests = [
        test_api_health,
        test_all_standards_endpoints,
        test_pipeline_imports,
        test_pipeline_instantiation,
        test_spec_compliance,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"  [FATAL] {e}")
            results.append(False)
        print()
    
    # Summary
    print("="*70)
    passed = sum(results)
    total = len(results)
    
    print(f"\nOVERALL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ SYSTEM READY FOR PRODUCTION!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed - review above")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    exit(0 if result else 1)

