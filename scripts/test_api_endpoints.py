#!/usr/bin/env python3
"""
Test new API endpoints (PHASE 1)
"""

import asyncio
import httpx
from datetime import datetime


async def test_endpoints():
    """Test all new standard endpoints"""
    
    base_url = "http://localhost:8000"
    endpoints = [
        ("GET", "/api/health", None),
        ("GET", "/api/stats/current", None),
        ("GET", "/api/events/list", None),
        ("GET", "/api/events/stats", None),
        ("GET", "/api/config/parameters", None),
    ]
    
    print("\n" + "="*60)
    print("PHASE 1: API Endpoints Test")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        passed = 0
        for method, path, body in endpoints:
            try:
                if method == "GET":
                    resp = await client.get(f"{base_url}{path}")
                else:
                    resp = await client.post(f"{base_url}{path}", json=body)
                
                status = resp.status_code
                ok = status == 200
                passed += ok
                
                symbol = "[OK]" if ok else f"[{status}]"
                print(f"{symbol} {method:6} {path:40}")
                
                # Show first 100 chars of response
                try:
                    data = resp.json()
                    sample = str(data)[:80]
                    print(f"        Response: {sample}...")
                except:
                    pass
                
            except Exception as e:
                print(f"[ERROR] {method:6} {path:40} - {e}")
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(endpoints)} endpoints working")
    print("="*60 + "\n")
    
    return passed == len(endpoints)


if __name__ == "__main__":
    result = asyncio.run(test_endpoints())
    exit(0 if result else 1)

