#!/usr/bin/env python3
"""
Quick status check for Ultra-Fast Video System
"""
import requests
import time

def quick_status_check():
    print("🔍 Quick Ultra-Fast System Status Check")
    print("-" * 40)
    
    base_url = "http://localhost:8000"
    
    # Test basic connectivity with very short timeout
    try:
        print("Testing basic server connectivity...")
        start = time.time()
        response = requests.get(f"{base_url}/", timeout=1)
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            print(f"✅ Server responding: {response.status_code} ({elapsed:.0f}ms)")
            print(f"   Page title: {'PigWeight Ultra Fast' if 'Ultra Fast' in response.text else 'Unknown'}")
        else:
            print(f"⚠️  Server returned: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("❌ Server timeout (>1s) - may be under heavy load")
        print("   This is normal during heavy video processing")
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    # Check if processes are running
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, timeout=5)
        python_processes = result.stdout.count('python.exe')
        print(f"🐍 Python processes running: {python_processes}")
        
        if python_processes >= 2:  # Main + worker processes
            print("✅ Multiple Python processes detected (main + workers)")
        else:
            print("⚠️  Expected more Python processes for full system")
            
    except Exception as e:
        print(f"❌ Process check failed: {e}")
    
    print("\n💡 System Status:")
    print("   • If server is responding slowly, it's processing video frames")
    print("   • RTSP timeouts are normal - system falls back gracefully") 
    print("   • WebSocket connections handle the real-time video stream")
    print("   • You can access the system at: http://localhost:8000")
    
if __name__ == "__main__":
    quick_status_check()
