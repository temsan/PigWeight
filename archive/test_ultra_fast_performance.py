#!/usr/bin/env python3
"""
Ultra-Fast Video System Performance Test
Tests the responsiveness and speed of the ultra-fast video processing system.
"""

import asyncio
import websockets
import requests
import time
import json
from pathlib import Path

class UltraFastPerformanceTest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        
    async def test_websocket_latency(self):
        """Test WebSocket connection latency and frame rate"""
        print("🚀 Testing WebSocket Video Stream Latency...")
        
        try:
            uri = f"{self.ws_url}/ws/video_ultra_fast"
            
            frame_count = 0
            start_time = time.time()
            latencies = []
            
            async with websockets.connect(uri) as websocket:
                print("✅ WebSocket connected successfully")
                
                # Receive frames for 10 seconds
                timeout_time = start_time + 10
                
                while time.time() < timeout_time:
                    try:
                        # Measure frame arrival time
                        frame_start = time.time()
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        frame_end = time.time()
                        
                        if isinstance(message, bytes):
                            frame_count += 1
                            latency_ms = (frame_end - frame_start) * 1000
                            latencies.append(latency_ms)
                            
                            if frame_count % 10 == 0:  # Print every 10th frame
                                print(f"Frame {frame_count}: {len(message)} bytes, {latency_ms:.1f}ms")
                    
                    except asyncio.TimeoutError:
                        print("⚠️  Frame timeout (>2s)")
                        break
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            min_latency = min(latencies) if latencies else 0
            
            print(f"\n📊 WebSocket Performance Results:")
            print(f"   • Frames received: {frame_count}")
            print(f"   • Average FPS: {avg_fps:.1f}")
            print(f"   • Average latency: {avg_latency:.1f}ms")
            print(f"   • Min latency: {min_latency:.1f}ms")
            print(f"   • Max latency: {max_latency:.1f}ms")
            
            return {
                'frames': frame_count,
                'fps': avg_fps,
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max_latency
            }
            
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            return None
    
    def test_api_responsiveness(self):
        """Test API endpoint response times"""
        print("\n🔥 Testing API Endpoint Response Times...")
        
        endpoints = [
            ("/api/health", "Health Check"),
            ("/api/cameras", "Camera List"),
        ]
        
        results = {}
        
        for endpoint, name in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    print(f"   ✅ {name}: {response_time_ms:.1f}ms")
                    results[endpoint] = response_time_ms
                else:
                    print(f"   ❌ {name}: HTTP {response.status_code}")
                    results[endpoint] = None
                    
            except Exception as e:
                print(f"   ❌ {name}: {e}")
                results[endpoint] = None
        
        return results
    
    def test_video_frame_seeking(self):
        """Test video frame seeking performance (requires a video file)"""
        print("\n⚡ Testing Video Frame Seeking Performance...")
        
        # First check if there are any active video sessions
        try:
            # Try to get a frame directly - this will work if system is running
            seek_times = []
            
            for t in [0, 5, 10, 15, 20]:  # Test different time positions
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{self.base_url}/api/video_file/frame_ultra_fast",
                        params={'id': 'test_session', 't': t},
                        timeout=2
                    )
                    end_time = time.time()
                    
                    seek_time_ms = (end_time - start_time) * 1000
                    seek_times.append(seek_time_ms)
                    
                    if response.status_code == 200:
                        # Check for performance headers
                        seek_header = response.headers.get('X-Seek-Ms', 'N/A')
                        infer_header = response.headers.get('X-Inference-Ms', 'N/A')
                        encode_header = response.headers.get('X-Encode-Ms', 'N/A')
                        
                        print(f"   ✅ Seek to {t}s: {seek_time_ms:.1f}ms total")
                        print(f"      └─ Seek: {seek_header}ms, Infer: {infer_header}ms, Encode: {encode_header}ms")
                    else:
                        print(f"   ⚠️  Seek to {t}s: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ Seek to {t}s: {e}")
            
            if seek_times:
                avg_seek = sum(seek_times) / len(seek_times)
                max_seek = max(seek_times)
                min_seek = min(seek_times)
                
                print(f"\n📈 Seeking Performance:")
                print(f"   • Average seek time: {avg_seek:.1f}ms")
                print(f"   • Fastest seek: {min_seek:.1f}ms")
                print(f"   • Slowest seek: {max_seek:.1f}ms")
                
                return {
                    'avg_seek_ms': avg_seek,
                    'min_seek_ms': min_seek,
                    'max_seek_ms': max_seek
                }
            
        except Exception as e:
            print(f"   ℹ️  Video seeking test skipped: {e}")
        
        return None
    
    def print_system_status(self):
        """Print overall system status"""
        print("\n🎯 Ultra-Fast System Status Check...")
        
        try:
            # Check server health
            response = requests.get(f"{self.base_url}/api/health", timeout=3)
            if response.status_code == 200:
                print("   ✅ Server is running and healthy")
            else:
                print(f"   ⚠️  Server health check returned: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Server appears to be down: {e}")
            return False
        
        return True
    
    async def run_full_test(self):
        """Run complete performance test suite"""
        print("🎮 Ultra-Fast Video System Performance Test")
        print("=" * 50)
        
        # Check if system is running
        if not self.print_system_status():
            print("❌ Cannot run tests - server is not responding")
            return
        
        # Test API responsiveness
        api_results = self.test_api_responsiveness()
        
        # Test WebSocket video streaming
        ws_results = await self.test_websocket_latency()
        
        # Test video seeking performance
        seek_results = self.test_video_frame_seeking()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📋 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        if ws_results:
            fps = ws_results.get('fps', 0)
            latency = ws_results.get('avg_latency_ms', 0)
            
            if fps >= 25:
                print(f"✅ Video streaming: {fps:.1f} FPS (EXCELLENT)")
            elif fps >= 15:
                print(f"⚠️  Video streaming: {fps:.1f} FPS (GOOD)")
            else:
                print(f"❌ Video streaming: {fps:.1f} FPS (NEEDS IMPROVEMENT)")
            
            if latency <= 50:
                print(f"✅ Frame latency: {latency:.1f}ms (ULTRA-FAST)")
            elif latency <= 100:
                print(f"⚠️  Frame latency: {latency:.1f}ms (FAST)")
            else:
                print(f"❌ Frame latency: {latency:.1f}ms (SLOW)")
        
        if seek_results:
            avg_seek = seek_results.get('avg_seek_ms', 0)
            if avg_seek <= 10:
                print(f"✅ Seeking speed: {avg_seek:.1f}ms (INSTANT)")
            elif avg_seek <= 50:
                print(f"⚠️  Seeking speed: {avg_seek:.1f}ms (VERY FAST)")
            else:
                print(f"❌ Seeking speed: {avg_seek:.1f}ms (SLOW)")
        
        print("\n🎉 Performance test completed!")

if __name__ == "__main__":
    test = UltraFastPerformanceTest()
    asyncio.run(test.run_full_test())
