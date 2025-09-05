#!/usr/bin/env python3
"""
Quick start script for PigWeight with BALANCED profile as default
CUDA 12.9 optimized version
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='PigWeight Quick Start - BALANCED profile by default')
    parser.add_argument('--profile', default='BALANCED', 
                       choices=['ULTRA_PERFORMANCE', 'BALANCED', 'POWER_SAVING', 'MINIMAL_RESOURCE'],
                       help='Performance profile (default: BALANCED)')
    parser.add_argument('--config', default='.env.optimized', help='Configuration file')
    parser.add_argument('--validate', action='store_true', help='Validate system before start')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    
    args = parser.parse_args()
    
    print("🚀 PigWeight Quick Start - Optimized for CUDA 12.9")
    print(f"📊 Selected Profile: {args.profile}")
    
    # Check if optimized config exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file {args.config} not found")
        print("Creating default optimized configuration...")
        if not Path('.env.optimized').exists():
            print("❌ .env.optimized not found. Please run setup first.")
            sys.exit(1)
        args.config = '.env.optimized'
    
    # Install dependencies if requested
    if args.install:
        print("📦 Installing optimized dependencies for CUDA 12.9...")
        try:
            subprocess.run([
                sys.executable, "main_optimized.py", "--install"
            ], check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    
    # Validate system if requested
    if args.validate:
        print("🔍 Validating system configuration...")
        try:
            subprocess.run([
                sys.executable, "main_optimized.py", "--validate-config", 
                "--config", args.config
            ], check=True)
            print("✅ System validation passed")
        except subprocess.CalledProcessError as e:
            print(f"❌ System validation failed: {e}")
            sys.exit(1)
    
    # Start the optimized server
    print(f"🌟 Starting PigWeight server with {args.profile} profile...")
    print("🎯 CUDA 12.9 optimizations enabled")
    print("📈 Expected: 60+ FPS, 50-100ms latency")
    print("💻 Monitoring: http://localhost:8000/static/dashboard.html")
    print("📊 WebSocket metrics: ws://localhost:8765/ws/metrics")
    print()
    
    try:
        # Build command
        cmd = [sys.executable, "main_optimized.py"]
        
        if args.profile != 'BALANCED':  # BALANCED is default, no need to specify
            cmd.extend(["--profile", args.profile])
            
        cmd.extend(["--config", args.config])
        
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()