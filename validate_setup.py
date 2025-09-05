#!/usr/bin/env python3
"""
Validation script for BALANCED profile setup with CUDA 12.9
"""

import sys
import os
from pathlib import Path

def check_files():
    """Check if required files exist"""
    required_files = [
        '.env.optimized',
        'main_optimized.py',
        'start_optimized.py',
        'core/optimized_config.py',
        'Dockerfile.optimized',
        'docker-compose.optimized.yml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return missing_files

def check_config():
    """Check configuration for BALANCED profile defaults"""
    config_checks = []
    
    try:
        with open('.env.optimized', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check BALANCED profile settings
        if 'QUALITY_INITIAL_LEVEL=HIGH' in content:
            config_checks.append('✅ BALANCED profile uses HIGH quality')
        else:
            config_checks.append('❌ BALANCED profile quality not set correctly')
            
        if 'H264_HIGH_BITRATE=3000000' in content:
            config_checks.append('✅ Enhanced bitrate for BALANCED profile')
        else:
            config_checks.append('❌ BALANCED profile bitrate not enhanced')
            
        if 'CUDA_VERSION=12.9' in content:
            config_checks.append('✅ CUDA 12.9 version specified')
        else:
            config_checks.append('❌ CUDA 12.9 not specified')
            
        if 'INFERENCE_ENABLE_TENSORRT=true' in content:
            config_checks.append('✅ TensorRT enabled for CUDA 12.9')
        else:
            config_checks.append('❌ TensorRT not enabled')
            
    except Exception as e:
        config_checks.append(f'❌ Error reading config: {e}')
    
    return config_checks

def check_python_compatibility():
    """Check Python and import compatibility"""
    checks = []
    
    # Python version
    if sys.version_info >= (3, 11):
        checks.append(f'✅ Python {sys.version_info.major}.{sys.version_info.minor} (compatible)')
    else:
        checks.append(f'❌ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.11+)')
    
    # Core imports
    try:
        from core.optimized_config import OptimizedConfig, PERFORMANCE_PROFILES
        checks.append('✅ Optimized config imports successfully')
        
        # Check BALANCED profile exists
        if 'BALANCED' in PERFORMANCE_PROFILES:
            balanced = PERFORMANCE_PROFILES['BALANCED']
            if balanced.get('QUALITY_INITIAL_LEVEL') == 'HIGH':
                checks.append('✅ BALANCED profile configured correctly')
            else:
                checks.append('❌ BALANCED profile quality not HIGH')
        else:
            checks.append('❌ BALANCED profile not found')
            
    except ImportError as e:
        checks.append(f'❌ Import error: {e}')
    except Exception as e:
        checks.append(f'❌ Configuration error: {e}')
    
    return checks

def main():
    """Main validation function"""
    print("🔍 PigWeight BALANCED Profile Validation")
    print("=" * 50)
    
    # Check files
    print("\n📁 File Existence Check:")
    missing_files = check_files()
    if not missing_files:
        print("✅ All required files present")
    else:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    # Check configuration
    print("\n⚙️ Configuration Check:")
    config_checks = check_config()
    for check in config_checks:
        print(f"   {check}")
    
    # Check Python compatibility
    print("\n🐍 Python Compatibility Check:")
    python_checks = check_python_compatibility()
    for check in python_checks:
        print(f"   {check}")
    
    # Overall result
    all_checks = config_checks + python_checks
    failed_checks = [check for check in all_checks if check.startswith('❌')]
    
    print("\n" + "=" * 50)
    if not missing_files and not failed_checks:
        print("🎯 VALIDATION PASSED")
        print("✅ BALANCED profile is ready as default")
        print("✅ CUDA 12.9 optimizations enabled")
        print("\n🚀 Ready to start:")
        print("   python start_optimized.py")
        print("   # or")
        print("   start_optimized.bat")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print(f"   Missing files: {len(missing_files)}")
        print(f"   Failed checks: {len(failed_checks)}")
        print("\n🔧 Please fix issues before starting")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)