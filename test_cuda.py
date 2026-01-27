#!/usr/bin/env python
"""Test CUDA availability and diagnose issues."""

import os
import sys

print("=" * 80)
print("CUDA DIAGNOSTICS")
print("=" * 80)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"   PATH: {os.environ.get('PATH', 'NOT SET')[:100]}...")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:100]}...")

# Check if CUDA libraries exist
print("\n2. CUDA Library Check:")
cuda_home = os.environ.get('CUDA_HOME')
if cuda_home:
    lib_path = os.path.join(cuda_home, 'lib64', 'libcudart.so')
    print(f"   Looking for: {lib_path}")
    print(f"   Exists: {os.path.exists(lib_path)}")
else:
    print("   CUDA_HOME not set!")

# Try importing numba
print("\n3. Numba Import:")
try:
    from numba import cuda
    print("   ✓ Numba imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import numba: {e}")
    sys.exit(1)

# Check CUDA availability
print("\n4. CUDA Availability:")
try:
    is_available = cuda.is_available()
    print(f"   cuda.is_available(): {is_available}")
    
    if not is_available:
        # Try to get more details
        print("\n5. Detailed Error:")
        try:
            cuda.detect()
        except Exception as e:
            print(f"   Error: {e}")
            print(f"   Type: {type(e).__name__}")
    else:
        print("\n5. GPU Details:")
        device = cuda.get_current_device()
        print(f"   Device name: {device.name.decode()}")
        print(f"   Compute capability: {device.compute_capability}")
        print(f"   Total memory: {device.total_memory / 1e9:.2f} GB")
        
except Exception as e:
    print(f"   ✗ Error checking CUDA: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

