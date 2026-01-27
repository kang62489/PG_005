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
cuda_home = os.environ.get("CUDA_HOME")
if cuda_home:
    print(f"   CUDA_HOME: {cuda_home}")
    print(f"   CUDA_HOME exists: {os.path.exists(cuda_home)}")

    # Check for lib64 directory
    lib64_path = os.path.join(cuda_home, "lib64")
    print(f"   lib64 exists: {os.path.exists(lib64_path)}")

    if os.path.exists(lib64_path):
        # List some key libraries
        libs_to_check = ["libcudart.so", "libcudart.so.11.0", "libcuda.so", "libcublas.so"]
        for lib in libs_to_check:
            lib_path = os.path.join(lib64_path, lib)
            exists = os.path.exists(lib_path)
            print(f"   {lib}: {exists}")

        # List all .so files
        try:
            import glob

            so_files = glob.glob(os.path.join(lib64_path, "*.so*"))
            print(f"   Total .so files in lib64: {len(so_files)}")
            if len(so_files) > 0:
                print(f"   First few: {[os.path.basename(f) for f in so_files[:5]]}")
        except Exception as e:
            print(f"   Error listing files: {e}")
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
