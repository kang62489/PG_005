#!/usr/bin/env python
"""Quick test to verify all imports work in headless mode."""
import os
import sys

# Set headless mode to prevent Qt initialization issues
os.environ["QT_QPA_PLATFORM"] = "offscreen"

print("Testing imports for headless/batch mode...")
print()

success = True

try:
    print("1. Testing numba and CUDA...")
    from numba import cuda
    print(f"   ✓ numba imported")
    print(f"   CUDA available: {cuda.is_available()}")
    print()
except Exception as e:
    print(f"   ✗ Failed: {e}")
    success = False
    print()

try:
    print("2. Testing numpy, scipy, matplotlib...")
    import numpy as np
    import scipy
    import matplotlib
    print(f"   ✓ numpy {np.__version__}")
    print(f"   ✓ scipy {scipy.__version__}")
    print(f"   ✓ matplotlib {matplotlib.__version__}")
    print()
except Exception as e:
    print(f"   ✗ Failed: {e}")
    success = False
    print()

try:
    print("3. Testing classes imports (without Qt)...")
    from classes import AbfClip, RegionAnalyzer, ResultsExporter, SpatialCategorizer
    print("   ✓ Core classes imported successfully")
    print()
except Exception as e:
    print(f"   ✗ Failed: {e}")
    success = False
    print()

try:
    print("4. Testing batch_process imports...")
    from batch_process import analyze_pair, get_picked_pairs, preprocess_single, HAS_QT
    print("   ✓ batch_process imported successfully")
    print(f"   Qt available: {HAS_QT}")
    print()
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    success = False
    print()

print("=" * 60)
if success:
    print("✓ All critical imports successful!")
    print("You can now submit the batch job with sbatch")
    sys.exit(0)
else:
    print("✗ Some imports failed - check errors above")
    sys.exit(1)
print("=" * 60)
