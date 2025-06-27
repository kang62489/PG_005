"""
This script pre-compiles JIT functions to avoid compilation issues.
Run this script once before running the main program.
"""
import numpy as np
from numba import jit, prange
import os

# Force Numba to recompile
os.environ['NUMBA_DISABLE_FUNCTION_CACHING'] = '1'

@jit(nopython=True, parallel=True)
def compute_spatial_averages(image_stack, roi_size):
    """
    Compute spatial averages for each ROI in parallel using Numba JIT.
    """
    n_frames, height, width = image_stack.shape
    processed_stack = np.zeros_like(image_stack, dtype=np.float32)
    
    # Process each frame in parallel
    for frame_idx in prange(n_frames):
        # Process each ROI
        for row in range(0, height - height % roi_size, roi_size):
            for col in range(0, width - width % roi_size, roi_size):
                # Calculate average value for current ROI
                roi_sum = 0.0
                for i in range(roi_size):
                    for j in range(roi_size):
                        roi_sum += image_stack[frame_idx, row + i, col + j]
                roi_avg = roi_sum / (roi_size * roi_size)
                
                # Fill ROI with average value
                for i in range(roi_size):
                    for j in range(roi_size):
                        processed_stack[frame_idx, row + i, col + j] = roi_avg
    
    return processed_stack

# Pre-compile the function with a small test array
test_data = np.random.rand(10, 64, 64).astype(np.float32)
_ = compute_spatial_averages(test_data, 32)
print("JIT functions pre-compiled successfully!")