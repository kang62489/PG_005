# Standard library imports
import logging

# Third-party imports
import math
import os
import warnings

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress numba performance warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# Suppress CUDA memory management info messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["NUMBA_DISABLE_CUDA"] = "0"
os.environ["NUMBA_CUDA_LOG_LEVEL"] = "40"  # WARNING level

logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
logging.getLogger("numba.cuda.cudadrv.runtime").setLevel(logging.ERROR)


@cuda.jit
def generate_normalized_gaussian_kernel(kernel_out: np.ndarray, sigma: float, size: int) -> None:
    """
    Generate and normalize 1D Gaussian kernel on GPU.

    This CUDA kernel creates a normalized 1D Gaussian filter kernel.
    Each thread computes one element, then thread 0 normalizes the entire kernel.

    Thread Index Examples (for kernel size=7):
        Thread 0: handles kernel[0] (leftmost, distance=-3 from center)
        Thread 1: handles kernel[1] (distance=-2 from center)  
        Thread 2: handles kernel[2] (distance=-1 from center)
        Thread 3: handles kernel[3] (center, distance=0, peak value)
        Thread 4: handles kernel[4] (distance=+1 from center)
        Thread 5: handles kernel[5] (distance=+2 from center)
        Thread 6: handles kernel[6] (rightmost, distance=+3 from center)

    Args:
        kernel_out: Output array to store the Gaussian kernel
        sigma: Standard deviation of the Gaussian distribution
        size: Size of the kernel (should be odd number)
    """
    # Get the thread index - each thread handles one kernel element
    # For size=7: threads 0,1,2,3,4,5,6 run simultaneously
    thread_idx = cuda.grid(1)

    # Boundary check to ensure we don't exceed kernel size
    if thread_idx >= size:
        return

    # Calculate the x-coordinate relative to kernel center
    # For a kernel of size N, center is at index N//2
    # Example: size=7, center=3, positions: [-3,-2,-1,0,1,2,3]
    kernel_center = size // 2
    distance_from_center = thread_idx - kernel_center

    # Compute Gaussian value: exp(-(x^2)/(2*sigma^2))
    # Using float32 for GPU efficiency
    kernel_out[thread_idx] = math.exp(-(distance_from_center * distance_from_center) / (2.0 * sigma * sigma))

    # Synchronize all threads before normalization
    cuda.syncthreads()

    # Let thread 0 normalize the entire kernel (simple approach for small kernels)
    if thread_idx == 0:
        # Calculate sum of all kernel elements
        kernel_sum = 0.0
        for i in range(size):
            kernel_sum += kernel_out[i]

        # Normalize each element to make total sum = 1.0
        for i in range(size):
            kernel_out[i] /= kernel_sum




@cuda.jit
def convolve_horizontal(
    input_img: np.ndarray, output_img: np.ndarray, kernel: np.ndarray, height: int, width: int, kernel_size: int
) -> None:
    """
    Perform horizontal convolution on GPU with 2D thread blocks.

    This kernel applies 1D convolution along the horizontal axis.
    Each thread processes one pixel, and 2D thread blocks are used
    for better memory coalescing and cache utilization.

    Args:
        input_img: Input 2D image array
        output_img: Output 2D image array
        kernel: 1D Gaussian kernel
        height: Image height
        width: Image width
        kernel_size: Size of the convolution kernel

    """
    # Get 2D thread coordinates
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Boundary check
    if row >= height or col >= width:
        return

    # Calculate convolution for this pixel
    result = 0.0
    kernel_half = kernel_size // 2
    weight_sum = 0.0  # For handling boundary conditions

    # Apply convolution kernel
    for kernel_pos in range(kernel_size):
        # Calculate source column index
        neighbor_col = col + kernel_pos - kernel_half

        # Handle boundary conditions by clamping to valid range
        if neighbor_col < 0:
            neighbor_col = 0
        elif neighbor_col >= width:
            neighbor_col = width - 1

        # Accumulate weighted pixel values
        pixel_value = input_img[row * width + neighbor_col]
        kernel_weight = kernel[kernel_pos]
        result += pixel_value * kernel_weight
        weight_sum += kernel_weight

    # Normalize by actual weight sum (important for boundary pixels)
    if weight_sum > 0:
        output_img[row * width + col] = result / weight_sum
    else:
        output_img[row * width + col] = input_img[row * width + col]


@cuda.jit
def convolve_vertical(
    input_img: np.ndarray, output_img: np.ndarray, kernel: np.ndarray, height: int, width: int, kernel_size: int
) -> None:
    """
    Perform vertical convolution on GPU with 2D thread blocks.

    This kernel applies 1D convolution along the vertical axis.
    Similar to horizontal convolution but operates on columns.

    Args:
        input_img: Input 2D image array (flattened)
        output_img: Output 2D image array (flattened)
        kernel: 1D Gaussian kernel
        height: Image height
        width: Image width
        kernel_size: Size of the convolution kernel

    """
    # Get 2D thread coordinates
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Boundary check
    if row >= height or col >= width:
        return

    # Calculate convolution for this pixel
    result = 0.0
    kernel_half = kernel_size // 2
    weight_sum = 0.0  # For handling boundary conditions

    # Apply convolution kernel
    for kernel_pos in range(kernel_size):
        # Calculate source row index
        neighbor_row = row + kernel_pos - kernel_half

        # Handle boundary conditions by clamping to valid range
        if neighbor_row < 0:
            neighbor_row = 0
        elif neighbor_row >= height:
            neighbor_row = height - 1

        # Accumulate weighted pixel values
        pixel_value = input_img[neighbor_row * width + col]
        kernel_weight = kernel[kernel_pos]
        result += pixel_value * kernel_weight
        weight_sum += kernel_weight

    # Normalize by actual weight sum (important for boundary pixels)
    if weight_sum > 0:
        output_img[row * width + col] = result / weight_sum
    else:
        output_img[row * width + col] = input_img[row * width + col]


def gpu_gaussian_blur(detrended_stack: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur using CUDA acceleration with optimized memory management.

    This function provides significant speedup over CPU implementation by:
    1. Using GPU parallel processing for convolution operations
    2. Implementing separable filtering (horizontal then vertical)
    3. Optimizing memory access patterns for GPU architecture
    4. Reusing GPU buffers to minimize memory allocation overhead

    Args:
        detrended_stack: Input array of shape (n_frames, height, width)
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Blurred array of same shape as input

    Performance Notes:
        - Expects input as float32 for optimal GPU performance
        - Uses 2D thread blocks (16x16) for better memory coalescing
        - Reuses GPU memory buffers across frames to reduce allocation overhead
        - Minimizes host-device memory transfers

    """
    n_frames, height, width = detrended_stack.shape

    # Ensure input is float32 for GPU efficiency
    if detrended_stack.dtype != np.float32:
        detrended_stack = detrended_stack.astype(np.float32)

    # Calculate optimal kernel size (6 sigma rule with odd size)
    kernel_size = math.ceil(sigma * 6)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Allocate GPU memory for kernel ONCE (reused across all frames)
    kernel_gpu = cuda.device_array(kernel_size, dtype=np.float32)

    # Generate Gaussian kernel on GPU
    threads_per_block = min(256, kernel_size)
    blocks_per_grid = math.ceil(kernel_size / threads_per_block)

    # Generate and normalize Gaussian kernel on GPU (combined operation)
    generate_normalized_gaussian_kernel[blocks_per_grid, threads_per_block](kernel_gpu, sigma, kernel_size)
    cuda.synchronize()

    # Allocate output array
    gaussian_stack = np.zeros_like(detrended_stack)

    # Configure 2D thread blocks for image processing
    # 16x16 blocks provide good balance of occupancy and shared memory usage
    threads_per_block_2d = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block_2d[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block_2d[1])
    blocks_per_grid_2d = (blocks_per_grid_x, blocks_per_grid_y)

    # Pre-allocate GPU buffers ONCE for all frames (major optimization)
    # This eliminates repeated allocation/deallocation overhead
    frame_size = height * width
    input_gpu = cuda.device_array(frame_size, dtype=np.float32)
    temp_gpu = cuda.device_array(frame_size, dtype=np.float32)
    output_gpu = cuda.device_array(frame_size, dtype=np.float32)

    # Pre-allocate host buffer for efficient memory transfer
    result_buffer = np.empty(frame_size, dtype=np.float32)

    # Process each frame using the same GPU buffers
    for frame_idx in range(n_frames):
        # Get current frame as contiguous array
        current_frame = np.ascontiguousarray(detrended_stack[frame_idx])

        # Copy frame data to pre-allocated GPU buffer (avoids new allocation)
        cuda.to_device(current_frame.flatten(), to=input_gpu)

        # Step 1: Apply horizontal convolution using pre-allocated buffers
        # This creates a separable filter implementation
        convolve_horizontal[blocks_per_grid_2d, threads_per_block_2d](
            input_gpu, temp_gpu, kernel_gpu, height, width, kernel_size
        )
        cuda.synchronize()

        # Step 2: Apply vertical convolution to complete the 2D Gaussian filter
        convolve_vertical[blocks_per_grid_2d, threads_per_block_2d](
            temp_gpu, output_gpu, kernel_gpu, height, width, kernel_size
        )
        cuda.synchronize()

        # Copy result back to host using pre-allocated buffer
        output_gpu.copy_to_host(result_buffer)
        gaussian_stack[frame_idx] = result_buffer.reshape(height, width)

    return gaussian_stack
