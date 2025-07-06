# Standard library imports
import logging

# Third-party imports
import math
import os
import warnings

import numpy as np
from numba import cuda, jit, prange
from numba.core.errors import NumbaPerformanceWarning

# Suppress numba performance warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# Suppress CUDA memory management info messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["NUMBA_DISABLE_CUDA"] = "0"
os.environ["NUMBA_CUDA_LOG_LEVEL"] = "40"  # WARNING level

logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
logging.getLogger("numba.cuda.cudadrv.runtime").setLevel(logging.ERROR)


@jit(nopython=True, parallel=True)
def gaussian_kernel_1d(sigma: float, size: int | None) -> np.ndarray:
    """
    Create a 1D Gaussian kernel array.

    Args:
        sigma: Standard deviation of the Gaussian distribution
        size: Size of the kernel (should be odd number)

    """
    if size is None:
        size = math.ceil(sigma * 6)
        if size % 2 == 0:
            size += 1

    x = np.arange(-(size // 2), (size // 2) + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # Normaize to make sum of kernel = 1


@jit(nopython=True)
def convolve_1d(arr: np.ndarray, kernel: np.ndarray, axis: int, mode: str = "valid") -> np.ndarray:
    """
    Apply 1D convolution along specified axis.

    Args:
        arr: Input 2D array
        kernel: 1D convolution kernel
        axis: Axis along which to apply convolution (0=vertical, 1=horizontal)
        mode: Convolution mode ('valid' or 'same')

    Returns:
        Convolved array

    """
    result = np.zeros_like(arr)
    kernel_size = len(kernel)
    kernel_half = kernel_size // 2

    if axis == 0:  # Vertical convolution
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = 0.0
                if mode == "same":
                    # For 'same' mode, normalize by the sum of weights actually used
                    weight_sum = 0.0
                    for k in range(kernel_size):
                        idx = i + k - kernel_half
                        if 0 <= idx < arr.shape[0]:
                            val += arr[idx, j] * kernel[k]
                            weight_sum += kernel[k]

                    # Normalize by the sum of weights that were actually used
                    if weight_sum > 0:
                        result[i, j] = val / weight_sum
                    else:
                        result[i, j] = arr[i, j]  # Fallback if no kernel weights were applied
                else:
                    # Original 'valid' mode behavior
                    for k in range(kernel_size):
                        idx = i + k - kernel_half
                        if 0 <= idx < arr.shape[0]:
                            val += arr[idx, j] * kernel[k]
                    result[i, j] = val
    else:  # axis == 1, Horizontal convolution
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = 0.0
                if mode == "same":
                    # For 'same' mode, normalize by the sum of weights actually used
                    weight_sum = 0.0
                    for k in range(kernel_size):
                        idx = j + k - kernel_half
                        if 0 <= idx < arr.shape[1]:
                            val += arr[i, idx] * kernel[k]
                            weight_sum += kernel[k]

                    # Normalize by the sum of weights that were actually used
                    if weight_sum > 0:
                        result[i, j] = val / weight_sum
                    else:
                        result[i, j] = arr[i, j]  # Fallback if no kernel weights were applied
                else:
                    # Original 'valid' mode behavior
                    for k in range(kernel_size):
                        idx = j + k - kernel_half
                        if 0 <= idx < arr.shape[1]:
                            val += arr[i, idx] * kernel[k]
                    result[i, j] = val

    return result


@jit(nopython=True, parallel=True)
def gaussian_blur(detrended_stack: np.ndarray, sigma: float, mode: str = "same") -> np.ndarray:
    """
    Apply Gaussian blur to detrended stack using separable convolution.

    Args:
        detrended_stack: Input array of shape (n_frames, height, width)
        sigma: Standard deviation for Gaussian kernel
        mode: Convolution mode ('valid' or 'same')

    Returns:
        Blurred array of same shape as input

    """
    n_frames, height, width = detrended_stack.shape
    gaussian_stack = np.zeros_like(detrended_stack)

    # Create 1D Gaussian kernel
    kernel = gaussian_kernel_1d(sigma)

    # Process each frame in parallel
    for i in prange(n_frames):
        # Apply horizontal convolution
        temp = convolve_1d(detrended_stack[i], kernel, axis=1, mode=mode)
        # Apply vertical convolution
        gaussian_stack[i] = convolve_1d(temp, kernel, axis=0, mode=mode)

    return gaussian_stack


@cuda.jit
def gaussian_kernel_1d_cuda(kernel_out: np.ndarray, sigma: float, size: int) -> None:
    """
    Generate 1D Gaussian kernel on GPU.

    This CUDA kernel creates a normalized 1D Gaussian filter kernel.
    Each thread computes one element of the kernel array.

    Args:
        kernel_out: Output array to store the Gaussian kernel
        sigma: Standard deviation of the Gaussian distribution
        size: Size of the kernel (should be odd number)

    """
    # Get the thread index - each thread handles one kernel element
    idx = cuda.grid(1)

    # Boundary check to ensure we don't exceed kernel size
    if idx >= size:
        return

    # Calculate the x-coordinate relative to kernel center
    # For a kernel of size N, center is at index N//2
    center = size // 2
    x = idx - center

    # Compute Gaussian value: exp(-(x^2)/(2*sigma^2))
    # Using float32 for GPU efficiency
    kernel_out[idx] = math.exp(-(x * x) / (2.0 * sigma * sigma))


@cuda.jit
def normalize_kernel_cuda(kernel: np.ndarray, size: int) -> None:
    """
    Normalize Gaussian kernel so sum equals 1.0.

    This kernel performs parallel reduction to compute the sum,
    then normalizes each element by dividing by the total sum.

    Args:
        kernel: Input/output kernel array to normalize
        size: Size of the kernel array

    """
    # Use shared memory for efficient parallel reduction
    # Allocate shared memory for partial sums
    shared_data = cuda.shared.array(256, dtype=np.float32)

    # Get thread and block information
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)

    # Load data into shared memory (or 0 if out of bounds)
    if idx < size:
        shared_data[tid] = kernel[idx]
    else:
        shared_data[tid] = 0.0

    # Synchronize threads before reduction
    cuda.syncthreads()

    # Perform parallel reduction in shared memory
    # This efficiently computes the sum of all kernel elements
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride and tid + stride < cuda.blockDim.x:
            shared_data[tid] += shared_data[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Thread 0 stores the block's partial sum
    if tid == 0:
        # Use atomic add to accumulate partial sums from all blocks
        cuda.atomic.add(kernel, size, shared_data[0])  # Store sum at end of array

    # Synchronize all threads before normalization
    cuda.syncthreads()

    # Normalize each element by the total sum
    if idx < size:
        total_sum = kernel[size]  # Sum stored at index 'size'
        if total_sum > 0:
            kernel[idx] /= total_sum


@cuda.jit
def convolve_horizontal_cuda(
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
    for k in range(kernel_size):
        # Calculate source column index
        src_col = col + k - kernel_half

        # Handle boundary conditions by clamping to valid range
        if src_col < 0:
            src_col = 0
        elif src_col >= width:
            src_col = width - 1

        # Accumulate weighted pixel values
        pixel_value = input_img[row * width + src_col]
        kernel_weight = kernel[k]
        result += pixel_value * kernel_weight
        weight_sum += kernel_weight

    # Normalize by actual weight sum (important for boundary pixels)
    if weight_sum > 0:
        output_img[row * width + col] = result / weight_sum
    else:
        output_img[row * width + col] = input_img[row * width + col]


@cuda.jit
def convolve_vertical_cuda(
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
    for k in range(kernel_size):
        # Calculate source row index
        src_row = row + k - kernel_half

        # Handle boundary conditions by clamping to valid range
        if src_row < 0:
            src_row = 0
        elif src_row >= height:
            src_row = height - 1

        # Accumulate weighted pixel values
        pixel_value = input_img[src_row * width + col]
        kernel_weight = kernel[k]
        result += pixel_value * kernel_weight
        weight_sum += kernel_weight

    # Normalize by actual weight sum (important for boundary pixels)
    if weight_sum > 0:
        output_img[row * width + col] = result / weight_sum
    else:
        output_img[row * width + col] = input_img[row * width + col]


def gaussian_blur_cuda(detrended_stack: np.ndarray, sigma: float) -> np.ndarray:
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
    # Add extra space for sum storage during normalization
    kernel_gpu = cuda.device_array(kernel_size + 1, dtype=np.float32)

    # Generate Gaussian kernel on GPU
    threads_per_block = min(256, kernel_size)
    blocks_per_grid = math.ceil(kernel_size / threads_per_block)

    # Step 1: Generate unnormalized Gaussian values
    gaussian_kernel_1d_cuda[blocks_per_grid, threads_per_block](kernel_gpu, sigma, kernel_size)
    cuda.synchronize()

    # Step 2: Normalize kernel (parallel reduction)
    # Reset sum accumulator
    kernel_gpu[kernel_size] = 0.0
    normalize_kernel_cuda[blocks_per_grid, threads_per_block](kernel_gpu, kernel_size)
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
        convolve_horizontal_cuda[blocks_per_grid_2d, threads_per_block_2d](
            input_gpu, temp_gpu, kernel_gpu, height, width, kernel_size
        )
        cuda.synchronize()

        # Step 2: Apply vertical convolution to complete the 2D Gaussian filter
        convolve_vertical_cuda[blocks_per_grid_2d, threads_per_block_2d](
            temp_gpu, output_gpu, kernel_gpu, height, width, kernel_size
        )
        cuda.synchronize()

        # Copy result back to host using pre-allocated buffer
        output_gpu.copy_to_host(result_buffer)
        gaussian_stack[frame_idx] = result_buffer.reshape(height, width)

    return gaussian_stack
