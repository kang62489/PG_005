import numpy as np
import imageio.v2 as imageio
from numba import cuda, float32, jit, prange
import math
import time
from pathlib import Path
from rich.console import Console
import sys
from typing import Tuple
import os
# Force Numba to recompile
os.environ['NUMBA_DISABLE_FUNCTION_CACHING'] = '1'
console = Console()

def check_cuda():
    """Check CUDA availability and print diagnostic information"""
    try:
        import os
        import glob
        
        # First check if CUDA is actually available through Numba
        if cuda.is_available():
            device = cuda.get_current_device()
            console.print(f"[green]CUDA is available. Using device: {device.name}")
            console.print(f"[green]Compute Capability: {device.compute_capability}")
            console.print(f"[green]Max threads per block: {device.MAX_THREADS_PER_BLOCK}")
            return True
            
        console.print("[bold red]CUDA is not available through Numba. Checking why...")
        
        # Check NVIDIA driver
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            device_name = pynvml.nvmlDeviceGetName(handle)
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            console.print(f"[yellow]NVIDIA driver is installed")
            console.print(f"[yellow]GPU: {device_name}")
            console.print(f"[yellow]Driver Version: {driver_version}")
        except Exception as e:
            console.print(f"[bold red]NVIDIA driver not found or not properly installed: {str(e)}")
            return False
        
        # Check CUDA installation
        cuda_path = os.environ.get('CUDA_PATH')
        if not cuda_path:
            # Try to find CUDA installation
            base_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            if os.path.exists(base_path):
                # Look for all CUDA versions
                versions = glob.glob(os.path.join(base_path, "v*"))
                if versions:
                    # Sort versions in descending order
                    versions.sort(reverse=True)
                    cuda_path = versions[0]
                    os.environ['CUDA_PATH'] = cuda_path
                    
                    # Also set other CUDA environment variables
                    os.environ['CUDA_HOME'] = cuda_path
                    os.environ['PATH'] = f"{os.path.join(cuda_path, 'bin')};{os.environ['PATH']}"
                    os.environ['PATH'] = f"{os.path.join(cuda_path, 'libnvvp')};{os.environ['PATH']}"
                    
                    console.print(f"[yellow]Found CUDA installation at: {cuda_path}")
                    
                    # Try to reinitialize CUDA
                    if cuda.is_available():
                        device = cuda.get_current_device()
                        console.print(f"[green]Successfully initialized CUDA with device: {device.name}")
                        return True
                    else:
                        console.print("[bold red]Found CUDA but failed to initialize")
                else:
                    console.print(f"[bold red]No CUDA versions found in {base_path}")
            else:
                console.print("[bold red]Could not find CUDA installation directory")
        else:
            console.print(f"[yellow]CUDA_PATH is set to: {cuda_path}")
            if not os.path.exists(cuda_path):
                console.print("[bold red]CUDA_PATH points to non-existent directory")
        
        # Print current environment variables for debugging
        console.print("\n[yellow]Current CUDA environment variables:")
        for key in ['CUDA_PATH', 'CUDA_HOME', 'PATH']:
            console.print(f"[yellow]{key}: {os.environ.get(key, 'Not set')}")
        
        return False
        
    except Exception as e:
        console.print(f"[bold red]Error checking CUDA: {str(e)}")
        return False

@cuda.jit
def detrend_kernel(pixel_data: np.ndarray, output: np.ndarray, window_size: int) -> None:
    """
    CUDA kernel for detrending pixels by removing moving average.
    
    Args:
        pixel_data: Input array of shape (n_pixels, n_frames)
        output: Output array of same shape as input
        window_size: Size of moving average window
    """
    pixel_idx = cuda.grid(1)
    if pixel_idx >= pixel_data.shape[0]:
        return
        
    n_frames = pixel_data.shape[1]
    for frame_idx in range(n_frames):
        # Define window boundaries
        half_window = window_size // 2
        window_start = max(0, frame_idx - half_window)
        window_end = min(n_frames, frame_idx + half_window + 1)
        
        # Calculate moving average
        window_sum = 0.0
        window_size_actual = window_end - window_start
        for k in range(window_start, window_end):
            window_sum += pixel_data[pixel_idx, k]
        moving_avg = window_sum / window_size_actual
        
        # Subtract moving average from original value
        output[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - moving_avg

# Import the JIT function from the separate module
try:
    from spatial_processing import compute_spatial_averages
    spatial_processing_imported = True
except ImportError:
    spatial_processing_imported = False
    print("Could not import spatial_processing module, will use non-JIT version")

def process_on_gpu(image_stack, roi_size, window_size=100):
    """
    Process image stack using GPU for detrending and CPU for spatial averaging.
    
    Args:
        image_stack: Input array of shape (n_frames, height, width)
        roi_size: Size of Region of Interest (ROI)
        window_size: Size of moving average window for detrending
    
    Returns:
        Tuple of (detrended_stack, averaged_stack)
    """
    n_frames, height, width = image_stack.shape
    
    # Prepare data for detrending
    pixels_time_series = image_stack.reshape(n_frames, -1).T
    detrended_pixels = np.zeros_like(pixels_time_series, dtype=np.float32)
    
    # Transfer data to GPU
    gpu_input = cuda.to_device(pixels_time_series.astype(np.float32))
    gpu_output = cuda.to_device(detrended_pixels)
    
    # Configure CUDA grid
    threads_per_block = 256
    blocks_per_grid = math.ceil(pixels_time_series.shape[0] / threads_per_block)
    
    # Perform detrending on GPU
    console.print("[cyan]Detrending pixels on GPU...")
    start_time = time.time()
    detrend_kernel[blocks_per_grid, threads_per_block](gpu_input, gpu_output, window_size)
    cuda.synchronize()
    detrended_pixels = gpu_output.copy_to_host()
    console.print(f"Detrending time: {time.time() - start_time:.2f} seconds")
    
    # Reshape detrended data back to original dimensions
    detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)
    detrended_stack -= np.min(detrended_stack)
    
    # Compute spatial averages using pure NumPy (no JIT)
    console.print("[cyan]Computing spatial averages (NumPy)...")
    start_time = time.time()
    
    # Pure NumPy implementation of spatial averaging
    n_frames, height, width = detrended_stack.shape
    averaged_stack = np.zeros_like(detrended_stack, dtype=np.float32)
    
    # Adjust height and width to be multiples of roi_size
    height_adjusted = height - (height % roi_size)
    width_adjusted = width - (width % roi_size)
    
    # Process each frame
    for frame_idx in range(n_frames):
        # Reshape to group pixels into ROIs
        frame_reshaped = detrended_stack[frame_idx, :height_adjusted, :width_adjusted].reshape(
            height_adjusted // roi_size, roi_size, width_adjusted // roi_size, roi_size
        )
        
        # Calculate mean for each ROI
        roi_means = frame_reshaped.mean(axis=(1, 3))
        
        # Expand back to original size
        for i in range(height_adjusted // roi_size):
            for j in range(width_adjusted // roi_size):
                y_start = i * roi_size
                y_end = (i + 1) * roi_size
                x_start = j * roi_size
                x_end = (j + 1) * roi_size
                averaged_stack[frame_idx, y_start:y_end, x_start:x_end] = roi_means[i, j]
    
    console.print(f"Spatial averaging time: {time.time() - start_time:.2f} seconds")
    
    return detrended_stack, averaged_stack

def process_on_cpu(image_stack: np.ndarray, roi_size: int, window_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process image stack entirely on CPU using JIT.
    
    Args:
        image_stack: Input array of shape (n_frames, height, width)
        roi_size: Size of Region of Interest (ROI)
        window_size: Size of moving average window for detrending
    
    Returns:
        Tuple of (detrended_stack, averaged_stack)
    """
    @jit(nopython=True, parallel=True)
    def detrend_parallel(pixel_data: np.ndarray, window_size: int) -> np.ndarray:
        """Detrend pixels by removing moving average in parallel."""
        n_pixels, n_frames = pixel_data.shape
        detrended = np.zeros_like(pixel_data, dtype=np.float32)
        
        for pixel_idx in prange(n_pixels):
            for frame_idx in range(n_frames):
                window_start = max(0, frame_idx - window_size//2)
                window_end = min(n_frames, frame_idx + window_size//2)
                moving_avg = np.mean(pixel_data[pixel_idx, window_start:window_end])
                detrended[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - moving_avg
        
        return detrended
    
    n_frames, height, width = image_stack.shape
    pixels_time_series = image_stack.reshape(n_frames, -1).T
    
    # Perform detrending
    console.print("[cyan]Detrending pixels on CPU...")
    start_time = time.time()
    detrended_pixels = detrend_parallel(pixels_time_series.astype(np.float32), window_size)
    detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)
    detrended_stack -= np.min(detrended_stack)
    console.print(f"Detrending time: {time.time() - start_time:.2f} seconds")
    
    # Compute spatial averages
    console.print("[cyan]Computing spatial averages on CPU...")
    start_time = time.time()
    averaged_stack = compute_spatial_averages(detrended_stack, roi_size)
    console.print(f"Spatial averaging time: {time.time() - start_time:.2f} seconds")
    
    return detrended_stack, averaged_stack

def test_gpu():
    """Quick test to verify GPU functionality"""
    try:
        # Small test array
        test_data = np.ones((1000,), dtype=np.float32)
        d_test = cuda.to_device(test_data)
        
        @cuda.jit
        def test_kernel(arr):
            idx = cuda.grid(1)
            if idx < arr.size:
                arr[idx] = 2.0
                
        test_kernel[2, 512](d_test)
        cuda.synchronize()
        result = d_test.copy_to_host()
        
        if np.allclose(result, 2.0):
            console.print("[green]GPU test successful!")
            return True
        else:
            console.print("[bold red]GPU test failed!")
            return False
    except Exception as e:
        console.print(f"[bold red]GPU test error: {str(e)}")
        return False

def main():
    # Parameters
    filepath = '2025_06_11-0012.tif'
    output_name_1 = f'Corr_{filepath}'
    output_name_2 = f'Conv_{filepath}'
    roi_size = 32
    
    # Delete existing output files
    for output_file in [output_name_1, output_name_2]:
        Path(output_file).unlink(missing_ok=True)
    
    try:
        # Load image stack
        console.print("[bold green]Loading image stack...")
        t_start = time.time()
        img_raw = imageio.volread(f"raw_images/{filepath}").astype(np.float32)
        console.print(f"Loading time: {time.time() - t_start:.2f} seconds")
        console.print(f"Image shape: {img_raw.shape}")
        
        # Process on either GPU or CPU
        console.print("[bold green]Processing data...")
        t_start = time.time()
        if cuda.is_available():
            detrended, averaged = process_on_gpu(img_raw, roi_size)
        else:
            detrended, averaged = process_on_cpu(img_raw, roi_size)
        console.print(f"Total processing time: {time.time() - t_start:.2f} seconds")
        
        # Save results
        console.print("[bold green]Saving results...")
        detrended_uint16 = np.clip(detrended, 0, 65535).astype(np.uint16)
        averaged_uint16 = np.clip(averaged, 0, 65535).astype(np.uint16)
        
        imageio.volwrite(output_name_1, detrended_uint16)
        imageio.volwrite(output_name_2, averaged_uint16)
        
        console.print("[bold green]Processing completed successfully!")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Check CUDA availability with diagnostics
    cuda_available = check_cuda()
    
    if cuda_available:
        # Verify GPU functionality
        cuda_available = test_gpu()
    
    if not cuda_available:
        console.print("[yellow]Falling back to CPU processing...")
    
    main()










