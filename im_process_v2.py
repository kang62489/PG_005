import numpy as np
import imageio.v2 as imageio
from numba import jit, prange
import time
from pathlib import Path
import argparse
import yaml
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler
import psutil
import glob

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

@jit(nopython=True, parallel=True)
def detrend_parallel(data, window_size=11):
    """Detrend pixels in parallel using Numba"""
    n_pixels, n_frames = data.shape
    result = np.zeros_like(data, dtype=np.float32)
    
    for i in prange(n_pixels):
        pixel_values = data[i].astype(np.float32)
        moving_avg = np.zeros_like(pixel_values)
        for j in range(n_frames):
            start_idx = max(0, j - window_size//2)
            end_idx = min(n_frames, j + window_size//2)
            moving_avg[j] = np.mean(pixel_values[start_idx:end_idx])
            
        pure_trend = moving_avg - min(moving_avg[0], moving_avg[-1])
        result[i] = pixel_values - pure_trend
    
    return result

@jit(nopython=True)
def spatial_average(frame, roi_size):
    """Compute ROI averages for a single frame"""
    h, w = frame.shape
    result = np.zeros_like(frame, dtype=np.float32)
    
    h_rois = h // roi_size
    w_rois = w // roi_size
    
    for i in range(h_rois):
        for j in range(w_rois):
            y_start = i * roi_size
            y_end = (i + 1) * roi_size
            x_start = j * roi_size
            x_end = (j + 1) * roi_size
            
            window_avg = np.mean(frame[y_start:y_end, x_start:x_end])
            result[y_start:y_end, x_start:x_end] = window_avg
    
    return result

@jit(nopython=True, parallel=True)
def process_frames(data, roi_size):
    """Process all frames in parallel"""
    n_frames, h, w = data.shape
    result = np.zeros_like(data, dtype=np.float32)
    
    for i in prange(n_frames):
        result[i] = spatial_average(data[i], roi_size)
    
    return result

def validate_input(filepath, roi_size):
    """Validate input parameters"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    # Check if file is a valid TIFF
    try:
        # Read the image to get its shape directly
        img = imageio.volread(filepath)
        shape = img.shape
        
        if len(shape) != 3:
            raise ValueError("Input must be a 3D TIFF stack")
        
        # Verify ROI size is valid
        if shape[1] % roi_size != 0 or shape[2] % roi_size != 0:
            raise ValueError(f"ROI size ({roi_size}) must be a factor of image dimensions ({shape[1]}, {shape[2]})")
            
    except Exception as e:
        raise ValueError(f"Invalid TIFF file: {str(e)}")

def process_file(filepath, roi_size, config):
    """Process a single file"""
    output_name_1 = f'Corr_{Path(filepath).name}'
    output_name_2 = f'Conv_{Path(filepath).name}'
    
    # Delete existing output files
    for output_file in [output_name_1, output_name_2]:
        Path(output_file).unlink(missing_ok=True)
    
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Load image stack
        task1 = progress.add_task("[cyan]Loading image stack...", total=1)
        t_start = time.time()
        img_raw = imageio.volread(filepath).astype(np.float32)
        progress.update(task1, advance=1)
        log.info(f"Loading time: {time.time() - t_start:.2f} seconds")
        log.info(f"Image shape: {img_raw.shape}")
        log.info(f"Memory usage: {get_memory_usage():.2f} GB")
        
        # Detrend pixels
        task2 = progress.add_task("[cyan]Detrending pixels...", total=1)
        t_start = time.time()
        frames, height, width = img_raw.shape
        forDetrend = img_raw.reshape(frames, -1).T
        detrended = detrend_parallel(forDetrend, config['window_size'])
        row_averages = np.mean(detrended, axis=1)
        minimum_average = np.min(row_averages)
        diff_averages = row_averages - minimum_average
        detrended -= diff_averages[:, np.newaxis]
        
        detrended = detrended.T.reshape(frames, height, width)
        # detrended -= np.min(detrended)
        progress.update(task2, advance=1)
        log.info(f"Detrending time: {time.time() - t_start:.2f} seconds")
        
        # Process spatial averaging
        task3 = progress.add_task("[cyan]Processing spatial averaging...", total=1)
        t_start = time.time()
        averaged = process_frames(detrended, roi_size)
        progress.update(task3, advance=1)
        log.info(f"Processing time: {time.time() - t_start:.2f} seconds")
        
        # Save results
        task4 = progress.add_task("[cyan]Saving results...", total=1)
        detrended_uint16 = np.clip(detrended, 0, 65535).astype(np.uint16)
        averaged_uint16 = np.clip(averaged, 0, 65535).astype(np.uint16)
        
        imageio.volwrite(output_name_1, detrended_uint16)
        imageio.volwrite(output_name_2, averaged_uint16)
        progress.update(task4, advance=1)
        
        log.info("[bold green]Processing completed successfully!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process TIFF image stacks')
    parser.add_argument('--input', '-i', help='Input file or directory pattern (e.g., "*.tif")')
    parser.add_argument('--roi-size', '-r', type=int, default=16, help='ROI size (default: 16)')
    parser.add_argument('--config', '-c', help='Configuration file path', default='config.yaml')
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'window_size': 100,  # default value
    }
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Handle input pattern
    if args.input:
        files = glob.glob(args.input)
    else:
        files = ['raw_images/2025_06_11-0013.tif']  # default file
    
    if not files:
        log.error("No input files found!")
        return
    
    # Process each file
    for filepath in files:
        try:
            log.info(f"Processing {filepath}")
            validate_input(filepath, args.roi_size)
            process_file(filepath, args.roi_size, config)
        except Exception as e:
            log.error(f"Error processing {filepath}: {str(e)}")

if __name__ == "__main__":
    # Warm up Numba functions
    log.info("Initializing Numba functions...")
    test_data = np.random.rand(10, 100).astype(np.float32)
    _ = detrend_parallel(test_data, 10)
    test_frame = np.random.rand(128, 128).astype(np.float32)
    _ = spatial_average(test_frame, 32)
    log.info("Initialization complete!")
    
    main()
