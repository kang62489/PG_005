# PG_005 Data Flow Structure

## Data Flow Overview

```
Raw TIFF Image Stack
         ↓
    im_process_main.py
         ↓
   [CUDA Detection]
         ↓
    ┌─────────┴─────────┐
    ↓                   ↓
CPU Path            GPU Path
    ↓                   ↓
    └─────────┬─────────┘
         ↓
   Three Output Files
```

## Detailed Data Flow

### 1. Input Stage
**File**: `im_process_main.py:45-46`
```
raw_images/filename.tif → imageio.volread() → img_raw (numpy array)
```
- **Input**: TIFF image stack from `raw_images/` directory
- **Shape**: `(n_frames, height, width)` as `np.float32`
- **Example**: `2025_10_13-0007.tif`

### 2. Processing Path Selection
**File**: `im_process_main.py:56-59`
```
if cuda.is_available():
    detrended, averaged, gaussian = process_on_gpu(img_raw, roi_size)
else:
    detrended, averaged, gaussian = process_on_cpu(img_raw, roi_size)
```

### 3A. CPU Processing Path
**File**: `functions/cpu_processing.py:23-84`

#### Step 1: Data Reshaping
**Location**: `cpu_processing.py:40`
```
img_raw (n_frames, height, width) 
    ↓ reshape
pixels_time_series (n_pixels, n_frames) [transposed]
```

#### Step 2: Detrending
**Function**: `detrend_parallel()` from `cpu_parallel_detrend.py:7-27`
**Called at**: `cpu_processing.py:57`
```
pixels_time_series → [moving average removal] → detrended_pixels
```
- Removes moving average baseline from each pixel's time series
- Uses sliding window (default: 101 frames)
- Output: `detrended_stack (n_frames, height, width)`

#### Step 3: Spatial Averaging
**Function**: `compute_spatial_averages()` from `spatial_processing.py:6-38`
**Called at**: `cpu_processing.py:72`
```
detrended_stack → [ROI averaging] → averaged_stack
```
- Divides image into ROI blocks (default: 4x4 pixels)
- Replaces each ROI with its average value
- Output: `averaged_stack (n_frames, height, width)`

#### Step 4: Gaussian Filtering
**Function**: `gaussian_blur()` from `gaussian_filter.py`
**Called at**: `cpu_processing.py:80`
```
detrended_stack → [Gaussian convolution] → gaussian_stack
```
- Applies Gaussian smoothing (default: σ=8.0)
- Output: `gaussian_stack (n_frames, height, width)`

### 3B. GPU Processing Path
**File**: `functions/gpu_processing.py:28-84`

#### Step 1: Data Transfer and Detrending
**GPU Transfer**: `gpu_processing.py:49-50`
**CUDA Kernel**: `gpu_processing.py:59`
**Copy Back**: `gpu_processing.py:61`
```
pixels_time_series → cuda.to_device() → GPU memory
    ↓
[CUDA kernel detrending] → detrended_pixels
    ↓
copy_to_host() → detrended_stack
```

#### Step 2: Spatial Averaging (CPU)
**Called at**: `gpu_processing.py:75`
Same as CPU path - uses `compute_spatial_averages()`

#### Step 3: Gaussian Filtering (GPU)
**Function**: `gaussian_blur_cuda()` from `gaussian_filter.py`
**Called at**: `gpu_processing.py:81`
```
detrended_stack → [GPU Gaussian convolution] → gaussian_stack
```

### 4. Output Stage
**File**: `im_process_main.py:74-76`
```
detrended → filename_Corr.tif
averaged  → filename_Conv.tif  
gaussian  → filename_Gauss.tif
```
- All outputs saved as `np.float16` for storage efficiency

## Data Transformations

### Shape Transformations
1. **Input**: `(n_frames, height, width)`
2. **Reshaping for detrending**: `(n_pixels, n_frames)` where `n_pixels = height × width`
3. **Back to image format**: `(n_frames, height, width)`
4. **Output**: Same shape as input

### Data Type Transformations
1. **Load**: `np.float32` (from TIFF)
2. **Processing**: `np.float32` (throughout pipeline)
3. **Save**: `np.float16` (for storage)

### Key Processing Operations

#### Detrending Algorithm
- **Purpose**: Remove baseline drift from time series
- **Method**: Subtract moving average, normalize to edge minimum
- **Window**: Sliding window around each time point

#### Spatial Averaging Algorithm  
- **Purpose**: Spatial downsampling/smoothing
- **Method**: Replace ROI blocks with their average value
- **ROI Size**: Configurable (default: 4x4 pixels)

#### Gaussian Filtering Algorithm
- **Purpose**: Noise reduction and smoothing
- **Method**: Separable Gaussian convolution
- **Parameters**: Configurable sigma (CPU: 8.0, GPU: 4.0)

## Parallel Processing Strategy

### CPU Path
- **Detrending**: Parallel across pixels using Numba `prange`
- **Spatial Averaging**: Parallel across frames using Numba `prange`
- **Gaussian**: CPU implementation with JIT compilation

### GPU Path  
- **Detrending**: CUDA kernel with thread-per-pixel parallelization
- **Spatial Averaging**: CPU fallback (no GPU implementation)
- **Gaussian**: GPU CUDA kernels for convolution

## Memory Management
- GPU transfers managed explicitly with `cuda.to_device()` and `copy_to_host()`
- Memory usage monitoring via `get_memory_usage()`
- Progress tracking with Rich progress bars