---
keywords: numba, cuda, gpu, jit, kernel, device_array, to_device, copy_to_host, syncthreads, grid
files_referenced: functions/check_cuda.py, functions/test_cuda.py, functions/gpu_detrend.py, functions/gpu_gauss.py, functions/gpu_process.py
---

# numba.cuda Reference — PG_005 Usage Guide

A practical reference for the `numba.cuda` commands used in this project,
using a concrete **8×8×12** TIFF stack as the running example throughout.

---

## Full Example — Processing an 8×8×12 Stack

```python
import math
import numpy as np
from numba import cuda                                          # (A) import
from functions import gpu_detrend_jitted, gpu_gaussian_blur

image_stack = np.random.rand(12, 8, 8).astype(np.float32)
# shape: (n_frames=12, height=8, width=8), n_pixels = 8*8 = 64

# (B) reshape: (12,8,8) → (64,12) so each row = one pixel's time series
pixels_time_series = image_stack.reshape(12, -1).T            # (64, 12)
detrended_pixels   = np.zeros_like(pixels_time_series)        # (64, 12)

# (C) copy input to GPU; pre-allocate output space on GPU for kernel to write into
gpu_input  = cuda.to_device(pixels_time_series)               # GPU array (64, 12) — input data
gpu_output = cuda.device_array(pixels_time_series.shape, dtype=np.float32)  # empty GPU space (64, 12) — output

# (D) configure how many threads/blocks to launch
threads_per_block = 256
blocks_per_grid   = math.ceil(64 / 256)                       # = 1

# (E) launch the kernel — asynchronous, CPU does not wait
gpu_detrend_jitted[blocks_per_grid, threads_per_block](gpu_input, gpu_output, window_size=5)

# (F) wait for GPU to finish, then copy result back to CPU
cuda.synchronize()
detrended_pixels = gpu_output.copy_to_host()                  # (64, 12) numpy array

# (G) reshape result back to (12, 8, 8)
detrended_stack = detrended_pixels.T.reshape(12, 8, 8)

# (H) Gaussian blur — handled inside gpu_gaussian_blur()
gaussian_stack = gpu_gaussian_blur(detrended_stack, sigma=1.5)
# returns gaussian_stack.shape == (12, 8, 8)
```

Data shape at each stage:
```
image_stack            (12, 8, 8)   — input
pixels_time_series     (64, 12)     — row i = pixel i's time series across 12 frames
gpu_input / gpu_output (64, 12)     — same arrays, living on GPU
detrended_pixels       (64, 12)     — result copied back from GPU
detrended_stack        (12, 8, 8)   — reshaped back
gaussian_stack         (12, 8, 8)   — final output
```

The sections below explain each labelled step **(A–H)**.

---

## (A) Import

```python
from numba import cuda
```

Must be imported **after** CUDA environment variables are set (see `check_cuda.py`).

---

## (B) Why Reshape to (64, 12)?

The GPU kernel assigns **one thread per pixel**. Each thread needs to process that pixel's
full time series (12 frames). So we reshape the stack so that:

```
pixels_time_series[0]  = time series of pixel (row=0, col=0) → [f0, f1, ..., f11]
pixels_time_series[1]  = time series of pixel (row=0, col=1) → [f0, f1, ..., f11]
...
pixels_time_series[63] = time series of pixel (row=7, col=7) → [f0, f1, ..., f11]
```

Thread 0 will process row 0, thread 1 will process row 1, and so on.

---

## (C) Copying Data to the GPU — `cuda.to_device`

```python
gpu_input  = cuda.to_device(pixels_time_series)   # allocates GPU memory + copies data in
gpu_output = cuda.to_device(detrended_pixels)      # allocates GPU memory + copies data in
```

- `cuda.to_device` — allocates a new GPU array and copies the host data into it. Use for **input**.
- `cuda.device_array` — allocates empty GPU space, no data transfer. Use for **output** — the kernel writes into it, so there's no point transferring anything.

Inside `gpu_gaussian_blur()`, the same GPU buffer is reused across all 12 frames.
Allocate once **before** the loop, refill each iteration:

```python
# Before the loop — allocate the buffer once
frame_gpu = cuda.device_array(64, dtype=np.float32)

# Inside the loop — overwrite with each frame's data, no new allocation
for frame_idx in range(12):
    cuda.to_device(detrended_stack[frame_idx].flatten(), to=frame_gpu)
    # ... run kernel on frame_gpu ...
```

- **`cuda.device_array`** — allocate empty GPU space once (like `np.empty`, no data transfer).
- **`to=`** — overwrite an existing GPU buffer with new data instead of allocating a new one each frame.

---

## (D) Configuring the Launch Grid — blocks and threads

The GPU runs threads in groups called **blocks**. You decide how many threads per block,
and how many blocks to launch:

```python
threads_per_block = 256                        # how many threads in each block
blocks_per_grid   = math.ceil(64 / 256)  # = 1   how many blocks to launch
```

```
Grid
└── Block 0  [ thread 0, thread 1, thread 2, ..., thread 255 ]
                   ↓         ↓         ↓               ↓
               pixel 0   pixel 1   pixel 2      (no pixel — idle)
```

Block sizes must be a multiple of 32 (GPU warp size), so 64 would also be valid.
256 (= 8 warps) is preferred because when a warp stalls waiting for memory, the GPU
switches to another warp to keep busy. With only 2 warps (64 threads) there's little
to switch to. With 8 warps (256 threads) the scheduler has more flexibility to hide
memory latency. 256 is the standard default for 1D kernels in this project.

For a real 512×512 stack: `math.ceil(262144 / 256) = 1024 blocks` — no idle threads.

---

## (E) Launching and Defining the Kernel — `@cuda.jit` and `kernel[blocks, threads](args)`

```python
gpu_detrend_jitted[blocks_per_grid, threads_per_block](gpu_input, gpu_output, window_size=5)
#                  ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
#                  how many blocks   threads per block
```

`gpu_detrend_jitted` is defined in `gpu_detrend.py` with `@cuda.jit`:

```python
@cuda.jit
def gpu_detrend_jitted(pixel_data, output, window_size):
    pixel_idx = cuda.grid(1)                          # "which thread am I?" → 0, 1, 2, ... 255
    if pixel_idx >= pixel_data.shape[0]:              # threads 64-255 have no pixel → exit
        return
    # ... process pixel_data[pixel_idx] across 12 frames ...
```

- **`@cuda.jit`** — compiles this function to run on the GPU. No return value allowed.
- **`cuda.grid(1)`** — each thread calls this and gets back its own unique integer (0, 1, 2 ... 255).
  Thread 5 gets `5`, thread 42 gets `42`. That integer is used to pick which row of `pixel_data` to work on.
- **`if pixel_idx >= pixel_data.shape[0]: return`** — 256 threads launched but only 64 pixels exist.
  Threads 64–255 would access out-of-bounds memory without this guard → crash.

```
thread  0: pixel_idx=0,   0  >= 64? NO  → processes pixel_data[0]  (12 frames)
thread 63: pixel_idx=63,  63 >= 64? NO  → processes pixel_data[63] (12 frames)
thread 64: pixel_idx=64,  64 >= 64? YES → return (idle)
thread255: pixel_idx=255, 255>= 64? YES → return (idle)
```

Inside the kernel, `cuda.local.array` gives each thread its own private scratch buffer:

```python
moving_averages = cuda.local.array(2048, dtype=np.float32)
# thread 0 gets its own copy, thread 1 gets its own copy, etc. — no sharing
# size must be a compile-time constant; only indices 0..11 are used for 12 frames
```

---

## (F) Waiting and Copying Back — `cuda.synchronize` and `copy_to_host`

```python
cuda.synchronize()                        # CPU blocks here until GPU finishes
detrended_pixels = gpu_output.copy_to_host()   # copies GPU array → new numpy array (64, 12)
```

Kernel launches are **asynchronous** — the CPU moves on immediately after `(E)`.
Without `synchronize()`, `copy_to_host()` would run before the GPU is done → wrong results.

Inside `gpu_gaussian_blur()`, a pre-allocated buffer avoids repeated allocation across 12 frames:

```python
result_buffer = np.empty(64, dtype=np.float32)    # allocated once outside the loop
output_gpu.copy_to_host(result_buffer)             # copies into existing buffer — no allocation
```

---

## (G) Reshape Back

```python
detrended_stack = detrended_pixels.T.reshape(12, 8, 8)
# (64, 12) → .T → (12, 64) → reshape → (12, 8, 8)
```

Reverses step (B). Each column of `detrended_pixels` becomes one frame of the stack.

---

## (H) Gaussian Blur — 2D kernel indexing

`gpu_gaussian_blur()` processes the stack **one frame at a time** in a loop.
Each frame is `(8, 8)` — naturally 2D, so threads are assigned a `(row, col)` pair instead of a flat index.

**Step 1 — Launch with a 2D grid/block config** (`gpu_gauss.py`)

```python
threads_per_block_2d = (16, 16)                   # 16×16 = 256 threads per block, arranged in 2D
blocks_per_grid_x = math.ceil(8 / 16)             # = 1
blocks_per_grid_y = math.ceil(8 / 16)             # = 1
blocks_per_grid_2d = (1, 1)

convolve_horizontal[(1,1), (16,16)](input_gpu, temp_gpu, kernel_gpu, 8, 8, kernel_size)
# → 1×1 blocks, each block 16×16 threads = 256 threads total
```

The grid and block are now both 2D tuples — `(x, y)` — matching the image shape.

---

**Step 2 — Each thread finds its (row, col)** (`gpu_gauss.py`)

```python
row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
```

This is the 2D equivalent of `cuda.grid(1)`. Each thread computes its own unique `(row, col)`:

```
thread (0,0)  → row=0, col=0  → pixel [0,0]
thread (0,1)  → row=0, col=1  → pixel [0,1]
...
thread (0,7)  → row=0, col=7  → pixel [0,7]
thread (1,0)  → row=1, col=0  → pixel [1,0]
...
thread (7,7)  → row=7, col=7  → pixel [7,7]
thread (0,8)  → row=0, col=8  → no pixel! (col=8 >= width=8)
...
thread (15,15)→ row=15,col=15 → no pixel! (both out of range)
```

Same as `cuda.grid(2)` — just written out manually. 256 threads launched, 64 active.

---

**Step 3 — Guard against out-of-bounds** (`gpu_gauss.py`)

```python
if row >= height or col >= width:   # height=8, width=8
    return
```

```
thread (7,7):  row=7 >= 8? NO,  col=7 >= 8? NO  → proceeds
thread (0,8):  row=0 >= 8? NO,  col=8 >= 8? YES → return (idle)
thread (8,0):  row=8 >= 8? YES                  → return (idle)
```

---

**Step 4 — `cuda.syncthreads()` when generating the Gaussian kernel weights**

Before convolving, the kernel weights are computed — each thread writes one weight,
then thread 0 normalizes all of them. Thread 0 must wait until every other thread has written:

```python
kernel_out[thread_idx] = math.exp(...)    # all threads write simultaneously
cuda.syncthreads()                         # barrier: wait until ALL threads have written
if thread_idx == 0:                        # only thread 0 runs this
    kernel_sum = 0.0
    for i in range(size):
        kernel_sum += kernel_out[i]        # safe — all values are ready
    for i in range(size):
        kernel_out[i] /= kernel_sum
```

Without `syncthreads()`, thread 0 might read `kernel_out[5]` before thread 5 has written it.

---

## Summary Table

| Command | Step | What It Does |
|---------|------|--------------|
| `from numba import cuda` | A | Import cuda module |
| `cuda.to_device(arr)` | C | Copy host array → new GPU array |
| `cuda.device_array(shape, dtype)` | C | Allocate empty array on GPU |
| `cuda.to_device(arr, to=buf)` | C | Copy into existing GPU buffer (no allocation) |
| `kernel[blocks, threads](args)` | D/E | Launch kernel with grid config |
| `@cuda.jit` | E | Compile function as GPU kernel |
| `cuda.grid(1)` | E | Each thread gets its unique index (1D) |
| `cuda.local.array(N, dtype)` | E | Thread-private scratch buffer |
| `cuda.synchronize()` | F | Wait for GPU to finish |
| `d_arr.copy_to_host()` | F | Copy GPU array → new numpy array |
| `d_arr.copy_to_host(buf)` | F | Copy into existing numpy array (no allocation) |
| `cuda.blockIdx/blockDim/threadIdx` | H | 2D thread coordinates |
| `cuda.syncthreads()` | H | Barrier: sync all threads in a block |
| `cuda.is_available()` | — | Check if GPU is usable |
| `cuda.get_current_device()` | — | Get device info (name, compute capability) |

---

*Last updated: 2026-05-07*
