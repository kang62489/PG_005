"""
zone_kernels.py  --  Heavy per-pixel steps of the spontaneous zone analysis (CPU Numba JIT + CUDA GPU).

  Step 1. Mask   : threshold -> 3x3 opening -> 3x3 closing  (GPU / numba), then
                   fill holes -> drop small blobs            (scipy labels, frames in parallel threads)
  Step 2. Traces : mean value of every detection footprint in every frame (GPU / numba)

The stack is float16; numba can't load float16, so kernels read the raw uint16 codes and
decode them through a 65536-entry lookup table (exact, identical to numpy's float16->float32).
Results match scipy / the numpy reference in classes/sp_zone_analyzer.py.
"""

## Modules
# Standard library imports
import math
from concurrent.futures import ThreadPoolExecutor
from functools import cache

# Third-party imports
import numpy as np
from numba import cuda, njit, prange
from scipy import ndimage

N_THREADS_FRAME_POOL = 16  # threads for the per-frame scipy label steps


@cache
def f16_lut() -> np.ndarray:
    """float32 value of every float16 bit pattern (index = uint16 code)."""
    return np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)


# ===========================================================================
#
#   STEP 1 -- MASK: threshold -> open -> close -> fill holes -> drop small blobs
#
# ===========================================================================

# --- 1a. CPU (numba, one frame per thread) --------------------------------
# scipy semantics: border_value=0 -> erosion treats outside-the-frame as False,
# dilation ignores outside-the-frame pixels.

@njit(inline="always")
def _erode_px(src: np.ndarray, y: int, x: int, h: int, w: int) -> bool:
    if y == 0 or x == 0 or y == h - 1 or x == w - 1:
        return False
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if not src[y + dy, x + dx]:
                return False
    return True


@njit(inline="always")
def _dilate_px(src: np.ndarray, y: int, x: int, h: int, w: int) -> bool:
    for dy in range(-1, 2):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        for dx in range(-1, 2):
            xx = x + dx
            if 0 <= xx < w and src[yy, xx]:
                return True
    return False


@njit(parallel=True)
def _cpu_threshold_open_close(codes: np.ndarray, lut: np.ndarray, thr: np.float32) -> np.ndarray:
    n_frames, h, w = codes.shape
    out = np.empty((n_frames, h, w), dtype=np.bool_)
    for t in prange(n_frames):
        a = np.empty((h, w), dtype=np.bool_)
        b = np.empty((h, w), dtype=np.bool_)
        for y in range(h):
            for x in range(w):
                a[y, x] = lut[codes[t, y, x]] > thr
        for y in range(h):  # opening = erode ...
            for x in range(w):
                b[y, x] = _erode_px(a, y, x, h, w)
        for y in range(h):  # ... then dilate
            for x in range(w):
                a[y, x] = _dilate_px(b, y, x, h, w)
        for y in range(h):  # closing = dilate ...
            for x in range(w):
                b[y, x] = _dilate_px(a, y, x, h, w)
        for y in range(h):  # ... then erode
            for x in range(w):
                out[t, y, x] = _erode_px(b, y, x, h, w)
    return out


# --- 1b. GPU (one thread per pixel) ---------------------------------------

@cuda.jit
def _gpu_threshold_kernel(codes: np.ndarray, lut: np.ndarray, thr: np.float32, out: np.ndarray) -> None:
    x, y, t = cuda.grid(3)
    if t < codes.shape[0] and y < codes.shape[1] and x < codes.shape[2]:
        out[t, y, x] = 1 if lut[codes[t, y, x]] > thr else 0


@cuda.jit
def _gpu_erode_kernel(src: np.ndarray, out: np.ndarray) -> None:
    x, y, t = cuda.grid(3)
    n_frames, h, w = src.shape
    if t >= n_frames or y >= h or x >= w:
        return
    if y == 0 or x == 0 or y == h - 1 or x == w - 1:
        out[t, y, x] = 0
        return
    val = 1
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if src[t, y + dy, x + dx] == 0:
                val = 0
    out[t, y, x] = val


@cuda.jit
def _gpu_dilate_kernel(src: np.ndarray, out: np.ndarray) -> None:
    x, y, t = cuda.grid(3)
    n_frames, h, w = src.shape
    if t >= n_frames or y >= h or x >= w:
        return
    val = 0
    for dy in range(-1, 2):
        yy = y + dy
        if 0 <= yy < h:
            for dx in range(-1, 2):
                xx = x + dx
                if 0 <= xx < w and src[t, yy, xx] != 0:
                    val = 1
    out[t, y, x] = val


def _gpu_threshold_open_close(codes: np.ndarray, lut: np.ndarray, thr: np.float32) -> np.ndarray:
    n_frames, h, w = codes.shape
    threads = (32, 8, 1)
    blocks = (math.ceil(w / 32), math.ceil(h / 8), n_frames)

    d_codes = cuda.to_device(codes)
    d_lut = cuda.to_device(lut)
    d_a = cuda.device_array((n_frames, h, w), dtype=np.uint8)
    d_b = cuda.device_array((n_frames, h, w), dtype=np.uint8)

    _gpu_threshold_kernel[blocks, threads](d_codes, d_lut, thr, d_a)
    del d_codes
    _gpu_erode_kernel[blocks, threads](d_a, d_b)   # opening
    _gpu_dilate_kernel[blocks, threads](d_b, d_a)
    _gpu_dilate_kernel[blocks, threads](d_a, d_b)  # closing
    _gpu_erode_kernel[blocks, threads](d_b, d_a)
    cuda.synchronize()
    return d_a.copy_to_host().view(np.bool_)


# --- 1c. Fill holes + drop small blobs (scipy labels, per frame) -----------

_CROSS = ndimage.generate_binary_structure(2, 1)  # 4-connectivity, scipy's default for both steps


def _fill_and_filter_frame(frame: np.ndarray, th_small_obj: int) -> np.ndarray:
    """fill holes (background 4-regions not touching the edge) -> keep blobs >= th_small_obj px."""
    bg_labels, _ = ndimage.label(~frame, structure=_CROSS)
    edge = np.unique(np.concatenate([bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1]]))
    is_edge_bg = np.zeros(bg_labels.max() + 1, dtype=bool)
    is_edge_bg[edge] = True
    filled = frame | ~is_edge_bg[bg_labels]  # label 0 = foreground, never an edge background label

    labels, n = ndimage.label(filled, structure=_CROSS)
    if n == 0:
        return filled
    sizes = np.bincount(labels.ravel())
    keep = sizes >= th_small_obj
    keep[0] = False
    return keep[labels]


# --- 1d. Dispatch ----------------------------------------------------------

def zone_mask(stack_f16: np.ndarray, threshold: float, th_small_obj: int, cuda_available: bool = False) -> np.ndarray:
    """Threshold -> open -> close -> fill holes -> drop small blobs, same result as the scipy chain."""
    codes = stack_f16.view(np.uint16)
    thr = np.float32(np.float16(threshold))  # numpy compares a float16 array against a float16-rounded scalar
    lut = f16_lut()
    if cuda_available:
        mask = _gpu_threshold_open_close(codes, lut, thr)
    else:
        mask = _cpu_threshold_open_close(codes, lut, thr)

    with ThreadPoolExecutor(N_THREADS_FRAME_POOL) as pool:
        frames = list(pool.map(lambda frame: _fill_and_filter_frame(frame, th_small_obj), mask))
    return np.stack(frames)


# ===========================================================================
#
#   STEP 2 -- TRACES: mean of every detection footprint in every frame
#
# ===========================================================================

# --- 2a. CPU (numba, one frame per thread) --------------------------------

@njit(parallel=True)
def _cpu_footprint_traces(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                          starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n_frames = codes_flat.shape[0]
    n_det = starts.shape[0]
    means = np.empty((n_det, n_frames), dtype=np.float32)
    for t in prange(n_frames):
        for d in range(n_det):
            s = np.float32(0.0)
            for k in range(starts[d], starts[d] + counts[d]):
                s += lut[codes_flat[t, linear_idx[k]]]
            means[d, t] = s / np.float32(counts[d])
    return means


# --- 2b. GPU (one block per detection x frame, shared-memory reduction) ----

TRACE_THREADS = 256


@cuda.jit
def _gpu_footprint_traces_kernel(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                                 starts: np.ndarray, counts: np.ndarray, means: np.ndarray) -> None:
    d = cuda.blockIdx.x
    t = cuda.blockIdx.y
    tid = cuda.threadIdx.x
    partial = cuda.shared.array(TRACE_THREADS, dtype=np.float32)

    s = np.float32(0.0)
    start = starts[d]
    for k in range(start + tid, start + counts[d], TRACE_THREADS):
        s += lut[codes_flat[t, linear_idx[k]]]
    partial[tid] = s
    cuda.syncthreads()

    stride = TRACE_THREADS // 2
    while stride > 0:
        if tid < stride:
            partial[tid] += partial[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        means[d, t] = partial[0] / np.float32(counts[d])


def _gpu_footprint_traces(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                          starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n_frames, n_det = codes_flat.shape[0], starts.shape[0]
    d_means = cuda.device_array((n_det, n_frames), dtype=np.float32)
    _gpu_footprint_traces_kernel[(n_det, n_frames), TRACE_THREADS](
        cuda.to_device(codes_flat), cuda.to_device(lut), cuda.to_device(linear_idx),
        cuda.to_device(starts), cuda.to_device(counts), d_means,
    )
    cuda.synchronize()
    return d_means.copy_to_host()


# --- 2c. Dispatch ----------------------------------------------------------

def footprint_traces(footprints: list, stack_f16: np.ndarray, cuda_available: bool = False) -> np.ndarray:
    """(n_detections, n_frames) float16: mean over each detection's own footprint, every frame."""
    n_frames, _, width = stack_f16.shape
    counts = np.array([len(coords) for coords in footprints], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    all_coords = np.concatenate(footprints, axis=0)
    linear_idx = (all_coords[:, 0] * width + all_coords[:, 1]).astype(np.int64)

    codes_flat = stack_f16.view(np.uint16).reshape(n_frames, -1)
    lut = f16_lut()
    if cuda_available:
        means = _gpu_footprint_traces(codes_flat, lut, linear_idx, starts, counts)
    else:
        means = _cpu_footprint_traces(codes_flat, lut, linear_idx, starts, counts)
    return means.astype(np.float16)
