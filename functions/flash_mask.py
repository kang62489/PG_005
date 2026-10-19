"""
Flash mask of a float16 z-stack, CPU Numba JIT + CUDA GPU (SREAnalyzer.detect, permutation test).

  Step 1. Threshold + open/close : threshold -> 3x3 opening -> 3x3 closing (GPU / numba CPU)
  Step 2. Fill + filter          : fill holes -> drop blobs < th_small_obj (scipy labels, per frame)
  Step 3. Sizes                  : fill holes -> size (px) of every flash (permutation test)

The threshold itself: functions/fit_hist.fit_flash_background (shared with img_proc).
The stack is float16; numba can't load float16, so kernels read the raw uint16 codes and decode them through a
65536-entry lookup table (exact, identical to numpy's float16->float32). Results match the scipy chain.

Example:
    >>> mask = flash_mask(stack_f16, threshold, th_small_obj, cuda_available=True)
    >>> sizes = mask_flash_sizes(_threshold_open_close(stack_f16, threshold, cuda_available=True))
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

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

N_THREADS_FRAME_POOL = 16     # threads for the per-frame scipy label steps (fill holes, sizes)


@cache
def f16_lut() -> np.ndarray:
    """float32 value of every float16 bit pattern (index = uint16 code)."""
    return np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)


# ===========================================================================
#
#   THRESHOLD + OPEN/CLOSE
#
# ===========================================================================

# fn: _erode_px
# fn: _dilate_px
# fn: _cpu_threshold_open_close
# fn: _gpu_threshold_kernel
# fn: _gpu_erode_kernel
# fn: _gpu_dilate_kernel
# fn: _gpu_threshold_open_close
# fn: _threshold_open_close, threshold -> 3x3 opening -> 3x3 closing (GPU / numba), before fill holes.

# --- CPU (numba, one frame per thread) -------------------------------------

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


# --- GPU (one thread per pixel) --------------------------------------------

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


# --- Dispatch --------------------------------------------------------------

def _threshold_open_close(stack_f16: np.ndarray, threshold: float, cuda_available: bool) -> np.ndarray:
    """Threshold -> 3x3 opening -> 3x3 closing (GPU / numba), before fill holes."""
    codes = stack_f16.view(np.uint16)
    thr = np.float32(np.float16(threshold))  # numpy compares a float16 array against a float16-rounded scalar
    if cuda_available:
        return _gpu_threshold_open_close(codes, f16_lut(), thr)
    return _cpu_threshold_open_close(codes, f16_lut(), thr)


# ===========================================================================
#
#   FILL + FILTER
#
# ===========================================================================

# fn: _fill_holes, fill holes = background 4-regions not touching the edge become foreground.
# fn: _fill_and_filter_frame, fill holes -> keep blobs >= th_small_obj px.
# fn: flash_mask, threshold -> open -> close -> fill holes -> drop small blobs, same result as the scipy chain.

_CROSS = ndimage.generate_binary_structure(2, 1)  # 4-connectivity, scipy's default for both steps


def _fill_holes(frame: np.ndarray) -> np.ndarray:
    """fill holes = background 4-regions not touching the edge become foreground."""
    bg_labels, _ = ndimage.label(~frame, structure=_CROSS)
    edge = np.unique(np.concatenate([bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1]]))
    is_edge_bg = np.zeros(bg_labels.max() + 1, dtype=bool)
    is_edge_bg[edge] = True
    return frame | ~is_edge_bg[bg_labels]  # label 0 = foreground, never an edge background label


def _fill_and_filter_frame(frame: np.ndarray, th_small_obj: int) -> np.ndarray:
    """fill holes -> keep blobs >= th_small_obj px."""
    filled = _fill_holes(frame)
    labels, n = ndimage.label(filled, structure=_CROSS)
    if n == 0:
        return filled
    sizes = np.bincount(labels.ravel())
    keep = sizes >= th_small_obj
    keep[0] = False
    return keep[labels]


def flash_mask(stack_f16: np.ndarray, threshold: float, th_small_obj: int, cuda_available: bool = False) -> np.ndarray:
    """Threshold -> open -> close -> fill holes -> drop small blobs, same result as the scipy chain."""
    mask = _threshold_open_close(stack_f16, threshold, cuda_available)
    with ThreadPoolExecutor(N_THREADS_FRAME_POOL) as pool:
        frames = list(pool.map(lambda frame: _fill_and_filter_frame(frame, th_small_obj), mask))
    return np.stack(frames)


# ===========================================================================
#
#   SIZES
#
# ===========================================================================

# fn: _frame_flash_sizes, fill holes -> size (px) of every flash of one frame, labelled once.
# fn: mask_flash_sizes, fill holes + size (px) of every 4-connected flash of a threshold -> open -> close mask, frame by
#     frame.

def _frame_flash_sizes(frame: np.ndarray) -> np.ndarray:
    """Fill holes -> size (px) of every flash of one frame, labelled once."""
    filled = _fill_holes(frame)
    labels, n = ndimage.label(filled, structure=_CROSS)
    return np.bincount(labels[filled])[1:] if n else np.zeros(0, dtype=np.int64)  # foreground px only


def mask_flash_sizes(opened_closed: np.ndarray) -> np.ndarray:
    """Fill holes + size (px) of every 4-connected flash of a threshold -> open -> close mask, frame by frame."""
    with ThreadPoolExecutor(N_THREADS_FRAME_POOL) as pool:
        sizes = list(pool.map(_frame_flash_sizes, opened_closed))
    return np.concatenate(sizes)
