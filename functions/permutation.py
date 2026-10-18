"""
Surrogates for the permutation test (docs/permutation_test.md), CPU Numba JIT + CUDA GPU.

  Step 1. Lags             : one random circular time lag per pixel (seeded)
  Step 2. Surrogate masks  : z-stack -> every trace rotated by its lag -> Gaussian blur -> ALS -> threshold ->
                             3x3 opening -> 3x3 closing = one sur_flash mask per lag map
                             (= img_proc.py blur + als_correct.py ALS + flash_mask on a time-shuffled stack;
                             fill holes + sizes: sre_analyzer.mask_flash_sizes, on the CPU)

GPU: the z-stack is uploaded once, pixel-major (H, W, T), so rotating a pixel's trace is one contiguous read; each
band of rows (+ blur halo) is rotated, blurred, ALS-corrected and thresholded on the GPU and only the 1-byte mask
comes back. The threshold test reproduces the pipeline's float16 rounding of the ALS stack. The input is float16
(the pipeline blurs float32) and the blurred band stays float32 (the pipeline stores it as float16): < 0.1 %.

Example:
    >>> lag_maps = (pixel_lags(z_f16.shape, seed) for seed in range(1000))
    >>> for mask in sur_flash_masks(z_f16, lag_maps, threshold, 4.0, 11, 0.05, 10, cuda_available=True):
    ...     sizes = mask_flash_sizes(mask)
"""

## Modules
# Standard library imports
import math
import os
import warnings
from collections.abc import Iterable, Iterator

# Third-party imports
import numpy as np
from numba import cuda, njit, prange
from numba.core.errors import NumbaPerformanceWarning

# Local imports
from classes.sre_analyzer import _gpu_dilate_kernel, _gpu_erode_kernel, _threshold_open_close, f16_lut

from .als import MAX_T, _cpu_als_kernel, _gpu_als_kernel
from .gaussian_blur import _cpu_kernel, _gpu_reflect_index, _reflect_index

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
os.environ.setdefault("NUMBA_CUDA_LOG_LEVEL", "40")


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 2: surrogate masks -----------------------------------------------
BAND_ROWS = 64                # output rows per band (+ blur halo above / below)
TILE = 32                     # lag kernel: 32 frames x 32 columns per block, transposed through shared memory
TILE_PAD = 33                 # TILE + 1: no shared-memory bank conflicts on the transposed read
THREADS_3D = (32, 8, 1)       # threads per block of the (x, y, t) kernels


# ===========================================================================
#
#   STEP 1 -- LAGS
#
# ===========================================================================

def pixel_lags(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    """Step 1: lag of every pixel, uniform in [0, T)."""
    n_t, h, w = shape
    return np.random.default_rng(seed).integers(0, n_t, size=(h, w), dtype=np.int32)


# ===========================================================================
#
#   STEP 2 -- SURROGATE MASKS
#
# ===========================================================================

def _f16_cut(threshold: float) -> tuple[np.float32, bool]:
    """float16(v) > float16(threshold)  <=>  v > cut (or v == cut if that tie rounds up, round-half-to-even)."""
    thr16 = np.float16(threshold)
    above = np.nextafter(thr16, np.float16(np.inf))
    cut = np.float32((np.float64(thr16) + np.float64(above)) / 2)  # 12 significant bits: exact in float32
    return cut, bool(above.view(np.uint16) % 2 == 0)


# --- 2a. GPU (bands of rows, one thread per pixel / frame) ------------------

@cuda.jit
def _gpu_lag_band(codes_px, lut, lags, g0, out) -> None:
    """out[t, y, x] = z[(t + lag) % T] of pixel (g0 + y, x); reads along t and writes along x, both coalesced."""
    tile = cuda.shared.array((TILE, TILE_PAD), dtype=np.float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x0, t0, y = cuda.blockIdx.x * TILE, cuda.blockIdx.y * TILE, cuda.blockIdx.z
    n_t, _, w = out.shape
    row = g0 + y
    for j in range(ty, TILE, cuda.blockDim.y):  # lanes = frames of one pixel
        x, t = x0 + j, t0 + tx
        if x < w and t < n_t:
            src_t = t + lags[row, x]
            if src_t >= n_t:
                src_t -= n_t
            tile[j, tx] = lut[codes_px[row, x, src_t]]
    cuda.syncthreads()
    for j in range(ty, TILE, cuda.blockDim.y):  # lanes = pixels of one frame
        x, t = x0 + tx, t0 + j
        if x < w and t < n_t:
            out[t, y, x] = tile[tx, j]


@cuda.jit
def _gpu_hblur(src, kernel, out) -> None:
    """out[t, y, x] = horizontal blur of src, reflect at the frame edge."""
    x, y, t = cuda.grid(3)
    n_t, n_rows, w = out.shape
    if t >= n_t or y >= n_rows or x >= w:
        return
    half = kernel.shape[0] // 2
    val = 0.0
    for kp in range(kernel.shape[0]):
        val += src[t, y, _gpu_reflect_index(x + kp - half, w)] * kernel[kp]
    out[t, y, x] = val


@cuda.jit
def _gpu_vblur(src, kernel, g0, r0, height, out) -> None:
    """out[t, y, x] = vertical blur of frame row r0 + y, read from src (rows g0 ..), reflect at the frame edge."""
    x, y, t = cuda.grid(3)
    n_t, n_rows, w = out.shape
    if t >= n_t or y >= n_rows or x >= w:
        return
    half = kernel.shape[0] // 2
    val = 0.0
    for kp in range(kernel.shape[0]):
        val += src[t, _gpu_reflect_index(r0 + y + kp - half, height) - g0, x] * kernel[kp]
    out[t, y, x] = val


@cuda.jit
def _gpu_threshold_band(data, baseline, cut, tie_up, r0, mask) -> None:
    """mask[t, r0 + y, x] = float16(data - baseline) > float16(threshold), band pixel px = y * W + x."""
    px, t = cuda.grid(2)
    if t >= data.shape[0] or px >= data.shape[1]:
        return
    v = data[t, px] - baseline[t, px]
    w = mask.shape[2]
    mask[t, r0 + px // w, px % w] = 1 if v > cut or (tie_up and v == cut) else 0


def _gpu_sur_flash_masks(z_f16: np.ndarray, lag_maps: Iterable[np.ndarray], threshold: float, kernel: np.ndarray,
                         lam: float, p: float, n_iter: int) -> Iterator[np.ndarray]:
    """z-stack uploaded once; per lag map: bands rotated -> blurred -> ALS -> thresholded, then open / close."""
    n_t, h, w = z_f16.shape
    half = len(kernel) // 2
    cut, tie_up = _f16_cut(threshold)
    cuda.current_context().deallocations.clear()
    d_codes = cuda.to_device(np.ascontiguousarray(z_f16.view(np.uint16).transpose(1, 2, 0)))  # (H, W, T)
    d_lut, d_kernel = cuda.to_device(f16_lut()), cuda.to_device(kernel)
    d_a = cuda.device_array((n_t, h, w), dtype=np.uint8)
    d_b = cuda.device_array((n_t, h, w), dtype=np.uint8)
    blocks_full = (math.ceil(w / 32), math.ceil(h / 8), n_t)
    for lags in lag_maps:
        d_lags = cuda.to_device(lags)
        for r0 in range(0, h, BAND_ROWS):
            r1 = min(r0 + BAND_ROWS, h)
            g0, g1 = max(r0 - half, 0), min(r1 + half, h)
            n_rows = r1 - r0
            d_lag = cuda.device_array((n_t, g1 - g0, w), dtype=np.float32)
            _gpu_lag_band[(math.ceil(w / TILE), math.ceil(n_t / TILE), g1 - g0), (TILE, 8)](
                d_codes, d_lut, d_lags, g0, d_lag)
            d_h = cuda.device_array_like(d_lag)
            _gpu_hblur[(math.ceil(w / 32), math.ceil((g1 - g0) / 8), n_t), THREADS_3D](d_lag, d_kernel, d_h)
            del d_lag
            d_blur = cuda.device_array((n_t, n_rows, w), dtype=np.float32)
            _gpu_vblur[(math.ceil(w / 32), math.ceil(n_rows / 8), n_t), THREADS_3D](d_h, d_kernel, g0, r0, h, d_blur)
            del d_h
            d_data = d_blur.reshape(n_t, n_rows * w)
            d_base = cuda.device_array_like(d_data)
            _gpu_als_kernel[math.ceil(n_rows * w / 128), 128](d_data, d_base, float(lam), float(p), int(n_iter))
            _gpu_threshold_band[(math.ceil(n_rows * w / 256), n_t), (256, 1)](d_data, d_base, cut, tie_up, r0, d_a)
            del d_blur, d_data, d_base
        _gpu_erode_kernel[blocks_full, THREADS_3D](d_a, d_b)  # opening
        _gpu_dilate_kernel[blocks_full, THREADS_3D](d_b, d_a)
        _gpu_dilate_kernel[blocks_full, THREADS_3D](d_a, d_b)  # closing
        _gpu_erode_kernel[blocks_full, THREADS_3D](d_b, d_a)
        cuda.synchronize()
        yield d_a.copy_to_host().view(np.bool_)


# --- 2b. CPU (numba, bands of rows, parallel over frames) ------------------

@njit(parallel=True, cache=True)
def _cpu_lag_hblur(codes: np.ndarray, lut: np.ndarray, lags: np.ndarray, kernel: np.ndarray, g0: int,
                   out: np.ndarray) -> None:
    """Lag + horizontal blur of rows g0 .., parallel over frames."""
    n_t, n_rows, w = out.shape
    half = kernel.shape[0] // 2
    for t in prange(n_t):
        for y in range(n_rows):
            row = g0 + y
            for x in range(w):
                val = 0.0
                for kp in range(kernel.shape[0]):
                    col = _reflect_index(x + kp - half, w)
                    src_t = t + lags[row, col]
                    if src_t >= n_t:
                        src_t -= n_t
                    val += lut[codes[src_t, row, col]] * kernel[kp]
                out[t, y, x] = val


@njit(parallel=True, cache=True)
def _cpu_vblur(src: np.ndarray, kernel: np.ndarray, g0: int, r0: int, height: int, out: np.ndarray) -> None:
    """Mirrors _gpu_vblur, parallel over frames."""
    n_t, n_rows, w = out.shape
    half = kernel.shape[0] // 2
    for t in prange(n_t):
        for y in range(n_rows):
            for x in range(w):
                val = 0.0
                for kp in range(kernel.shape[0]):
                    val += src[t, _reflect_index(r0 + y + kp - half, height) - g0, x] * kernel[kp]
                out[t, y, x] = val


def _cpu_sur_als(z_f16: np.ndarray, lags: np.ndarray, kernel: np.ndarray, lam: float, p: float,
                    n_iter: int) -> np.ndarray:
    """ALS-corrected stack (float16, as *_BIEXP_ALS.tif) of the lag-shuffled stack, band by band."""
    n_t, h, w = z_f16.shape
    half = len(kernel) // 2
    codes, lut = z_f16.view(np.uint16), f16_lut()
    out = np.empty((n_t, h, w), dtype=np.float16)
    for r0 in range(0, h, BAND_ROWS):
        r1 = min(r0 + BAND_ROWS, h)
        g0, g1 = max(r0 - half, 0), min(r1 + half, h)
        band_h = np.empty((n_t, g1 - g0, w), dtype=np.float32)
        _cpu_lag_hblur(codes, lut, lags, kernel, g0, band_h)
        blurred = np.empty((n_t, r1 - r0, w), dtype=np.float32)
        _cpu_vblur(band_h, kernel, g0, r0, h, blurred)
        del band_h
        data = blurred.reshape(n_t, -1)
        baseline = np.empty_like(data)
        _cpu_als_kernel(data, baseline, float(lam), float(p), int(n_iter))
        out[:, r0:r1] = (data - baseline).reshape(n_t, r1 - r0, w)
        del blurred, data, baseline
    return out


def _cpu_sur_flash_masks(z_f16: np.ndarray, lag_maps: Iterable[np.ndarray], threshold: float, kernel: np.ndarray,
                         lam: float, p: float, n_iter: int) -> Iterator[np.ndarray]:
    """Per lag map: the float16 ALS stack (band by band), then sre_analyzer's threshold -> open -> close."""
    for lags in lag_maps:
        yield _threshold_open_close(_cpu_sur_als(z_f16, lags, kernel, lam, p, n_iter), threshold, False)


# --- 2c. Dispatch ----------------------------------------------------------

def sur_flash_masks(z_f16: np.ndarray, lag_maps: Iterable[np.ndarray], threshold: float, sigma: float, lam: float,
                    p: float, n_iter: int, cuda_available: bool) -> Iterator[np.ndarray]:
    """Step 2: per lag map, the opened + closed threshold mask (T, H, W) of the shuffled -> blurred -> ALS stack."""
    if z_f16.shape[0] > MAX_T:
        msg = f"T={z_f16.shape[0]} exceeds MAX_T={MAX_T} of the ALS kernel"
        raise ValueError(msg)
    size = math.ceil(sigma * 6)
    kernel = _cpu_kernel(sigma, size + 1 - size % 2)  # same odd 6-sigma kernel as gaussian_blur_run
    if cuda_available:
        return _gpu_sur_flash_masks(z_f16, lag_maps, threshold, kernel, lam, p, n_iter)
    return _cpu_sur_flash_masks(z_f16, lag_maps, threshold, kernel, lam, p, n_iter)
