"""
run_als_baseline.py  --  Per-pixel ALS baseline estimation
===========================================================
Pipeline per file:
  Input : *_BIEXP_GAUSS.tif  (T x H x W, uint16)
  1. ALS pixel-wise  ->  slowly-varying F0 per pixel
  2. Save *_BIEXP_BASELINE.tif  (float32)
  3. Compute dF/F0 = (Gauss - Baseline) / Baseline
  4. Save *_BIEXP_DFF0.tif  (float32)

GPU path  : one Numba CUDA thread per pixel, Thomas algorithm per ALS iteration.
CPU path  : same Thomas algorithm vectorised across all pixels with NumPy.
"""

import time
from pathlib import Path

import numpy as np
import tifffile
from numba import cuda

# ── Config ────────────────────────────────────────────────
PROC_DIR = Path(__file__).parent.parent / "proc_tiffs"

FILES = [
    # "2026_03_20-0028_BIEXP_GAUSS.tif",
    # "2026_03_20-0029_BIEXP_GAUSS.tif",
    "2026_05_15-0050.tif",
    # "2026_03_20-0041_BIEXP_GAUSS.tif",
    # "2025_06_11-0002_BIEXP_GAUSS.tif",
    # "2025_06_11-0003_BIEXP_GAUSS.tif",
]

LAM = 1e2    # smoothness: larger → smoother baseline
P = 0.05    # asymmetry: small → baseline hugs the lower envelope
N_ITER = 10  # ALS iterations (10 is usually sufficient)
MAX_LEN = 2048  # compile-time local-array size; must be >= max(T, H, W)
AXIS = "x"   # axis along which ALS is applied: "x" (width), "y" (height), "z" (time)


# ── GPU kernel ────────────────────────────────────────────


@cuda.jit
def _als_gpu_kernel(data, baseline, lam, p, n_iter) -> None:
    """ALS baseline per segment using Thomas algorithm (1st-order differences).

    data / baseline shape: (n_segments, seg_len).
    Each thread handles one segment independently.
    Local arrays are fixed at MAX_LEN = 2048 (must be >= seg_len).

    Axis reshaping is handled by the driver before calling this kernel:
      axis="z" (time) : (H*W, T)   — one segment per pixel over time
      axis="x" (width): (T*H, W)   — one segment per row per frame
      axis="y" (height): (T*W, H)  — one segment per column per frame

    System solved each iteration: (W + λ L'L) z = W y
      L'L  main diag : [1, 2, 2, ..., 2, 1]
      L'L  off  diag : -1  (constant)
    """
    px = cuda.grid(1)
    if px >= data.shape[0]:
        return

    T = data.shape[1]

    # Thread-local arrays (stored in GPU local/register memory)
    z = cuda.local.array(MAX_LEN, dtype=np.float32)   # current baseline estimate
    w = cuda.local.array(MAX_LEN, dtype=np.float32)   # ALS weights
    c = cuda.local.array(MAX_LEN, dtype=np.float32)   # Thomas: modified upper diagonal
    d = cuda.local.array(MAX_LEN, dtype=np.float32)   # Thomas: modified RHS

    for i in range(T):
        z[i] = data[px, i]
        w[i] = 1.0

    for _ in range(n_iter):
        # ── Forward sweep (Thomas algorithm) ─────────────────────────────
        # main[0]: L'L contributes 1 at endpoints, 2 elsewhere
        denom = w[0] + lam * 1.0
        c[0] = -lam / denom
        d[0] = w[0] * data[px, 0] / denom

        for i in range(1, T):
            lll_m = 1.0 if i == T - 1 else 2.0
            # denom = modified main = b[i] - a[i]*c'[i-1]
            #       = (w[i] + λ*lll_m) - (-λ)*c[i-1]
            #       = w[i] + λ*lll_m + λ*c[i-1]   (c[i-1] is negative)
            denom = w[i] + lam * lll_m + lam * c[i - 1]
            if i < T - 1:
                c[i] = -lam / denom
            # d'[i] = (d[i] - a[i]*d'[i-1]) / denom
            #       = (w[i]*y[i] - (-λ)*d[i-1]) / denom
            d[i] = (w[i] * data[px, i] + lam * d[i - 1]) / denom

        # ── Backward substitution ─────────────────────────────────────────
        z[T - 1] = d[T - 1]
        for i in range(T - 2, -1, -1):
            z[i] = d[i] - c[i] * z[i + 1]

        # ── Update weights ────────────────────────────────────────────────
        for i in range(T):
            w[i] = p if data[px, i] > z[i] else (1.0 - p)

    for i in range(T):
        baseline[px, i] = z[i]


# ── Reshape helpers ───────────────────────────────────────


def _reshape_for_als(gauss: np.ndarray, axis: str) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Reshape (T, H, W) into (n_segments, seg_len) for ALS along the given axis.

    Returns:
        data_2d  : contiguous float32 array of shape (n_segments, seg_len)
        orig_shape: (T, H, W) for use by the reshape-back helper
    """
    T, H, W = gauss.shape
    orig_shape = (T, H, W)

    if axis == "z":
        # Segment along time: one segment per pixel → (H*W, T)
        data_2d = np.ascontiguousarray(gauss.reshape(T, H * W).T, dtype=np.float32)
    elif axis == "x":
        # Segment along width: one segment per (frame, row) → (T*H, W)
        data_2d = np.ascontiguousarray(gauss.reshape(T * H, W), dtype=np.float32)
    elif axis == "y":
        # Segment along height: one segment per (frame, col) → (T*W, H)
        # transpose to (T, W, H) first so height is the last (contiguous) dim
        data_2d = np.ascontiguousarray(gauss.transpose(0, 2, 1).reshape(T * W, H), dtype=np.float32)
    else:
        msg = f"axis must be 'x', 'y', or 'z'; got {axis!r}"
        raise ValueError(msg)

    return data_2d, orig_shape


def _reshape_back(baseline_2d: np.ndarray, orig_shape: tuple[int, int, int], axis: str) -> np.ndarray:
    """Inverse of _reshape_for_als: map (n_segments, seg_len) back to (T, H, W)."""
    T, H, W = orig_shape

    if axis == "z":
        return baseline_2d.T.reshape(T, H, W)
    if axis == "x":
        return baseline_2d.reshape(T, H, W)
    # axis == "y": undo transpose(0, 2, 1) by transposing again
    return baseline_2d.reshape(T, W, H).transpose(0, 2, 1)


# ── GPU driver ────────────────────────────────────────────


def gpu_als_baseline(
    gauss: np.ndarray,
    lam: float = LAM,
    p: float = P,
    n_iter: int = N_ITER,
    axis: str = AXIS,
) -> np.ndarray:
    """Run ALS on GPU along the specified axis. Input shape: (T, H, W) float32.

    axis="x" : smooth along image width  (W dimension)
    axis="y" : smooth along image height (H dimension)
    axis="z" : smooth along time         (T dimension, classic per-pixel baseline)
    """
    T, H, W = gauss.shape
    seg_len = {"x": W, "y": H, "z": T}[axis]
    if seg_len > MAX_LEN:
        msg = f"seg_len={seg_len} (axis={axis!r}) exceeds MAX_LEN={MAX_LEN}; increase MAX_LEN and recompile"
        raise ValueError(msg)

    data_2d, orig_shape = _reshape_for_als(gauss, axis)  # (n_segs, seg_len)
    n_segs = data_2d.shape[0]

    data_gpu = cuda.to_device(data_2d)
    baseline_gpu = cuda.device_array_like(data_2d)

    threads = 128
    blocks = (n_segs + threads - 1) // threads
    _als_gpu_kernel[blocks, threads](data_gpu, baseline_gpu, float(lam), float(p), int(n_iter))
    cuda.synchronize()

    baseline_2d = baseline_gpu.copy_to_host()           # (n_segs, seg_len)
    return _reshape_back(baseline_2d, orig_shape, axis)  # (T, H, W)


# ── CPU driver ────────────────────────────────────────────


def cpu_als_baseline(
    gauss: np.ndarray,
    lam: float = LAM,
    p: float = P,
    n_iter: int = N_ITER,
    axis: str = AXIS,
) -> np.ndarray:
    """Run ALS on CPU (Thomas algorithm vectorised across segments).

    axis="x" : smooth along image width  (W dimension)
    axis="y" : smooth along image height (H dimension)
    axis="z" : smooth along time         (T dimension, classic per-pixel baseline)

    The only Python loop is over seg_len (the ALS dimension). All segments
    are processed simultaneously via NumPy broadcasting.
    """
    data_2d, orig_shape = _reshape_for_als(gauss, axis)   # (n_segs, seg_len), float32
    y = data_2d.T.astype(np.float64)                       # (seg_len, n_segs) for easy indexing

    seg_len = y.shape[0]
    lll_m = np.full(seg_len, 2.0)
    lll_m[0] = lll_m[-1] = 1.0

    z = y.copy()
    w = np.ones_like(y)
    c = np.empty_like(y)
    d = np.empty_like(y)

    for _ in range(n_iter):
        # Forward sweep — all segments in parallel via numpy broadcast
        denom = w[0] + lam * lll_m[0]          # (n_segs,)
        c[0] = -lam / denom
        d[0] = w[0] * y[0] / denom

        for i in range(1, seg_len):
            denom = w[i] + lam * lll_m[i] + lam * c[i - 1]
            if i < seg_len - 1:
                c[i] = -lam / denom
            d[i] = (w[i] * y[i] + lam * d[i - 1]) / denom

        # Backward substitution
        z[-1] = d[-1]
        for i in range(seg_len - 2, -1, -1):
            z[i] = d[i] - c[i] * z[i + 1]

        # Update weights
        w = np.where(y > z, p, 1.0 - p)

    baseline_2d = z.T.astype(np.float32)                  # (n_segs, seg_len)
    return _reshape_back(baseline_2d, orig_shape, axis)    # (T, H, W)


# ── Main ──────────────────────────────────────────────────

USE_GPU = cuda.is_available()
print("GPU detected — using CUDA kernel" if USE_GPU else "No GPU — using CPU fallback")
print(f"ALS axis: {AXIS!r}  (smoothing along {'width W' if AXIS == 'x' else 'height H' if AXIS == 'y' else 'time T'})")

for fname in FILES:
    fpath = PROC_DIR / fname
    if not fpath.exists():
        print(f"\n[SKIP] {fname} not found in {PROC_DIR}")
        continue

    stem = fpath.stem.replace("_BIEXP_GAUSS", "")
    t0 = time.time()
    print(f"\n{'=' * 60}\n  {fname}")

    gauss = tifffile.imread(fpath).astype(np.float32)
    T, H, W = gauss.shape
    print(f"  Shape: ({T}, {H}, {W})   loaded {time.time() - t0:.1f}s")

    # ── ALS baseline ─────────────────────────────────────
    print(f"  Running ALS  (lam={LAM:.0e}, p={P}, n_iter={N_ITER}, axis={AXIS!r})...")
    baseline = gpu_als_baseline(gauss, axis=AXIS) if USE_GPU else cpu_als_baseline(gauss, axis=AXIS)
    print(f"  ALS done  ({time.time() - t0:.1f}s)")

    # ── Save baseline ─────────────────────────────────────
    tifffile.imwrite(PROC_DIR / f"{stem}_BIEXP_BASELINE.tif", baseline.astype(np.float16))
    print(f"  Saved  -> {stem}_BIEXP_BASELINE.tif")

    # ── Compute and save dF/F0 ───────────────────────────
    # f0 = np.maximum(baseline, 1.0)          # guard against zero baseline
    dff0 = ((gauss - baseline) / baseline).astype(np.float16)
    tifffile.imwrite(PROC_DIR / f"{stem}_BIEXP_DFF0.tif", dff0)
    print(f"  Saved  -> {stem}_BIEXP_DFF0.tif   ({time.time() - t0:.1f}s total)")

    del gauss, baseline, dff0

print(f"\n{'=' * 60}\nAll done!")
