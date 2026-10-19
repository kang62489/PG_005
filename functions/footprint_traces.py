"""
Footprint traces of a float16 z-stack, CPU Numba JIT + CUDA GPU (SREAnalyzer.group_trace).

  Step 1. Traces : mean over each flash's own footprint pixels, in every frame

Example:
    >>> traces = footprint_traces(footprints, stack_f16, cuda_available=True)   # (n_flashes, n_frames)
"""

## Modules
# Third-party imports
import numpy as np
from numba import cuda, njit, prange

# Local imports
from .flash_mask import f16_lut

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

TRACE_THREADS = 256           # GPU threads per (flash, frame) block of the flash traces


# ===========================================================================
#
#   TRACES
#
# ===========================================================================

# fn: _cpu_footprint_traces, CPU: one frame per thread.
# fn: _gpu_footprint_traces_kernel, GPU: one block per detection x frame, shared-memory reduction.
# fn: _gpu_footprint_traces
# fn: footprint_traces, (n_detections, n_frames) float16: mean over each detection's own footprint, every frame.

@njit(parallel=True)
def _cpu_footprint_traces(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                          starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """CPU: one frame per thread."""
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


@cuda.jit
def _gpu_footprint_traces_kernel(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                                 starts: np.ndarray, counts: np.ndarray, means: np.ndarray) -> None:
    """GPU: one block per detection x frame, shared-memory reduction."""
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
