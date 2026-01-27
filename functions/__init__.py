# Try to import CUDA-related functions, but don't fail if dependencies are missing
try:
    from .check_cuda import check_cuda
    from .gpu_detrend import gpu_detrend_jitted
    from .gpu_gauss import gpu_gaussian_blur
    from .gpu_process import process_on_gpu
    from .test_cuda import test_cuda

    CUDA_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    import sys

    print(f"Warning: Failed to import GPU functions: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    CUDA_AVAILABLE = False
    check_cuda = None
    gpu_detrend_jitted = None
    gpu_gaussian_blur = None
    process_on_gpu = None
    test_cuda = None

from .cpu_detrend import cpu_detrend_jitted
from .cpu_gauss import cpu_gaussian_blur
from .cpu_process import process_on_cpu
from .get_memory_use import get_memory_usage
from .imaging_segments_zscore_normalization import img_seg_zscore_norm
from .spike_centered_processes import spike_centered_avg, spike_centered_median

__all__ = [
    "CUDA_AVAILABLE",
    "check_cuda",
    "cpu_detrend_jitted",
    "cpu_gaussian_blur",
    "get_memory_usage",
    "gpu_detrend_jitted",
    "gpu_gaussian_blur",
    "img_seg_zscore_norm",
    "process_on_cpu",
    "process_on_gpu",
    "spike_centered_avg",
    "spike_centered_median",
    "test_cuda",
]
