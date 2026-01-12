# Try to import CUDA-related functions, but don't fail if dependencies are missing
try:
    from .check_cuda import check_cuda
    from .GPU_detrend import gpu_detrend_jitted
    from .GPU_gauss import gpu_gaussian_blur
    from .GPU_process import process_on_gpu
    from .test_cuda import test_cuda
    CUDA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUDA_AVAILABLE = False
    check_cuda = None
    gpu_detrend_jitted = None
    gpu_gaussian_blur = None
    process_on_gpu = None
    test_cuda = None

from .CPU_detrend import cpu_detrend_jitted
from .CPU_gauss import cpu_gaussian_blur
from .CPU_process import process_on_cpu
from .get_memory_use import get_memory_usage

# Spatial categorization functions (new)
from .spatial_categorization import (
    categorize_spatial_connected,
    categorize_spatial_morphological,
    process_segment_spatial,
    quick_spatial_categorize,
)

__all__ = [
    "check_cuda",
    "cpu_detrend_jitted",
    "cpu_gaussian_blur",
    "get_memory_usage",
    "gpu_detrend_jitted",
    "gpu_gaussian_blur",
    "process_on_cpu",
    "process_on_gpu",
    "test_cuda",
    "CUDA_AVAILABLE",
    # Spatial categorization
    "categorize_spatial_connected",
    "categorize_spatial_morphological",
    "process_segment_spatial",
    "quick_spatial_categorize",
]
