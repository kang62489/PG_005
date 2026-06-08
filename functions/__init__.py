"""Lazy imports — heavy functions (numba, scipy, openpyxl) load only when first accessed."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .als import als_run
    from .check_cuda import check_cuda
    from .detrend import biexp_detrend, mov_detrend
    from .gaussian_blur import gaussian_blur_run
    from .get_memory_use import get_memory_usage
    from .imaging_segments_zscore_normalization import img_seg_zscore_norm
    from .spike_centered_processes import spike_centered_avg, spike_centered_median
    from .tau_estimate import sample_tau
    from .test_cuda import test_cuda
    from .xlsx_reader import get_picked_pairs

__all__ = [
    "als_run",
    "biexp_detrend",
    "check_cuda",
    "gaussian_blur_run",
    "get_memory_usage",
    "get_picked_pairs",
    "img_seg_zscore_norm",
    "mov_detrend",
    "sample_tau",
    "spike_centered_avg",
    "spike_centered_median",
    "test_cuda",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "als_run":        (".als",                                      "als_run"),
    "biexp_detrend":       (".detrend",                                   "biexp_detrend"),
    "mov_detrend":         (".detrend",                                   "mov_detrend"),
    "gaussian_blur_run":   (".gaussian_blur",                             "gaussian_blur_run"),
    "get_memory_usage":    (".get_memory_use",                            "get_memory_usage"),
    "img_seg_zscore_norm": (".imaging_segments_zscore_normalization",     "img_seg_zscore_norm"),
    "spike_centered_avg":  (".spike_centered_processes",                  "spike_centered_avg"),
    "spike_centered_median": (".spike_centered_processes",                "spike_centered_median"),
    "sample_tau":          (".tau_estimate",                              "sample_tau"),
    "get_picked_pairs":    (".xlsx_reader",                               "get_picked_pairs"),
    "check_cuda":          (".check_cuda",                                "check_cuda"),
    "test_cuda":           (".test_cuda",                                 "test_cuda"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        value = getattr(module, attr)
        globals()[name] = value  # cache so __getattr__ is only called once
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
