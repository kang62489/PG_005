"""Lazy imports — heavy functions (numba, scipy, openpyxl) load only when first accessed."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .als import als_run
    from .check_cuda import check_cuda
    from .detrend import biexp_detrend, mov_detrend
    from .file_status import (
        abf_ready,
        als_mode,
        als_ready,
        build_filename_index,
        build_proc_file_index,
        gauss_mode,
        gauss_ready,
        raw_tiff_ready,
    )
    from .gaussian_blur import gaussian_blur_run
    from .get_memory_use import get_memory_usage
    from .spike_centered_processes import spike_centered_avg, spike_centered_median
    from .tau_estimate import sample_tau
    from .test_cuda import test_cuda
    from .xlsx_reader import get_picked_pairs
    from .zscore_img_segs import zscore_img_segs

__all__ = [
    "abf_ready",
    "als_mode",
    "als_ready",
    "als_run",
    "biexp_detrend",
    "build_filename_index",
    "build_proc_file_index",
    "check_cuda",
    "gauss_mode",
    "gauss_ready",
    "gaussian_blur_run",
    "get_memory_usage",
    "get_picked_pairs",
    "zscore_img_segs",
    "mov_detrend",
    "raw_tiff_ready",
    "sample_tau",
    "spike_centered_avg",
    "spike_centered_median",
    "test_cuda",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "als_run":        (".als",                                      "als_run"),
    "biexp_detrend":       (".detrend",                                   "biexp_detrend"),
    "mov_detrend":         (".detrend",                                   "mov_detrend"),
    "abf_ready":           (".file_status",                               "abf_ready"),
    "als_mode":            (".file_status",                               "als_mode"),
    "als_ready":           (".file_status",                               "als_ready"),
    "build_filename_index": (".file_status",                              "build_filename_index"),
    "build_proc_file_index": (".file_status",                             "build_proc_file_index"),
    "gauss_mode":          (".file_status",                               "gauss_mode"),
    "gauss_ready":         (".file_status",                               "gauss_ready"),
    "raw_tiff_ready":      (".file_status",                               "raw_tiff_ready"),
    "gaussian_blur_run":   (".gaussian_blur",                             "gaussian_blur_run"),
    "get_memory_usage":    (".get_memory_use",                            "get_memory_usage"),
    "zscore_img_segs":     (".zscore_img_segs",                           "zscore_img_segs"),
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
