"""Lazy imports — heavy functions (numba, scipy, openpyxl) load only when first accessed."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .als import als_run
    from .check_cuda import check_cuda
    from .database_ops import (
        compute_region_stats,
        count_unique_cells,
        get_cell_recording_status,
        get_excluded_recordings,
        lookup_rec_from_db,
        populate_animal_id_values,
    )
    from .detrend import biexp_detrend
    from .file_status import (
        abf_ready,
        als_exists,
        als_ready,
        build_filename_index,
        build_proc_file_index,
        gauss_exists,
        gauss_ready,
        raw_tiff_ready,
    )
    from .gaussian_blur import gaussian_blur_run
    from .get_memory_use import get_memory_usage
    from .list_parser import list_parser
    from .plot_results import plot_full_trace, plot_spatiotemporal_summary
    from .spike_alignment import spike_centered_avg, spike_centered_median
    from .tau_estimate import sample_tau
    from .test_cuda import test_cuda
    from .xlsx_reader import get_picked_pairs
    from .xlsx_writer import write_cell_summary_xlsx
    from .zscore_img_segs import zscore_img_segs

__all__ = [
    "abf_ready",
    "als_exists",
    "als_ready",
    "als_run",
    "biexp_detrend",
    "build_filename_index",
    "build_proc_file_index",
    "check_cuda",
    "compute_region_stats",
    "count_unique_cells",
    "gauss_exists",
    "gauss_ready",
    "gaussian_blur_run",
    "get_cell_recording_status",
    "get_excluded_recordings",
    "get_memory_usage",
    "get_picked_pairs",
    "list_parser",
    "lookup_rec_from_db",
    "zscore_img_segs",
    "plot_full_trace",
    "plot_spatiotemporal_summary",
    "populate_animal_id_values",
    "raw_tiff_ready",
    "sample_tau",
    "spike_centered_avg",
    "spike_centered_median",
    "test_cuda",
    "write_cell_summary_xlsx",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "als_run":        (".als",                                      "als_run"),
    "biexp_detrend":       (".detrend",                                   "biexp_detrend"),
    "abf_ready":           (".file_status",                               "abf_ready"),
    "als_exists":          (".file_status",                               "als_exists"),
    "als_ready":           (".file_status",                               "als_ready"),
    "build_filename_index": (".file_status",                              "build_filename_index"),
    "build_proc_file_index": (".file_status",                             "build_proc_file_index"),
    "gauss_exists":        (".file_status",                               "gauss_exists"),
    "gauss_ready":         (".file_status",                               "gauss_ready"),
    "raw_tiff_ready":      (".file_status",                               "raw_tiff_ready"),
    "gaussian_blur_run":   (".gaussian_blur",                             "gaussian_blur_run"),
    "get_memory_usage":    (".get_memory_use",                            "get_memory_usage"),
    "list_parser":         (".list_parser",                               "list_parser"),
    "plot_spatiotemporal_summary": (".plot_results",                      "plot_spatiotemporal_summary"),
    "plot_full_trace":     (".plot_results",                              "plot_full_trace"),
    "lookup_rec_from_db":  (".database_ops",                           "lookup_rec_from_db"),
    "populate_animal_id_values": (".database_ops",                     "populate_animal_id_values"),
    "count_unique_cells":  (".database_ops",                           "count_unique_cells"),
    "compute_region_stats": (".database_ops",                          "compute_region_stats"),
    "get_excluded_recordings": (".database_ops",                       "get_excluded_recordings"),
    "get_cell_recording_status": (".database_ops",                     "get_cell_recording_status"),
    "zscore_img_segs":     (".zscore_img_segs",                           "zscore_img_segs"),
    "spike_centered_avg":  (".spike_alignment",                           "spike_centered_avg"),
    "spike_centered_median": (".spike_alignment",                         "spike_centered_median"),
    "sample_tau":          (".tau_estimate",                              "sample_tau"),
    "get_picked_pairs":    (".xlsx_reader",                               "get_picked_pairs"),
    "write_cell_summary_xlsx": (".xlsx_writer",                           "write_cell_summary_xlsx"),
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
