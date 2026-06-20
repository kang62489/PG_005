"""Lazy imports — heavy classes (scipy, matplotlib, skimage) load only when first accessed."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abf_clip import AbfClip
    from .bk_worker import BackgroundWorker
    from .dialog_pick_list import DialogPickList
    from .helper_cell_dropdown import CellDropdownDelegate
    from .helper_checkable_dropdown import CheckableDropdown
    from .model_from_dataframe import ModelFromDataFrame
    from .mpl_canvas import MplCanvas
    from .region_analyzer import RegionAnalyzer
    from .results_exporter import ResultsExporter
    from .spatial_categorization import SpatialCategorizer

from .dialog_confirm import DialogConfirm
from .dialog_get_path import DialogGetFile, DialogGetPath

__all__ = [
    "AbfClip",
    "BackgroundWorker",
    "CellDropdownDelegate",
    "CheckableDropdown",
    "DialogConfirm",
    "DialogGetPath",
    "DialogGetFile",
    "DialogPickList",
    "ModelFromDataFrame",
    "MplCanvas",
    "RegionAnalyzer",
    "ResultsExporter",
    "SpatialCategorizer",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AbfClip": (".abf_clip", "AbfClip"),
    "BackgroundWorker": (".bk_worker", "BackgroundWorker"),
    "CellDropdownDelegate": (".helper_cell_dropdown", "CellDropdownDelegate"),
    "DialogPickList": (".dialog_pick_list", "DialogPickList"),
    "CheckableDropdown": (".helper_checkable_dropdown", "CheckableDropdown"),
    "ModelFromDataFrame": (".model_from_dataframe", "ModelFromDataFrame"),
    "MplCanvas": (".mpl_canvas", "MplCanvas"),
    "RegionAnalyzer": (".region_analyzer", "RegionAnalyzer"),
    "ResultsExporter": (".results_exporter", "ResultsExporter"),
    "SpatialCategorizer": (".spatial_categorization", "SpatialCategorizer"),
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
