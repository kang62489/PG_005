"""Lazy imports — heavy classes (scipy, matplotlib, skimage) load only when first accessed."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abf_clip import AbfClip
    from .dialog_pick_list import DialogPickList
    from .helper_checkable_dropdown import CheckableDropdown
    from .model_from_dataframe import ModelFromDataFrame
    from .plot_results import PlotPeaks, PlotRegion, PlotSegs, PlotSpatialDist
    from .region_analyzer import RegionAnalyzer
    from .results_exporter import ResultsExporter
    from .spatial_categorization import SpatialCategorizer

__all__ = [
    "AbfClip",
    "CheckableDropdown",
    "DialogPickList",
    "ModelFromDataFrame",
    "PlotPeaks",
    "PlotRegion",
    "PlotSegs",
    "PlotSpatialDist",
    "RegionAnalyzer",
    "ResultsExporter",
    "SpatialCategorizer",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AbfClip": (".abf_clip", "AbfClip"),
    "DialogPickList": (".dialog_pick_list", "DialogPickList"),
    "CheckableDropdown": (".helper_checkable_dropdown", "CheckableDropdown"),
    "ModelFromDataFrame": (".model_from_dataframe", "ModelFromDataFrame"),
    "PlotPeaks": (".plot_results", "PlotPeaks"),
    "PlotRegion": (".plot_results", "PlotRegion"),
    "PlotSegs": (".plot_results", "PlotSegs"),
    "PlotSpatialDist": (".plot_results", "PlotSpatialDist"),
    "RegionAnalyzer": (".region_analyzer", "RegionAnalyzer"),
    "ResultsExporter": (".results_exporter", "ResultsExporter"),
    "SpatialCategorizer": (".spatial_categorization", "SpatialCategorizer"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __package__)
        value = getattr(module, attr)
        globals()[name] = value  # cache so __getattr__ is only called once
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
