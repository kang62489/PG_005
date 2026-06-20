"""Temporary diagnostic: build the full spatiotemporal summary export figure."""

import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path

import tifffile

from classes.plot_results import plot_spatiotemporal_summary
from classes.region_analyzer import RegionAnalyzer
from classes.spatial_categorization import SpatialCategorizer

RESULTS_DIR = Path("results")
STEM = "2025_12_15-0013_BIEXP_GAUSS"
SPIKE_FRAME_IDX = 20

cat_segment = tifffile.imread(RESULTS_DIR / f"{STEM}_CAT.tif")
median_segment = tifffile.imread(RESULTS_DIR / f"{STEM}_MED.tif")

categorizer = SpatialCategorizer.morphological()
categorizer.source_frames = [median_segment[i] for i in range(median_segment.shape[0])]
categorizer.categorized_frames = [cat_segment[i] for i in range(cat_segment.shape[0])]

analyzer = RegionAnalyzer(cat_segment[SPIKE_FRAME_IDX], obj="10X")

title_info = {
    "animal_id": "neoChAT-204",
    "slice": "1R",
    "at": "SITE_1",
    "obj": "10X",
    "tiff_serial": "0013",
    "abf_serial": "0013",
}

fig = plot_spatiotemporal_summary(
    categorizer=categorizer,
    region_analyzer=analyzer,
    median_segment=median_segment,
    spike_frame_idx=SPIKE_FRAME_IDX,
    frame_duration_ms=50.0,
    title_info=title_info,
)

out_path = RESULTS_DIR / f"{STEM}_SUMMARY_CHECK.png"
fig.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}")
