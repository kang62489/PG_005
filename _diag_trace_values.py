"""Temporary diagnostic: print raw temporal trace values for inspection."""

from pathlib import Path

import tifffile

from classes.region_analyzer import RegionAnalyzer

RESULTS_DIR = Path("results")
STEM = "2025_12_15-0013_BIEXP_GAUSS"
SPIKE_FRAME_IDX = 20

cat_segment = tifffile.imread(RESULTS_DIR / f"{STEM}_CAT.tif")
median_segment = tifffile.imread(RESULTS_DIR / f"{STEM}_MED.tif")

print("median_segment dtype:", median_segment.dtype)
print("cat_segment dtype:", cat_segment.dtype)

analyzer = RegionAnalyzer(cat_segment[SPIKE_FRAME_IDX], obj="10X")
print("bright_largest area_px:", analyzer.bright_largest["area_px"] if analyzer.bright_largest else None)
print("dim_largest area_px:", analyzer.dim_largest["area_px"] if analyzer.dim_largest else None)

traces = analyzer.get_temporal_traces(median_segment)
print("bright_trace dtype:", traces["bright_trace"].dtype)

for offset in range(-3, 4):
    idx = SPIKE_FRAME_IDX + offset
    bright_val = traces["bright_trace"][idx]
    dim_val = traces["dim_trace"][idx]
    frame_regions = analyzer.find_largest_regions(cat_segment[idx])
    bright_detected = frame_regions["bright"] is not None
    print(f"offset={offset:+d} bright_trace={bright_val!r} dim_trace={dim_val!r} bright_detected_this_frame={bright_detected}")
