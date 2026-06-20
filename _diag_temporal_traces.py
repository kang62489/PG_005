"""Temporary diagnostic: plot bright/dim/total temporal traces for visual check."""

import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from classes.region_analyzer import RegionAnalyzer

RESULTS_DIR = Path("results")
STEM = "2025_12_15-0025_BIEXP_GAUSS"
SPIKE_FRAME_IDX = 20

cat_segment = tifffile.imread(RESULTS_DIR / f"{STEM}_CAT.tif")
median_segment = tifffile.imread(RESULTS_DIR / f"{STEM}_MED.tif")

spike_frame = cat_segment[SPIKE_FRAME_IDX]
analyzer = RegionAnalyzer(spike_frame, obj="10X")
traces = analyzer.get_temporal_traces(median_segment)

n_frames = median_segment.shape[0]
x = np.arange(n_frames)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, traces["bright_trace"], label="bright_trace", color="orange")
ax.plot(x, traces["dim_trace"], label="dim_trace", color="dodgerblue")
ax.plot(x, traces["total_trace"], label="total_trace", color="black", linestyle="--")
ax.axvline(SPIKE_FRAME_IDX, color="red", linestyle=":", label="spike frame")
ax.set_xlabel("Frame index")
ax.set_ylabel("Mean z-score")
ax.set_title(f"{STEM} — temporal traces")
ax.legend()

out_path = RESULTS_DIR / f"{STEM}_TRACES_CHECK.png"
fig.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}")
