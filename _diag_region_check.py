"""Temporary diagnostic: visualize RegionAnalyzer's bright/dim region detection
on the spike frame of an existing _CAT.tif, to sanity-check the merged-dim-region
logic without re-running the full pipeline. Safe to delete after use.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from classes.region_analyzer import DIM_SEARCH_MARGIN, RegionAnalyzer

cat_path = "results/2025_12_15-0013_BIEXP_GAUSS_CAT.tif"
data = tifffile.imread(cat_path)
spike_frame_idx = data.shape[0] // 2
spike_frame = data[spike_frame_idx]

ra = RegionAnalyzer(spike_frame, obj="10X")
results = ra.get_results()
bright = results["bright_largest"]
dim = results["dim_largest"]

fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
cmap_cat = ListedColormap(["black", "gray", "white"])
ax.imshow(spike_frame, cmap=cmap_cat, vmin=0, vmax=2)

if bright is not None:
    min_row, min_col, max_row, max_col = bright["_bbox"]
    x_span = max_col - min_col
    y_span = max_row - min_row

    # Green box: bright region's own bbox
    ax.add_patch(
        Rectangle(
            (min_col, min_row), x_span, y_span,
            edgecolor="lime", facecolor="none", linewidth=2, label="Bright bbox (green)",
        )
    )

    # Yellow box: dim search window
    win_min_row = min_row - DIM_SEARCH_MARGIN * y_span
    win_max_row = max_row + DIM_SEARCH_MARGIN * y_span
    win_min_col = min_col - DIM_SEARCH_MARGIN * x_span
    win_max_col = max_col + DIM_SEARCH_MARGIN * x_span
    ax.add_patch(
        Rectangle(
            (win_min_col, win_min_row), win_max_col - win_min_col, win_max_row - win_min_row,
            edgecolor="yellow", facecolor="none", linewidth=2, linestyle="--", label="Dim search window (yellow)",
        )
    )

    if bright["contour"] is not None:
        c = bright["contour"]
        ax.plot(c[:, 1], c[:, 0], color="magenta", linewidth=1.5, label="Bright contour")
    by, bx = bright["centroid"]
    ax.scatter(bx, by, c="red", s=60, marker="+", linewidths=2, label="Bright centroid")

if dim is not None:
    for i, c in enumerate(dim["contour"]):
        ax.plot(c[:, 1], c[:, 0], color="cyan", linewidth=1.2, label="Dim contour (merged)" if i == 0 else None)
    dy, dx = dim["centroid"]
    ax.scatter(dx, dy, c="blue", s=60, marker="x", linewidths=2, label="Dim (merged) centroid")

ax.set_xlim(0, spike_frame.shape[1])
ax.set_ylim(spike_frame.shape[0], 0)
ax.legend(loc="lower left", fontsize=8)
ax.axis("off")
ax.set_title(f"Spike frame {spike_frame_idx} — bright + merged dim region check")

out_path = "results/2025_12_15-0013_BIEXP_GAUSS_REGION_CHECK.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"bright: {bright}")
print(f"dim area_px={dim['area_px'] if dim else None}, n_contour_pieces={len(dim['contour']) if dim else 0}")
print(f"Saved diagnostic to {out_path}")
