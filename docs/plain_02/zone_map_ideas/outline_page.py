# ruff: noqa: INP001, E402
"""
outline_page.py  --  Preview of Jeff's idea: a 2nd zone-map page with all zone outlines on a white background.

Steps
-----
1. Run SpontaneousZoneAnalyzer on each example recording (same settings as spontaneous_analysis.py).
2. Page 1 = current overview (plot_zone_overview, unchanged).
3. Page 2 = NEW: every recur_zone as a colored outline (same colors as page 1), NR zones gray dashed,
   striatum outline black dashed, on white.
4. Save both pages side by side as one PNG per recording.

Usage:
    .venv/Scripts/python.exe docs/plain_02/zone_map_ideas/outline_page.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from functions import check_cuda

_CUDA, _MSG = check_cuda()  # must run before anything imports numba

import numpy as np
import tifffile
from matplotlib.figure import Figure
from matplotlib.image import imsave

from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import CROSSOVER_RATIO
from functions import direction_labels, img_zscore_convert, load_st_bd, plot_zone_overview, zone_colors
from functions.plot_results import _label_zone
from spontaneous_analysis import MAP_Z_MIN, NR_COLOR, _figure_to_rgb, _gray_range, striatum_of

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEMS = ["2025_06_11-0008", "2025_06_11-0003", "2026_01_08-0012"]
STBD = ROOT / "data" / "bd_20260922_000.json"
OUT_DIR = Path(__file__).resolve().parent
NR_OUTLINE = (0.45, 0.45, 0.45)  # NR zones on white: darker gray than NR_COLOR, dashed
SCALE_UM = 200  # scale bar length (µm)


def outline_page(shape: tuple[int, int], masks: dict, centroids: dict, colors: dict, nr_ids: set, title: str,
                 um_per_px: float, striatum_outline: np.ndarray | None, axis_labels: tuple[str, str] | None) -> Figure:
    """All zone outlines on white: recur_zones in their page-1 color, NR zones gray dashed."""
    fig = Figure(figsize=(11, 11), layout="tight")
    ax = fig.add_subplot()
    ax.imshow(np.ones((*shape, 3)))  # white canvas with the image extent
    for zone_id, mask in masks.items():
        is_nr = zone_id in nr_ids
        ax.contour(mask.astype(float), levels=[0.5], colors=[NR_OUTLINE if is_nr else colors[zone_id]],
                   linewidths=1.5 if is_nr else 2.5, linestyles="--" if is_nr else "-")
    if striatum_outline is not None:
        closed = np.vstack([striatum_outline, striatum_outline[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", ls="--", lw=1.5)
        ax.set_xlim(-0.5, shape[1] - 0.5)
        ax.set_ylim(shape[0] - 0.5, -0.5)
    for zone_id in masks:
        if zone_id in centroids:
            _label_zone(ax, zone_id, centroids[zone_id])
    bar_px = SCALE_UM / um_per_px  # black scale bar, bottom right
    x1, y = shape[1] * 0.95, shape[0] * 0.95
    ax.plot([x1 - bar_px, x1], [y, y], color="black", lw=4)
    ax.text(x1 - bar_px / 2, y - shape[0] * 0.015, f"{SCALE_UM} µm", ha="center", va="bottom", fontsize=12)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0], fontsize=12)
        ax.set_ylabel(axis_labels[1], fontsize=12)
    return fig


def main() -> None:
    """Steps 1-4 for every stem."""
    print(_MSG)
    stbd = load_st_bd(STBD)["recordings"]
    for stem in STEMS:
        proc_path = ROOT / "proc_tiffs" / f"{stem}_BIEXP_ALS.tif"
        analyzer = SpontaneousZoneAnalyzer(tifffile.imread(proc_path), obj="10X", sigma_ratio=CROSSOVER_RATIO,
                                           cuda_available=_CUDA)
        analyzer.run()
        shape = (analyzer.height, analyzer.width)
        striatum = striatum_of(stbd, stem, shape)
        striatum_outline = np.array(stbd[stem]["striatum_outline_px"]) if striatum is not None else None
        axis_labels = direction_labels(stbd[stem]["dorsal"], stbd[stem]["medial"]) if stem in stbd else None

        # same masks / centroids / colors as export_zone_maps
        masks = {**analyzer.zone_masks, **analyzer.non_recur_masks}
        centroids = {**analyzer.zone_centroids, **analyzer.non_recur_centroids}
        colors = {**zone_colors(list(analyzer.zone_masks)), **dict.fromkeys(analyzer.non_recur_masks, NR_COLOR)}
        max_proj, _, vmax = _gray_range(analyzer)
        n_text = f"{len(analyzer.zone_masks)} recur_zones + {len(analyzer.non_recur_masks)} NR zones"

        page1 = _figure_to_rgb(plot_zone_overview(
            img_zscore_convert(max_proj.astype(np.float32), analyzer.bg_center, analyzer.bg_sigma), masks, centroids,
            colors, MAP_Z_MIN, vmax, f"{stem}  page 1 (current)\n{n_text}", analyzer.um_per_px, striatum_outline,
            axis_labels))
        page2 = _figure_to_rgb(outline_page(
            shape, masks, centroids, colors, set(analyzer.non_recur_masks),
            f"{stem}  page 2 (new): all outlines\n{n_text} (NR = gray dashed)", analyzer.um_per_px,
            striatum_outline, axis_labels))

        out_path = OUT_DIR / f"{stem}_outline_preview.png"
        imsave(out_path, np.hstack([page1, page2]))  # both pages 11 x 11 in at MAP_DPI -> same size
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
