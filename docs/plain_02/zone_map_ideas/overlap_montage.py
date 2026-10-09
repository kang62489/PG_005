# ruff: noqa: INP001, E402
"""
overlap_montage.py  --  Preview: one panel per recur_zone with the zones it overlaps (white background).

Steps
-----
1. Run SpontaneousZoneAnalyzer (same settings as spontaneous_analysis.py).
2. Per recur_zone i: set = {i} + every zone (recur or NR) sharing >= MIN_OVERLAP_PX pixels with it.
3. Identical sets -> one panel. Zones with no overlap -> all together in one "no overlap" panel.
4. Montage PNG per recording (outline colors = page-1 colors, NR gray dashed, striatum black dashed).

Usage:
    .venv/Scripts/python.exe docs/plain_02/zone_map_ideas/overlap_montage.py
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

from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import CROSSOVER_RATIO
from functions import load_st_bd, zone_colors
from functions.plot_results import _label_zone
from spontaneous_analysis import MAP_DPI, striatum_of

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEMS = ["2025_06_11-0008", "2025_06_11-0003", "2026_01_08-0012"]
STBD = ROOT / "data" / "bd_20260922_000.json"
OUT_DIR = Path(__file__).resolve().parent
MIN_OVERLAP_PX = 1  # zones sharing at least this many pixels count as overlapping
NR_OUTLINE = (0.45, 0.45, 0.45)  # NR zones: dark gray, dashed
N_COLS = 4


def overlap_sets(masks: dict, recur_ids: list) -> tuple[list[tuple], list]:
    """(unique overlap sets, recur_zones with no overlap); a set = sorted ids, the recur_zone itself first."""
    sets, seen, alone = [], set(), []
    for i in recur_ids:
        others = [j for j in masks if j != i and (masks[i] & masks[j]).sum() >= MIN_OVERLAP_PX]
        if not others:
            alone.append(i)
            continue
        key = frozenset([i, *others])
        if key not in seen:
            seen.add(key)
            sets.append((i, *others))
    return sets, alone


def draw_panel(ax, ids: list, masks: dict, centroids: dict, colors: dict, nr_ids: set, title: str,
               striatum_outline: np.ndarray | None, shape: tuple[int, int]) -> None:
    """Outlines of the given zones on white."""
    ax.imshow(np.ones((*shape, 3)))
    for zone_id in ids:
        is_nr = zone_id in nr_ids
        ax.contour(masks[zone_id].astype(float), levels=[0.5], colors=[NR_OUTLINE if is_nr else colors[zone_id]],
                   linewidths=1.2 if is_nr else 2, linestyles="--" if is_nr else "-")
    if striatum_outline is not None:
        closed = np.vstack([striatum_outline, striatum_outline[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", ls="--", lw=1)
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(shape[0] - 0.5, -0.5)
    for zone_id in ids:
        if zone_id in centroids:
            _label_zone(ax, zone_id, centroids[zone_id])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    """Steps 1-4 for every stem."""
    print(_MSG)
    stbd = load_st_bd(STBD)["recordings"]
    for stem in STEMS:
        analyzer = SpontaneousZoneAnalyzer(tifffile.imread(ROOT / "proc_tiffs" / f"{stem}_BIEXP_ALS.tif"), obj="10X",
                                           sigma_ratio=CROSSOVER_RATIO, cuda_available=_CUDA)
        analyzer.run()
        shape = (analyzer.height, analyzer.width)
        striatum_outline = (np.array(stbd[stem]["striatum_outline_px"])
                            if striatum_of(stbd, stem, shape) is not None else None)
        masks = {**analyzer.zone_masks, **analyzer.non_recur_masks}
        centroids = {**analyzer.zone_centroids, **analyzer.non_recur_centroids}
        colors = zone_colors(list(analyzer.zone_masks))
        nr_ids = set(analyzer.non_recur_masks)

        sets, alone = overlap_sets(masks, list(analyzer.zone_masks))
        n_sets_before = len(analyzer.zone_masks) - len(alone)
        print(f"{stem}: {len(analyzer.zone_masks)} recur_zones -> {n_sets_before} with overlap "
              f"-> {len(sets)} unique panels; {len(alone)} with no overlap {alone}")
        for s in sets:
            print(f"   recur_zone {s[0]}: overlaps {list(s[1:])}")

        panels = [(list(s), f"recur_zone {s[0]} + overlaps {', '.join(map(str, s[1:]))}") for s in sets]
        if alone:
            panels.append((alone, f"no overlap: {', '.join(map(str, alone))}"))
        n_rows = -(-len(panels) // N_COLS)
        fig = Figure(figsize=(4.5 * N_COLS, 4.8 * n_rows + 0.6), layout="constrained")
        fig.suptitle(f"{stem}: {len(analyzer.zone_masks)} recur_zones + {len(nr_ids)} NR zones "
                     f"(overlap = sharing >= {MIN_OVERLAP_PX} px; identical sets shown once)", fontsize=13)
        for k, (ids, title) in enumerate(panels):
            draw_panel(fig.add_subplot(n_rows, N_COLS, k + 1), ids, masks, centroids, colors, nr_ids, title,
                       striatum_outline, shape)
        out_path = OUT_DIR / f"{stem}_overlap_montage.png"
        fig.savefig(out_path, dpi=MAP_DPI)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
