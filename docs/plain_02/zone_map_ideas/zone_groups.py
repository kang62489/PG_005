# ruff: noqa: INP001, E402
"""
zone_groups.py  --  Zone-map idea: group recur_zones that share a centre, one panel per group.

Steps
-----
1. Run SpontaneousZoneAnalyzer (same settings as spontaneous_analysis.py); recur_zones only (NR zones ignored).
2. Per recur_zone: centroid = centre of mass of its mask; circle centred on the centroid, radius by MODE:
   "inner" = distance to the zone's nearest edge (largest circle inside the zone, file *_zone_groups.png),
   "far" = distance to the zone's farthest pixel (circle enclosing the zone, file *_zone_groups_far.png).
3. Sort zones by area. Loop: take the smallest remaining zone; its group = every remaining zone whose centroid
   lies inside its circle; draw the group's outlines largest first (bottom); remove the group; repeat.
4. Last two panels: the largest zone of every group together, without and with NR zones.

Usage:
    .venv/Scripts/python.exe docs/plain_02/zone_map_ideas/zone_groups.py [inner|far]   (default inner)
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
from matplotlib.patches import Circle
from scipy import ndimage

from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import CROSSOVER_RATIO
from functions import load_st_bd, zone_colors
from spontaneous_analysis import MAP_DPI, striatum_of

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEMS = ["2025_06_11-0008", "2025_06_11-0003", "2026_01_08-0012"]
STBD = ROOT / "data" / "bd_20260922_000.json"
OUT_DIR = Path(__file__).resolve().parent
NR_OUTLINE = (0.45, 0.45, 0.45)  # NR zones: dark gray, dashed
N_COLS = 6  # at most 6 panels per row
MODE = sys.argv[1] if len(sys.argv) > 1 else "inner"  # "inner" or "far" (see step 2)
CIRCLE_TEXT = {"inner": "inner (nearest edge)", "far": "enclosing (farthest pixel)"}


def centre_and_radius(mask: np.ndarray, mode: str) -> tuple[float, float, float]:
    """(row, col, radius) around the centre of mass.

    mode "inner": distance to the nearest pixel outside the zone (largest circle at the centroid inside it; 0 if outside).
    mode "far": distance to the zone's farthest pixel (smallest circle at the centroid enclosing the whole zone).
    """
    cy, cx = ndimage.center_of_mass(mask)
    if mode == "far":
        rows, cols = np.nonzero(mask)
        return cy, cx, float(np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2).max())
    inside = ndimage.distance_transform_edt(mask)
    return cy, cx, float(inside[int(round(cy)), int(round(cx))])


def group_zones(masks: dict, mode: str) -> list[tuple[int, list]]:
    """Step 3: [(smallest zone id, group ids sorted largest first)]."""
    info = {z: centre_and_radius(m, mode) for z, m in masks.items()}
    area = {z: int(m.sum()) for z, m in masks.items()}
    remaining = sorted(masks, key=area.get)
    groups = []
    while remaining:
        seed = remaining[0]
        sy, sx, r = info[seed]
        members = [z for z in remaining if (info[z][0] - sy) ** 2 + (info[z][1] - sx) ** 2 <= r**2 or z == seed]
        groups.append((seed, sorted(members, key=area.get, reverse=True)))
        remaining = [z for z in remaining if z not in members]
    return groups, info


def panel(ax, ids: list, masks: dict, colors: dict, title: str, shape: tuple[int, int], outline,
          nr_masks: dict | None = None, circle: tuple | None = None) -> None:
    """Outlines on white: recur_zones in their map color (largest first = bottom), optional NR zones + circle."""
    ax.imshow(np.ones((*shape, 3)))
    for z in ids:
        ax.contour(masks[z].astype(float), levels=[0.5], colors=[colors[z]], linewidths=2)
        cy, cx = ndimage.center_of_mass(masks[z])
        ax.text(cx, cy, str(z), color="white", fontsize=9, fontweight="bold", ha="center", va="center",
                bbox={"boxstyle": "circle", "fc": colors[z], "ec": "none"})
    for nr_id, m in (nr_masks or {}).items():
        ax.contour(m.astype(float), levels=[0.5], colors=[NR_OUTLINE], linewidths=1.2, linestyles="--")
        cy, cx = ndimage.center_of_mass(m)
        ax.text(cx, cy, nr_id, color=NR_OUTLINE, fontsize=8, ha="center", va="center")
    if circle is not None:
        ax.add_patch(Circle((circle[1], circle[0]), circle[2], fill=False, ec="black", ls=":", lw=1.5))
    if outline is not None:
        closed = np.vstack([outline, outline[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", ls="--", lw=1)
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(shape[0] - 0.5, -0.5)
    ax.set_title(title, fontsize=10)
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
        outline = (np.array(stbd[stem]["striatum_outline_px"]) if striatum_of(stbd, stem, shape) is not None
                   else None)
        masks = analyzer.zone_masks
        colors = zone_colors(list(masks))
        groups, info = group_zones(masks, MODE)
        print(f"{stem}: {len(masks)} recur_zones -> {len(groups)} groups: {[g for _, g in groups]}")

        n_panels = len(groups) + 2
        n_rows = -(-n_panels // N_COLS)
        n_cols = min(n_panels, N_COLS)
        fig = Figure(figsize=(4.5 * n_cols, 4.7 * n_rows + 0.6), layout="constrained")
        fig.suptitle(f"{stem}: {len(masks)} recur_zones -> {len(groups)} groups "
                     f"(dotted circle = the smallest zone's {CIRCLE_TEXT[MODE]} circle; largest outline drawn first)", fontsize=12)
        for k, (seed, ids) in enumerate(groups):
            panel(fig.add_subplot(n_rows, n_cols, k + 1), ids, masks, colors,
                  f"group {k + 1}: zones {', '.join(map(str, ids))}", shape, outline, circle=info[seed])
        largest = [ids[0] for _, ids in groups]
        panel(fig.add_subplot(n_rows, n_cols, len(groups) + 1), largest, masks, colors,
              f"largest of each group: {', '.join(map(str, largest))}", shape, outline)
        panel(fig.add_subplot(n_rows, n_cols, len(groups) + 2), largest, masks, colors,
              "largest of each group + NR zones", shape, outline, nr_masks=analyzer.non_recur_masks)
        out_path = OUT_DIR / f"{stem}_zone_groups{'_far' if MODE == 'far' else ''}.png"
        fig.savefig(out_path, dpi=MAP_DPI)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
