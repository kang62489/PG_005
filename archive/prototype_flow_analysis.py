"""Scratch: spike-aligned pre-masked TV-L1 flow, drawn over the paired CAT masks.

  Step 1. Load : MED/CAT exported by the pipeline (Phase 3 run, current ALS + σ settings)
  Step 2. Flow : functions.hotspot_flow.compute_flow_pairs -- the exact pipeline code
  Step 3. Plot : background = paired CAT masks (from-only / to-only / both, 3 colors),
                 arrows auto-scaled per recording so sub-pixel flow stays visible

Output: output/test8/optical_flow/{tag}_FLOW_CAT.png
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.ndimage import maximum_filter

PROJECT_ROOT = Path("D:/Programs/PG_005")
sys.path.insert(0, str(PROJECT_ROOT))

from classes.spatial_categorization import CATEGORY_BRIGHT  # noqa: E402
from functions.hotspot_flow import compute_flow_pairs  # noqa: E402

# ===== CONFIG =====
SRC_DIR = PROJECT_ROOT / "output" / "test7" / "phase3" / "after"
FLOW_DIR = PROJECT_ROOT / "output" / "test8" / "optical_flow"

QUIVER_STEP = 24              # px between arrows
ARROW_REF_PCTL = 95           # this percentile of drawn |flow| (all panels of a recording) ...
ARROW_REF_LEN_PX = QUIVER_STEP  # ... is drawn this long

MASK_COLORS = {               # RGB on a black background
    "from": (0.20, 0.60, 0.86),   # blue: bright in the "from" frame only
    "to":   (0.95, 0.61, 0.07),   # orange: bright in the "to" frame only
    "both": (0.75, 0.75, 0.75),   # gray: bright in both
}
ARROW_COLOR = "lime"

RECORDINGS = [  # tag used in the *_MED.tif / *_CAT.tif filenames
    "2025_06_11-0002_A1S4RC1_BIEXP_ALS",
    "2025_06_11-0003_A1S4RC1_BIEXP_ALS",
    "2025_11_13-0017_A1S1RC2_BIEXP_ALS",
    "2025_11_13-0018_A1S1RC2_BIEXP_ALS",
    "2025_12_15-0012_A1S3RC1_BIEXP_ALS",
    "2025_12_15-0013_A1S3RC1_BIEXP_ALS",
]


def cat_pair_rgb(bright_from: np.ndarray, bright_to: np.ndarray) -> np.ndarray:
    """(H, W, 3) image: from-only / to-only / both in MASK_COLORS, black elsewhere."""
    rgb = np.zeros((*bright_from.shape, 3))
    rgb[bright_from & ~bright_to] = MASK_COLORS["from"]
    rgb[~bright_from & bright_to] = MASK_COLORS["to"]
    rgb[bright_from & bright_to] = MASK_COLORS["both"]
    return rgb


def plot_flow_panels(cat: np.ndarray, flow_pairs: list[dict], title: str) -> Figure:
    """One panel per pair: paired CAT masks + lime arrows near the "from" frame's bright pixels."""
    n_panels = len(flow_pairs)
    fig = Figure(figsize=(6 * n_panels, 6.6), layout="constrained")
    height, width = cat.shape[1], cat.shape[2]
    grid_y, grid_x = np.mgrid[0:height:QUIVER_STEP, 0:width:QUIVER_STEP]

    # --- arrows drawn per panel + one shared scale for the whole recording ---
    drawn = []
    for pair in flow_pairs:
        near = maximum_filter(cat[pair["idx_from"]] == CATEGORY_BRIGHT, size=QUIVER_STEP)[grid_y, grid_x]
        u, v = pair["u"][grid_y, grid_x][near], pair["v"][grid_y, grid_x][near]
        drawn.append((grid_x[near], grid_y[near], u, v))
    all_mag = np.concatenate([np.hypot(u, v) for _, _, u, v in drawn]) if drawn else np.array([])
    ref_mag = float(np.percentile(all_mag, ARROW_REF_PCTL)) if all_mag.size else 0.0
    scale = ref_mag / ARROW_REF_LEN_PX if ref_mag > 0 else 1.0  # |flow| px per drawn px

    for i, (pair, (px, py, u, v)) in enumerate(zip(flow_pairs, drawn, strict=True)):
        bright_from = cat[pair["idx_from"]] == CATEGORY_BRIGHT
        bright_to = cat[pair["idx_to"]] == CATEGORY_BRIGHT
        ax = fig.add_subplot(1, n_panels, i + 1)
        ax.imshow(cat_pair_rgb(bright_from, bright_to), origin="upper", interpolation="nearest")
        if px.size:
            ax.quiver(px, py, u, v, color=ARROW_COLOR, angles="xy", scale_units="xy", scale=scale,
                      width=0.003, headwidth=2.5, headlength=3, headaxislength=2.5)
        keep = pair["keep_mask"]
        mean_mag = float(np.hypot(pair["u"], pair["v"])[keep].mean()) if keep.any() else 0.0
        ax.set_title(f"{pair['label']}\nmean |flow| in mask = {mean_mag:.2f} px", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(
        handles=[Patch(color=MASK_COLORS["from"], label="bright in 'from' only"),
                 Patch(color=MASK_COLORS["to"], label="bright in 'to' only"),
                 Patch(color=MASK_COLORS["both"], label="bright in both")],
        loc="outside lower center", ncol=3, fontsize=11, frameon=False,
    )
    fig.suptitle(f"{title}  (pre-masked)  --  arrow scale: {ARROW_REF_LEN_PX} px drawn = {ref_mag:.2f} px flow "
                 f"({ARROW_REF_PCTL}th pct)", fontsize=13)
    return fig


def run() -> None:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    for tag in RECORDINGS:
        print(f"\n=== {tag} ===", flush=True)
        med = tifffile.imread(SRC_DIR / "median" / f"{tag}_MED.tif")
        cat = tifffile.imread(SRC_DIR / "categorized" / f"{tag}_CAT.tif")
        pairs = compute_flow_pairs(med, cat, med.shape[0] // 2)

        fig = plot_flow_panels(cat, pairs, title=tag)
        out_path = FLOW_DIR / f"{tag}_FLOW_CAT.png"
        fig.savefig(out_path, dpi=110)
        print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    run()
