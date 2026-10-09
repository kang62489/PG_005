# ruff: noqa: INP001, E402
"""
fig3_flash_area.py  --  plain_02 Fig. 3: area of every 10X GACh3.0 flash vs one ChI's axon arbor.

Steps
-----
1. Recordings: spontaneous_summary.xlsx, sensor GACh3.0 (all rows there are 10X).
2. Per recording: saved mask (results/spontaneous/mask), blobs < TH_SMALL_OBJ px removed -> per-frame flashes with
   the pipeline's own step 2a (spatiotemporally_connect_flashes: adjacent blobs merged, > 80 % of the frame dropped).
3. Cache one CSV per recording (output/plain_02/flash_areas/, regenerable; re-runs skip finished ones).
4. Yardsticks from Aosaki & Kawaguchi 1996 Fig. b (arbor_hull.py): convex hull and bounding box.
5. Figure: A = the traced ChI with hull + box | B = histogram of flash area + both yardsticks, % below / reaching.

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/fig3_flash_area.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tifffile
from arbor_hull import CLEAR_BOXES, PANEL, measure
from matplotlib.figure import Figure
from obj_coverage import clean_and_table  # runs check_cuda() before anything imports numba

from classes.region_analyzer import PIXEL_SCALE
from classes.sp_zone_analyzer import CONNECT_RADIUS, MAX_FLASH_FRAC, spatiotemporally_connect_flashes

# ── CONFIG ────────────────────────────────────────────────────────────────────
SUMMARY_PATH = ROOT / "results" / "spontaneous" / "spontaneous_summary.xlsx"
MASK_DIR = ROOT / "results" / "spontaneous" / "mask"
CACHE_DIR = ROOT / "output" / "plain_02" / "flash_areas"  # regenerable cache, same step-2a output
EDGE_10X_DIR = ROOT / "output" / "plain_02" / "edge_10x_blob"  # blob tables for Fig. 4
OUT_PATH = ROOT / "docs" / "plain_02" / "figures" / "fig3_flash_area_10X.png"
SENSOR = "GACh3.0"
UM2_PER_PX = (1.0 / PIXEL_SCALE["10X"]) ** 2
BIN_K_UM2 = 20  # histogram bin width (x 10^3 um^2)
HULL_STYLE = {"color": "#E69F00", "ls": "-", "lw": 2}  # convex hull: orange, solid
BOX_STYLE = {"color": "#555555", "ls": "--", "lw": 1.5}  # bounding box: dark gray, dashed


def flash_table(stem: str) -> pd.DataFrame:
    """Step 2: per-frame flashes of one recording, area in px and um^2.

    The saved masks were made with 4,000 px; blobs < TH_SMALL_OBJ (now 5,000 px) are removed first. The blob table
    of the cleaned mask is saved for Fig. 4 (edge_by_obj.py), so each mask is read once.
    """
    mask = tifffile.imread(MASK_DIR / f"{stem}_HOTSPOT_MASK.tif") > 0  # old file name until the rerun
    mask, blobs = clean_and_table(mask)
    pd.DataFrame(blobs, columns=["frame", "area_px", "r0", "c0", "r1", "c1"]).to_csv(
        EDGE_10X_DIR / f"{stem}.csv", index=False)
    _, h, w = mask.shape
    det, _, _ = spatiotemporally_connect_flashes(mask, MAX_FLASH_FRAC * h * w, CONNECT_RADIUS)
    det = det.drop(columns="joint_label").rename(columns={"area": "area_px"})
    det["area_um2"] = det["area_px"] * UM2_PER_PX
    return det.assign(recording=stem)


def draw_arbor(ax, m: dict) -> None:
    """Panel A: traced ChI (panel b of the paper figure) + convex hull + bounding box."""
    rows, cols = PANEL
    img = m["img"].copy()
    for clear_rows, clear_cols in CLEAR_BOXES:
        img[clear_rows, clear_cols] = 1.0  # the paper's bar sits above the crop -> drop its text, draw our own bar
    ax.imshow(img[PANEL], cmap="gray", vmin=0, vmax=1)
    bar_px = 100 * m["px_per_um"]
    x_end, y_bar = img[PANEL].shape[1] - 10, 25
    ax.plot([x_end - bar_px, x_end], [y_bar, y_bar], color="black", lw=2.5)
    ax.text(x_end - bar_px / 2, y_bar + 8, "100 µm", ha="center", va="top", fontsize=10)
    xs, ys = m["xs"] - cols.start, m["ys"] - rows.start
    loop = np.r_[m["hull"].vertices, m["hull"].vertices[:1]]
    ax.plot(xs[loop], ys[loop], **HULL_STYLE)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], **BOX_STYLE)
    ax.set_title(f"A  One rat ChI, slice (Aosaki & Kawaguchi 1996)\n"
                 f"extent {m['width_um']:.0f} × {m['height_um']:.0f} µm", fontsize=11, loc="left")
    ax.axis("off")


def main() -> None:
    """Steps 1-5."""
    summary = pd.read_excel(SUMMARY_PATH, sheet_name="recordings")
    stems = summary.loc[summary["sensor"] == SENSOR, "recording"].tolist()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EDGE_10X_DIR.mkdir(parents=True, exist_ok=True)
    for k, stem in enumerate(stems, 1):
        out = CACHE_DIR / f"{stem}_hotspots.csv"
        if not out.exists():
            flash_table(stem).to_csv(out, index=False)
            print(f"[{k}/{len(stems)}] {stem} cached")
    flashes = pd.concat([pd.read_csv(CACHE_DIR / f"{s}_hotspots.csv") for s in stems], ignore_index=True)
    area_k = flashes["area_um2"].to_numpy() / 1e3
    n_rec = flashes["recording"].nunique()

    m = measure()
    sticks = [("convex hull", m["hull_um2"] / 1e3, HULL_STYLE), ("bounding box", m["box_um2"] / 1e3, BOX_STYLE)]

    fig = Figure(figsize=(16, 6), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.7])
    draw_arbor(fig.add_subplot(grid[0]), m)

    ax = fig.add_subplot(grid[1])
    ax.hist(area_k, bins=np.arange(0, area_k.max() + BIN_K_UM2, BIN_K_UM2), color="#0072B2", edgecolor="white",
            linewidth=0.6)
    lines = []
    for name, value, style in sticks:
        ax.axvline(value, **style)
        n_reach = int((area_k >= value).sum())
        lines.append(f"{name} {value:.0f} × 10³ µm²: below {100 * (area_k < value).mean():.1f} %, "
                     f"reaching {100 * n_reach / len(area_k):.1f} % ({n_reach})")
        print(lines[-1])
    handles = [ax.lines[i] for i in range(len(sticks))]
    ax.legend(handles, lines, title="one ChI axon arbor (panel A)", loc="upper right", frameon=False, fontsize=10)
    ax.set_xlabel("area of one flash (× 10³ µm²)")
    ax.set_ylabel("flashes")
    ax.set_title(f"B  10X {SENSOR}: {len(flashes)} flashes in {n_rec} recordings "
                 f"(median {np.median(area_k):.0f} × 10³ µm²)", fontsize=11, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
