# ruff: noqa: INP001
"""
edge_touch.py  --  plain_02: how many 40X / 60X flashes touch the frame boundary (= may be cut by the field edge)?

Steps
-----
1. Per recording: flash mask (output/plain_02/mask/, from obj_coverage.py) -> blobs per frame (same labelling).
2. A flash touches the boundary if any of its pixels lies within EDGE_PX of the first / last row or column
   (the mask is never on in the outermost 1-px ring -- edge effect of the mask cleanup -- so a cut flash ends at 1).
3. Cache one CSV per recording (output/plain_02/edge/, resumable) with area + bounding box.
4. Counts per objective: all flashes, and by size (share of the field of view).

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/edge_touch.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from classes.region_analyzer import PIXEL_SCALE  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "output" / "plain_02"
EDGE_DIR = DATA_DIR / "edge"
SIZE_BINS_PCT = [0, 1, 10, 25, 50, 100]  # flash area as % of the field of view
FRAME_PX = 1024  # frame side (px)
EDGE_PX = 1  # px: a flash whose bounding box reaches this close to the frame edge touches it


def edge_rows(mask: np.ndarray) -> list[dict]:
    """One row per blob per frame: frame (1-based), area (px), bounding box (r1 / c1 exclusive)."""
    rows = []
    for f, frame in enumerate(mask, 1):
        for blob in regionprops(label(frame)):
            r0, c0, r1, c1 = blob.bbox
            rows.append({"frame": f, "area_px": int(blob.area), "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    return rows


def main() -> None:
    """Steps 1-4."""
    EDGE_DIR.mkdir(exist_ok=True)
    flashes = []
    for rec_csv in sorted((DATA_DIR / "per_recording").glob("*.csv")):
        stem = rec_csv.stem
        obj = pd.read_csv(rec_csv, nrows=1)["obj"]
        if obj.empty:  # recording without flashes
            continue
        out = EDGE_DIR / f"{stem}.csv"
        if not out.exists():
            mask = tifffile.imread(DATA_DIR / "mask" / f"{stem}_FLASH_MASK.tif") > 0
            pd.DataFrame(edge_rows(mask), columns=["frame", "area_px", "r0", "c0", "r1", "c1"]).to_csv(out, index=False)
            print(f"{stem} done")
        flashes.append(pd.read_csv(out).assign(recording=stem, obj=obj.item()))
    table = pd.concat(flashes, ignore_index=True)
    far = FRAME_PX - EDGE_PX  # exclusive bbox end reaching row / col FRAME_PX - 1 - EDGE_PX
    table["touches_edge"] = ((table["r0"] <= EDGE_PX) | (table["c0"] <= EDGE_PX)
                             | (table["r1"] >= far) | (table["c1"] >= far))
    table["area_um2"] = table["area_px"] / table["obj"].map(PIXEL_SCALE) ** 2
    table["fov_pct"] = 100 * table["area_px"] / 1024**2
    table["size"] = pd.cut(table["fov_pct"], SIZE_BINS_PCT, labels=[
        f"{a}-{b} %" for a, b in zip(SIZE_BINS_PCT[:-1], SIZE_BINS_PCT[1:], strict=True)])

    for obj in ["60X", "40X"]:
        sub = table[table["obj"] == obj]
        n_in = int((~sub["touches_edge"]).sum())
        print(f"\n{obj}: {len(sub)} flashes; not touching the boundary {n_in} ({100 * n_in / len(sub):.1f} %), "
              f"touching {len(sub) - n_in} ({100 * (len(sub) - n_in) / len(sub):.1f} %)")
        by_size = sub.groupby("size", observed=False).agg(
            flashes=("touches_edge", "size"), touching=("touches_edge", "sum"))
        by_size["not_touching"] = by_size["flashes"] - by_size["touching"]
        by_size["touching_pct"] = (100 * by_size["touching"] / by_size["flashes"]).round(1)
        print(by_size.to_string())
        inside = sub.loc[~sub["touches_edge"], "area_um2"]
        print(f"not touching: median {inside.median():.0f} um^2, max {inside.max():.0f} um^2 "
              f"({100 * inside.max() / (1024 / PIXEL_SCALE[obj]) ** 2:.1f} % of the field)")


if __name__ == "__main__":
    main()
