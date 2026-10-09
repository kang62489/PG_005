# ruff: noqa: INP001
"""
edge_touch_10x.py  --  plain_02: how many 10X flashes of Fig. 3 touch the frame boundary?

Steps
-----
1. Recordings: spontaneous_summary.xlsx, sensor GACh3.0 (all 10X), as fig3_flash_area.py.
2. Per recording: saved mask (results/spontaneous/mask) -> per-frame flashes with the pipeline's step 2a
   (spatiotemporally_connect_flashes, the same flashes as Fig. 3).
3. A flash touches the boundary if any footprint pixel lies within EDGE_PX of the frame edge (edge_touch.py);
   cache one CSV per recording (output/plain_02/edge_10x/, resumable).
4. Counts: all flashes, by size, and vs the arbor yardsticks of Fig. 3.

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/edge_touch_10x.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from edge_touch import EDGE_PX  # noqa: E402

from classes.region_analyzer import PIXEL_SCALE  # noqa: E402
from classes.sp_zone_analyzer import CONNECT_RADIUS, MAX_FLASH_FRAC, spatiotemporally_connect_flashes  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
SUMMARY_PATH = ROOT / "results" / "spontaneous" / "spontaneous_summary.xlsx"
MASK_DIR = ROOT / "results" / "spontaneous" / "mask"
CACHE_DIR = ROOT / "output" / "plain_02" / "edge_10x"
UM2_PER_PX = (1.0 / PIXEL_SCALE["10X"]) ** 2
ARBOR_K_UM2 = {"convex hull": 77.6, "bounding box": 103.8}  # Fig. 3 yardsticks (x 10^3 um^2, arbor_hull.py)


def edge_table(stem: str) -> pd.DataFrame:
    """Step 2-3: per-frame flashes of one recording, area (px) + touches the boundary."""
    mask = tifffile.imread(MASK_DIR / f"{stem}_HOTSPOT_MASK.tif") > 0  # old file name until the rerun
    _, h, w = mask.shape
    det, footprints, _ = spatiotemporally_connect_flashes(mask, MAX_FLASH_FRAC * h * w, CONNECT_RADIUS)
    touches = [bool((fp[:, 0] <= EDGE_PX).any() or (fp[:, 1] <= EDGE_PX).any()
                    or (fp[:, 0] >= h - 1 - EDGE_PX).any() or (fp[:, 1] >= w - 1 - EDGE_PX).any())
               for fp in footprints]
    return pd.DataFrame({"frame": det["frame"].to_numpy(), "area_px": det["area"].to_numpy(),
                         "touches_edge": touches})


def main() -> None:
    """Steps 1-4."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_excel(SUMMARY_PATH, sheet_name="recordings")
    stems = summary.loc[summary["sensor"] == "GACh3.0", "recording"].tolist()
    for k, stem in enumerate(stems, 1):
        out = CACHE_DIR / f"{stem}.csv"
        if not out.exists():
            edge_table(stem).to_csv(out, index=False)
            print(f"[{k}/{len(stems)}] {stem} done")
    table = pd.concat([pd.read_csv(CACHE_DIR / f"{s}.csv").assign(recording=s) for s in stems], ignore_index=True)
    area_k = table["area_px"].to_numpy() * UM2_PER_PX / 1e3
    edge = table["touches_edge"].to_numpy(dtype=bool)

    print(f"\n10X: {len(table)} flashes in {table['recording'].nunique()} recordings; "
          f"not touching {(~edge).sum()} ({100 * (~edge).mean():.1f} %), touching {edge.sum()} ({100 * edge.mean():.1f} %)")
    bins = [0, 25, 50, 78, 104, 200, 1000]
    classes = pd.cut(area_k, bins)
    by_size = pd.DataFrame({"size": classes, "edge": edge}).groupby("size", observed=False)["edge"].agg(
        ["size", "sum", "mean"])
    by_size.columns = ["flashes", "touching", "touching_frac"]
    by_size["touching_pct"] = (100 * by_size.pop("touching_frac")).round(1)
    print("by area (x 10^3 um^2):")
    print(by_size.to_string())
    for name, value in ARBOR_K_UM2.items():
        reach = area_k >= value
        print(f"reaching the {name} ({value} x 10^3 um^2): {reach.sum()} flashes, of which touching "
              f"{(reach & edge).sum()} ({100 * (reach & edge).sum() / reach.sum():.1f} %), "
              f"not touching {(reach & ~edge).sum()}")
    print(f"not touching: median {np.median(area_k[~edge]):.1f}, max {area_k[~edge].max():.1f} x 10^3 um^2")


if __name__ == "__main__":
    main()
