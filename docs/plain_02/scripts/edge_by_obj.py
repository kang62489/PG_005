# ruff: noqa: INP001
"""
edge_by_obj.py  --  plain_02 Fig. 4: share of flashes touching the frame edge, per objective (GACh3.0).

Steps
-----
1. 40X / 60X: blob areas + bounding boxes from edge_touch.py (output/plain_02/edge/).
2. 10X: blob tables of the pipeline masks after the TH_SMALL_OBJ cleanup, written by fig3_flash_area.py
   (output/plain_02/edge_10x_blob/). Same flash definition for all objectives: one 4-connected blob of one frame,
   no merging, no size cap.
3. Touching = bounding box within EDGE_PX of the frame edge.
4. Figure: % of all flashes touching the frame edge, one bar per objective.

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/edge_by_obj.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from edge_touch import EDGE_PX, FRAME_PX  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "output" / "plain_02"
CACHE_10X = DATA_DIR / "edge_10x_blob"
OUT_PATH = ROOT / "docs" / "plain_02" / "figures" / "fig4_edge_by_obj.png"
COLORS = {"10X": "#0072B2", "40X": "#009E73", "60X": "#E69F00"}  # Okabe-Ito


def load() -> pd.DataFrame:
    """Steps 1-3: one row per flash with obj, recording, touches_edge."""
    parts = []
    for edge_csv in sorted((DATA_DIR / "edge").glob("*.csv")):
        obj = pd.read_csv(DATA_DIR / "per_recording" / edge_csv.name, nrows=1)["obj"]
        if obj.empty:  # recording without flashes
            continue
        obj = obj.item()
        parts.append(pd.read_csv(edge_csv).assign(recording=edge_csv.stem, obj=obj))

    for csv in sorted(CACHE_10X.glob("*.csv")):  # written by fig3_flash_area.py (run it first)
        parts.append(pd.read_csv(csv).assign(recording=csv.stem, obj="10X"))

    flashes = pd.concat(parts, ignore_index=True)
    far = FRAME_PX - EDGE_PX
    flashes["touches_edge"] = ((flashes["r0"] <= EDGE_PX) | (flashes["c0"] <= EDGE_PX)
                               | (flashes["r1"] >= far) | (flashes["c1"] >= far))
    return flashes


def main() -> None:
    """Steps 1-4."""
    flashes = load()
    objs = ["60X", "40X", "10X"]
    pct = [100 * flashes.loc[flashes["obj"] == obj, "touches_edge"].mean() for obj in objs]
    for obj, value in zip(objs, pct, strict=True):
        print(f"{obj}: {(flashes['obj'] == obj).sum()} flashes, touching the frame edge {value:.1f} %")

    fig = Figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.add_subplot()
    ax.bar(objs, pct, color=[COLORS[obj] for obj in objs], width=0.6)
    for k, value in enumerate(pct):
        ax.text(k, value + 1.5, f"{value:.0f} %", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("flashes touching the frame edge (%)")
    ax.set_title("Fig. 4: flashes touching the frame edge", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
