# ruff: noqa: INP001
"""
flash_area_obj.py  --  plain_02 Figs. 1-2: area of every flash at 60X and 40X (GACh3.0), split by whether it
touches the frame boundary, + table of the recordings used.

Steps
-----
1. Read the per-recording flash CSVs: obj from obj_coverage.py (output/plain_02/per_recording/), area + bounding
   box from edge_touch.py (output/plain_02/edge/). Flash = one connected blob of one frame (no merging, no size cap).
2. Area in um^2 = px / (px per um)^2 (PIXEL_SCALE of the objective); touches the boundary = bounding box within
   EDGE_PX of the frame edge (see edge_touch.py).
3. One histogram per objective (Fig. 1 = 60X, Fig. 2 = 40X), stacked: not touching / touching the boundary;
   dashed line = whole field of view.
4. Table of recordings (date, animal, slice) + summary -> docs/plain_02/tables/flash_area_recordings.xlsx.
   Independent slice = (recording date, SLICE in rec_data.db).

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/flash_area_obj.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_touch import EDGE_PX, FRAME_PX  # noqa: E402

from classes.region_analyzer import PIXEL_SCALE  # noqa: E402
from functions.database_ops import lookup_rec_from_db  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / "docs" / "plain_02" / "figures"
DATA_DIR = ROOT / "output" / "plain_02"  # regenerable masks / caches
XLSX_PATH = ROOT / "docs" / "plain_02" / "tables" / "flash_area_recordings.xlsx"
OBJS = {"60X": 1, "40X": 2}  # objective -> figure number
N_BINS = 40  # log-spaced histogram bins between the smallest flash and the field-of-view area
COLOR_IN = "#0072B2"  # not touching the boundary (Okabe-Ito blue)
COLOR_EDGE = "#E69F00"  # touching the boundary (Okabe-Ito orange)


def load_flashes() -> pd.DataFrame:
    """Step 1-2: one row per flash with obj, area_um2, touches_edge."""
    parts = []
    for edge_csv in sorted((DATA_DIR / "edge").glob("*.csv")):
        obj = pd.read_csv(DATA_DIR / "per_recording" / edge_csv.name, nrows=1)["obj"]
        if obj.empty:  # recording without flashes
            continue
        obj = obj.item()
        parts.append(pd.read_csv(edge_csv).assign(recording=edge_csv.stem, obj=obj))
    flashes = pd.concat(parts, ignore_index=True)
    far = FRAME_PX - EDGE_PX
    flashes["touches_edge"] = ((flashes["r0"] <= EDGE_PX) | (flashes["c0"] <= EDGE_PX)
                               | (flashes["r1"] >= far) | (flashes["c1"] >= far))
    flashes["area_um2"] = flashes["area_px"] / flashes["obj"].map(PIXEL_SCALE) ** 2
    return flashes


def figure(flashes: pd.DataFrame, obj: str, fig_no: int, fov_k: float) -> Path:
    """Area histogram, stacked: not touching / touching the boundary; dashed line = whole field of view."""
    area = flashes["area_um2"].to_numpy()
    edge = flashes["touches_edge"].to_numpy()
    fov = fov_k * 1e3
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot()
    ax.hist([area[~edge], area[edge]], bins=np.logspace(np.log10(area.min()), np.log10(fov), N_BINS + 1), stacked=True,
            color=[COLOR_IN, COLOR_EDGE], edgecolor="white", linewidth=0.6,
            label=[f"inside the frame ({100 * (~edge).mean():.0f} %)",
                   f"touching the frame edge ({100 * edge.mean():.0f} %)"])
    ax.axvline(fov, color="#555555", ls="--", lw=1.5)
    ax.text(fov, ax.get_ylim()[1] * 0.5, "frame size ", ha="right", va="center", fontsize=10)
    ax.legend(frameon=False, loc="upper center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(area.min() * 0.9, fov * 1.1)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("flash area (µm², log scale)")
    ax.set_ylabel("flashes")
    ax.set_title(f"Fig. {fig_no}: {obj}", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out_path = OUT_DIR / f"fig{fig_no}_flash_area_{obj}.png"
    fig.savefig(out_path, dpi=120)
    return out_path


def main() -> None:
    """Steps 1-4."""
    flashes = load_flashes()

    raw_names = [f"{stem.split('_BIEXP')[0]}.tif" for stem in flashes["recording"].unique()]
    info = pd.DataFrame(lookup_rec_from_db(pl.DataFrame({"raw_tiff_name": raw_names}), ROOT / "data" / "rec_data.db",
                                           ROOT / "data" / "exp_info.db").to_dicts())  # no pyarrow in the venv
    info["recording"] = info["Filename"].str.removesuffix(".tif") + "_BIEXP_ALS"
    info["date"] = info["Filename"].str.split("-").str[0]
    info = info[["recording", "date", "ANIMAL_ID", "SLICE"]].rename(columns={"ANIMAL_ID": "animal",
                                                                          "SLICE": "slice"})

    per_rec = (flashes.groupby(["recording", "obj"], as_index=False)
               .agg(flashes=("area_um2", "size"), touching_edge=("touches_edge", "sum"),
                    median_area_um2=("area_um2", "median"), max_area_um2=("area_um2", "max"))
               .merge(info, on="recording"))
    per_rec["recording"] = per_rec["recording"].str.removesuffix("_BIEXP_ALS")
    per_rec = per_rec[["obj", "recording", "date", "animal", "slice", "flashes", "touching_edge",
                       "median_area_um2", "max_area_um2"]].round(0)

    summary = []
    for obj, fig_no in OBJS.items():
        rec = per_rec[per_rec["obj"] == obj]
        sub = flashes[flashes["obj"] == obj]
        inside = sub.loc[~sub["touches_edge"], "area_um2"]
        n_slices = rec[["date", "slice"]].drop_duplicates().shape[0]
        fov_um2 = (FRAME_PX / PIXEL_SCALE[obj]) ** 2
        summary.append({"obj": obj, "recordings": len(rec), "independent_slices": n_slices,
                        "animals": rec["animal"].nunique(), "flashes": len(sub),
                        "not_touching": len(inside), "touching": int(sub["touches_edge"].sum()),
                        "fov_um2": round(fov_um2), "median_area_um2": round(sub["area_um2"].median()),
                        "median_area_not_touching_um2": round(inside.median()),
                        "max_area_not_touching_um2": round(inside.max())})
        print(f"saved {figure(sub, obj, fig_no, fov_um2 / 1e3)}")

    with pd.ExcelWriter(XLSX_PATH) as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        for obj in OBJS:
            per_rec[per_rec["obj"] == obj].to_excel(writer, sheet_name=obj, index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"saved {XLSX_PATH}")


if __name__ == "__main__":
    main()
