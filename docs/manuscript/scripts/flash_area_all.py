# ruff: noqa: INP001, E402
"""
flash_area_all.py  --  area of every flash at 60X, 40X and 10X (GACh3.0), NO lower size limit.

Same as plain_02 Figs. 1-3, but blobs < 5,000 px are kept and the 10X figure has no ChI-arbor comparison.
Independent output: everything goes under --out_root (default output/flash_area/), nothing into results/.

Steps
-----
1. Select: proc_20260922_000.txt -> recordings with an ALS tiff, OBJ 60X / 40X / 10X, SENSOR GACh3.0.
2. Global histogram: exact count of every float16 value of the stack (lossless, any bin count can be rebuilt with
   fit_hist.rebin_code_counts) + the pipeline's background fit (ZONE_HIST_BINS bins)
   -> value_counts/{stem}.npz (values, counts, lo, hi, bg_center, bg_sigma, threshold, n_bins).
3. Detect: threshold = background peak + CROSSOVER_RATIO sigma, mask cleanup (open / close / fill holes),
   blob size filter off (th_small_obj = 0).
4. Flash = one 4-connected blob of one frame, all objectives (as plain_02 Figs. 1-2 and 4; no step-2a merge at
   10X: ~450 noise specks per frame would chain into frame-wide clusters).
   Area + bounding box + touches the frame edge -> flash_tables/{stem}.csv
   (resumable: recordings with both files are skipped; --max_new N detects at most N new recordings per call).
5. Per objective: histogram stacked inside the frame / touching the frame edge, dashed line = frame size,
   log x and log y (noise specks and large flashes on one plot) -> figures/flash_area_{obj}.png
   + the plotted bins -> figures/flash_area_{obj}_bins.csv; per-recording fit values -> recordings.csv.

Usage:
    .venv/Scripts/python.exe docs/manuscript/scripts/flash_area_all.py --max_new 5
    Saion: sbatch docs/manuscript/scripts/flash_area_all.slm
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from functions import check_cuda

_CUDA, _MSG = check_cuda()  # must run before anything imports numba

import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter
from scipy import ndimage

from classes.region_analyzer import PIXEL_SCALE
from classes.sp_zone_analyzer import CROSSOVER_RATIO
from functions.fit_hist import (
    ZONE_HIST_BINS,
    ZONE_HIST_RANGE_PCT,
    fit_background,
    float16_code_counts,
    percentiles_from_counts,
)
from functions.zone_kernels import zone_mask
from spontaneous_analysis import select_recordings

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROC_LIST = ROOT / "data" / "proc_20260922_000.txt"
OUT_ROOT = ROOT / "output" / "flash_area"  # -> flash_tables/, value_counts/, figures/, recordings.csv
OBJS = ["60X", "40X", "10X"]
SENSOR = "GACh3.0"
EDGE_PX = 1  # px: a flash whose bounding box reaches this close to the frame edge touches it (as plain_02)
N_BINS = 40  # log-spaced histogram bins between the smallest flash and the frame area
COLOR_IN = "#0072B2"  # inside the frame (Okabe-Ito blue)
COLOR_EDGE = "#E69F00"  # touching the frame edge (Okabe-Ito orange)
COLUMNS = ["frame", "area_px", "r0", "c0", "r1", "c1", "touches_edge", "frame_px"]
CHUNK = 100  # frames labelled per call (int32 labels of 100 frames ~ 0.4 GB)
IN_PLANE_4 = np.zeros((3, 3, 3), dtype=bool)
IN_PLANE_4[1] = ndimage.generate_binary_structure(2, 1)  # 4-connected within a frame, never across frames


def global_histogram(stack: np.ndarray) -> dict:
    """Step 2: exact float16 value counts + the pipeline's background fit (same numbers fit_background uses)."""
    values, counts = float16_code_counts(stack)
    lo, hi = percentiles_from_counts(values, counts, ZONE_HIST_RANGE_PCT)
    bg_center, bg_sigma = fit_background(stack, cuda_available=_CUDA)
    return {"values": values, "counts": counts, "lo": lo, "hi": hi, "bg_center": bg_center, "bg_sigma": bg_sigma,
            "threshold": float(bg_center + CROSSOVER_RATIO * bg_sigma), "n_bins": ZONE_HIST_BINS}


def flash_table(stack: np.ndarray, threshold: float) -> pd.DataFrame:
    """Steps 3-4: per-frame flashes of one recording, no lower size limit."""
    mask = zone_mask(stack, threshold, 0, _CUDA)
    _, h, w = mask.shape
    parts = []
    for start in range(0, len(mask), CHUNK):  # labels in raster order = frame by frame, as per-frame labelling
        chunk = mask[start:start + CHUNK]
        labels, n = ndimage.label(chunk, structure=IN_PLANE_4)
        if n == 0:
            continue
        sizes = np.bincount(labels[chunk], minlength=n + 1)[1:]  # foreground px only (~5 %): ~6x faster
        boxes = np.array([(b[0].start, b[1].start, b[2].start, b[1].stop, b[2].stop)
                          for b in ndimage.find_objects(labels)])  # (frame, r0, c0, r1, c1), r1 / c1 exclusive
        parts.append(np.column_stack([boxes[:, 0] + start + 1, sizes, boxes[:, 1:]]))
    table = pd.DataFrame(np.concatenate(parts) if parts else np.empty((0, 6), dtype=np.int64), columns=COLUMNS[:6])
    table["touches_edge"] = ((table["r0"] <= EDGE_PX) | (table["c0"] <= EDGE_PX)
                             | (table["r1"] >= h - EDGE_PX) | (table["c1"] >= w - EDGE_PX))
    table["frame_px"] = h * w
    return table


def figure(flashes: pd.DataFrame, obj: str, n_rec: int, out_dir: Path) -> Path:
    """Step 5: area histogram, stacked inside / touching the frame edge, + its bin table."""
    area = flashes["area_um2"].to_numpy()
    edge = flashes["touches_edge"].to_numpy()
    fov = flashes["fov_um2"].max()
    bins = np.logspace(np.log10(area.min()), np.log10(fov), N_BINS + 1)
    pd.DataFrame({"bin_lo_um2": bins[:-1], "bin_hi_um2": bins[1:], "n_inside": np.histogram(area[~edge], bins)[0],
                  "n_touching": np.histogram(area[edge], bins)[0]}).to_csv(
        out_dir / f"flash_area_{obj}_bins.csv", index=False)

    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot()
    ax.hist([area[~edge], area[edge]], bins=bins, stacked=True, color=[COLOR_IN, COLOR_EDGE], edgecolor="white",
            linewidth=0.6, label=[f"inside the frame ({(~edge).sum():,}, {100 * (~edge).mean():.0f} %)",
                                  f"touching the frame edge ({edge.sum():,}, {100 * edge.mean():.0f} %)"])
    ax.axvline(fov, color="#555555", ls="--", lw=1.5)
    ax.text(fov, 10, "frame size ", ha="right", va="center", fontsize=10)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.95, 1), fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")  # noise specks (~10^5 per bin) and large flashes (a few per bin) on one plot
    ax.set_xlim(area.min() * 0.9, fov * 1.1)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("flash area (µm², log scale)")
    ax.set_ylabel("flashes (log scale)")
    ax.set_title(f"{obj} {SENSOR}: {len(flashes):,} flashes in {n_rec} recordings", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out_path = out_dir / f"flash_area_{obj}.png"
    fig.savefig(out_path, dpi=120)
    return out_path


def main() -> None:
    """Steps 1-5."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new", type=int, default=0, help="detect at most N new recordings (0 = all)")
    parser.add_argument("--proc_list", type=Path, default=PROC_LIST)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "rec_data.db")
    parser.add_argument("--exp_db", type=Path, default=ROOT / "data" / "exp_info.db")
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    print(_MSG)
    table_dir, counts_dir, fig_dir = (args.out_root / name for name in ("flash_tables", "value_counts", "figures"))
    for folder in (table_dir, counts_dir, fig_dir):
        folder.mkdir(parents=True, exist_ok=True)

    recordings = select_recordings(args.proc_list, args.db, args.exp_db, True)
    recordings = recordings.filter(recordings["OBJ"].is_in(OBJS) & (recordings["SENSOR"] == SENSOR))
    print(f"{len(recordings)} recordings: {recordings['OBJ'].value_counts().sort('OBJ').rows()}")

    n_new = 0
    for i, row in enumerate(recordings.iter_rows(named=True), 1):
        stem = Path(row["proc_tiff_path"]).stem
        table_path, counts_path = table_dir / f"{stem}.csv", counts_dir / f"{stem}.npz"
        if table_path.exists() and counts_path.exists():
            continue
        if args.max_new and n_new >= args.max_new:
            print(f"stopped after {n_new} new recordings -- run again to continue")
            return
        stack = np.asarray(tifffile.imread(row["proc_tiff_path"]), dtype=np.float16)  # ALS is float16: no copy
        hist = global_histogram(stack)
        tmp_path = counts_path.with_suffix(".tmp.npz")  # write + rename: a killed job leaves no partial file
        np.savez(tmp_path, **hist)
        tmp_path.replace(counts_path)
        table = flash_table(stack, hist["threshold"])
        tmp_path = table_path.with_suffix(".csv.tmp")
        table.to_csv(tmp_path, index=False)
        tmp_path.replace(table_path)
        n_new += 1
        print(f"[{i}/{len(recordings)}] {stem} ({row['OBJ']}): threshold {hist['threshold']:.5f}, "
              f"{len(table)} flashes")
        del stack, table  # free before the next read (stack ~2.5 GB, limit 16G)

    fits = []
    for row in recordings.iter_rows(named=True):
        stem = Path(row["proc_tiff_path"]).stem
        with np.load(counts_dir / f"{stem}.npz") as hist:
            fits.append({"recording": stem, "obj": row["OBJ"],
                         **{key: hist[key].item() for key in ("lo", "hi", "bg_center", "bg_sigma", "threshold",
                                                              "n_bins")}})
    pd.DataFrame(fits).to_csv(args.out_root / "recordings.csv", index=False)
    print(f"saved {args.out_root / 'recordings.csv'}")

    for obj in OBJS:
        stems = [Path(p).stem for p in recordings.filter(recordings["OBJ"] == obj)["proc_tiff_path"]]
        flashes = pd.concat([pd.read_csv(table_dir / f"{s}.csv", usecols=["area_px", "touches_edge", "frame_px"],
                                         dtype={"touches_edge": bool})  # header-only CSV would make it object
                             .assign(recording=s) for s in stems], ignore_index=True)
        flashes["area_um2"] = flashes["area_px"] / PIXEL_SCALE[obj] ** 2
        flashes["fov_um2"] = flashes["frame_px"] / PIXEL_SCALE[obj] ** 2
        n_rec = flashes["recording"].nunique()
        print(f"{obj}: {len(flashes)} flashes in {n_rec} of {len(stems)} recordings, "
              f"touching {100 * flashes['touches_edge'].mean():.1f} %, "
              f"< 5,000 px {100 * (flashes['area_px'] < 5000).mean():.1f} %, "
              f"median {flashes['area_um2'].median():.0f} µm²")
        print(f"saved {figure(flashes, obj, n_rec, fig_dir)} (+ _bins.csv)")


if __name__ == "__main__":
    main()
