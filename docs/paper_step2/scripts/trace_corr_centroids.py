# ruff: noqa: INP001
"""
Scratch: how far apart are the tracks inside each trace-corr group? (hint for MAX_CENTROID_DEVIATION)

  Step 1. Detect + group each recording with current settings (2 sigma, 115 px chaining)
  Step 2. Per trace-corr group: hotspot mean centroids -> distance between every two of them
  Step 3. Per recording: stats (xlsx per_recording sheet), distance histogram, centroid map -> output/paper_step2/

Run from the repo root:
    .venv/Scripts/python.exe docs/paper_step2/scripts/trace_corr_centroids.py
"""

## Modules
# Standard library imports
import sys
from itertools import combinations
from pathlib import Path

# Third-party imports
import matplotlib as mpl
import numpy as np
import pandas as pd
import tifffile

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Local imports
from classes import SpontaneousZoneAnalyzer  # noqa: E402
from classes.sp_zone_analyzer import MAX_CENTROID_DEVIATION, track_centroids  # noqa: E402
from functions import check_cuda  # noqa: E402

# ===========================================================================
#   CONFIG
# ===========================================================================

PROC_DIR = Path("proc_tiffs")
OUT_DIR = Path("output/paper_step2")
RECORDINGS = ["2025_06_11-0003", "2025_12_15-0012",  # all 10X; + top 5 by trace-corr group count
              "2025_11_13-0042", "2025_11_13-0043", "2025_06_11-0013", "2026_01_08-0021", "2026_01_08-0036"]


def group_distances(analyzer: SpontaneousZoneAnalyzer) ->tuple[pd.Series, pd.DataFrame]:
    """Distances between hotspot centroids in the same trace-corr group, and the centroids (for the map)."""
    centroids = track_centroids(analyzer.detections).set_index("joint_label")["mean_centroid"]
    dists, track_rows = [], []
    for _, grp in analyzer.trace_corr_groups.iterrows():
        pts = np.array([centroids[label] for label in grp["labels"]])
        dists += [float(np.hypot(*(pts[i] - pts[j]))) for i, j in combinations(range(len(pts)), 2)]
        track_rows += [{"group": grp["group"], "y": p[0], "x": p[1]} for p in pts]
    return pd.Series(dists, dtype=float), pd.DataFrame(track_rows)


def plot_map(tracks: pd.DataFrame, rec: str, shape: tuple[int, int], path: Path) -> None:
    """Track centroids coloured by trace-corr group, group centre circled with r = 115 px."""
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab20")
    for k, (group, sub) in enumerate(tracks.groupby("group")):
        color = cmap(k % 20)
        ax.scatter(sub["x"], sub["y"], color=color, s=30, label=f"#{group} (n={len(sub)})")
        cy, cx = sub["y"].mean(), sub["x"].mean()
        ax.add_patch(plt.Circle((cx, cy), MAX_CENTROID_DEVIATION, fill=False, color=color, lw=0.8, ls="--"))
        ax.text(cx, cy, str(group), color=color, fontsize=9, ha="center", va="center", weight="bold")
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title(f"{rec}: trace-corr track centroids (dashed circle r = {MAX_CENTROID_DEVIATION} px)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Run all recordings, save table, histogram and maps."""
    cuda_available, _ = check_cuda()
    summary_rows = []
    for rec in RECORDINGS:
        print(f"=== {rec} ===")
        stack = tifffile.imread(PROC_DIR / f"{rec}_BIEXP_ALS.tif")
        analyzer = SpontaneousZoneAnalyzer(stack, obj="10X", cuda_available=cuda_available)
        del stack
        analyzer.detect()
        analyzer.group()
        um = analyzer.um_per_px
        dists, tracks = group_distances(analyzer)
        if dists.empty:
            print(f"{rec}: no trace-corr groups -- skipped")
            continue
        summary_rows.append(summary_row(rec, len(analyzer.trace_corr_groups), dists, um))
        map_path = OUT_DIR / f"{rec}_trace_corr_centroids.png"
        plot_map(tracks, rec, (analyzer.height, analyzer.width), map_path)
        print(f"saved {map_path.resolve()}")
        hist_path = OUT_DIR / f"{rec}_trace_corr_dist_hist.png"
        plot_hist(dists, rec, um, hist_path)
        print(f"saved {hist_path.resolve()}")

    summary = pd.DataFrame(summary_rows)
    xlsx_path = OUT_DIR / "trace_corr_centroids.xlsx"
    summary.to_excel(xlsx_path, sheet_name="per_recording", index=False)
    print(f"saved {xlsx_path.resolve()}")

    pd.set_option("display.width", 250)
    print(summary.round(1).to_string(index=False))


def summary_row(rec: str, n_groups: int, dists: pd.Series, um: float) -> dict:
    """Mean, SD, median, Q1, Q3 of one recording's distances, in px and um."""
    stats = {"mean": dists.mean(), "sd": dists.std(),
             "median": dists.median(), "q1": dists.quantile(0.25), "q3": dists.quantile(0.75)}
    row = {"recording": rec, "n_trace_corr_groups": n_groups, "n_distances": len(dists)}
    row |= {f"{k}_px": v for k, v in stats.items()}
    row |= {f"{k}_um": v * um for k, v in stats.items()}
    return row


def plot_hist(dist_px: pd.Series, rec: str, um: float, path: Path) -> None:
    """Histogram of one recording's within-group distances with the 115 px line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dist_px, bins=30, color="0.5")
    ax.axvline(MAX_CENTROID_DEVIATION, color="r", ls="--", label=f"{MAX_CENTROID_DEVIATION} px")
    ax.set_xlabel(f"distance between two hotspots in one trace-corr group (px; 1 px = {um:.2f} um)")
    ax.set_ylabel("count")
    ax.set_title(rec)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
