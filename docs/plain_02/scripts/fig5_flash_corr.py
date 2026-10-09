# ruff: noqa: INP001, E402
"""
fig5_flash_corr.py  --  plain_02 Fig. 5: traces of three flashes and their r (0 lag).

Steps
-----
1. Run SpontaneousZoneAnalyzer on one recording (same settings as spontaneous_analysis.py).
2. Zone Z = the trace-corr recur_zone with the most units.
   A, B = the best-r flash pair between two units of Z in different events (= the r the pipeline uses).
   C = the largest flash of the largest recur_zone sharing 0 px with Z.
3. Trace = mean ALS value inside each flash's own footprint, every frame (footprint_traces, as in the pipeline).
4. Figure: footprints on the max projection | the three traces + r of A with A, B, C.

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/fig5_flash_corr.py
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
from functions import img_zscore_convert
from functions.zone_kernels import footprint_traces
from spontaneous_analysis import MAP_Z_MIN, _gray_range

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEM = "2026_01_08-0012"
OUT_PATH = ROOT / "docs" / "plain_02" / "figures" / f"fig5_flash_corr_{STEM}.png"
COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}  # Okabe-Ito (colorblind-safe)


def event_starts(frames: np.ndarray) -> np.ndarray:
    """Event id per active frame (event = run of consecutive frames)."""
    frames = np.unique(frames)
    return frames, np.cumsum(np.r_[True, np.diff(frames) > 1])


def main() -> None:
    """Steps 1-4."""
    print(_MSG)
    analyzer = SpontaneousZoneAnalyzer(tifffile.imread(ROOT / "proc_tiffs" / f"{STEM}_BIEXP_ALS.tif"), obj="10X",
                                       sigma_ratio=CROSSOVER_RATIO, cuda_available=_CUDA)
    analyzer.run()
    det, fps = analyzer.detections.reset_index(drop=True), analyzer.fps
    traces = footprint_traces(analyzer.footprints, analyzer.stack_f16, _CUDA).astype(np.float32)
    r_all = np.nan_to_num(np.corrcoef(traces))

    # Step 2a. zone Z: trace-corr recur_zone with the most units
    zones = analyzer.zones
    trace_corr = zones[zones["source"].str.startswith("trace_corr")]
    z_row = trace_corr.loc[trace_corr["joint_labels"].map(len).idxmax()]
    z_id, z_units = z_row.zone_id, list(z_row.joint_labels)

    # Step 2b. A, B: best-r flash pair between two units of Z that lie in different events
    z_frames, z_event = event_starts(det.loc[det["joint_label"].isin(z_units), "frame"].to_numpy())
    event_of = dict(zip(z_frames, z_event, strict=True))
    best = (-2.0, None, None)
    for i in np.flatnonzero(det["joint_label"].isin(z_units)):
        for j in np.flatnonzero(det["joint_label"].isin(z_units)):
            if i < j and det.at[i, "joint_label"] != det.at[j, "joint_label"] \
                    and event_of[det.at[i, "frame"]] != event_of[det.at[j, "frame"]] and r_all[i, j] > best[0]:
                best = (r_all[i, j], i, j)
    _, a, b = best

    # Step 2c. C: largest flash of the largest recur_zone with 0 px overlap with Z
    z_mask = analyzer.zone_masks[z_id]
    apart = [k for k, m in analyzer.zone_masks.items() if k != z_id and not (m & z_mask).any()]
    c_id = max(apart, key=lambda k: analyzer.zone_masks[k].sum())
    c_units = zones.loc[zones["zone_id"] == c_id, "joint_labels"].item()
    c_rows = np.flatnonzero(det["joint_label"].isin(c_units))
    c = c_rows[np.argmax([len(analyzer.footprints[k]) for k in c_rows])]

    picks = {"A": a, "B": b, "C": c}
    zone_of = {"A": z_id, "B": z_id, "C": c_id}
    for name, k in picks.items():
        print(f"{name}: recur_zone {zone_of[name]}, frame {det.at[k, 'frame']}, "
              f"{len(analyzer.footprints[k])} px, r with A = {r_all[a, k]:.3f}")

    # Step 4. figure
    max_proj, _, vmax = _gray_range(analyzer)
    z_proj = img_zscore_convert(max_proj.astype(np.float32), analyzer.bg_center, analyzer.bg_sigma)
    fig = Figure(figsize=(16, 7), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.6])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_map.imshow(z_proj, cmap="gray", vmin=MAP_Z_MIN, vmax=vmax)
    for name, k in picks.items():
        mask = np.zeros(z_proj.shape, dtype=bool)
        mask[analyzer.footprints[k][:, 0], analyzer.footprints[k][:, 1]] = True
        ax_map.contour(mask.astype(float), levels=[0.5], colors=[COLORS[name]], linewidths=2.5)
        cy, cx = analyzer.footprints[k].mean(axis=0)
        ax_map.text(cx, cy, name, color="white", fontsize=14, fontweight="bold", ha="center", va="center",
                    bbox={"boxstyle": "circle", "fc": COLORS[name]})
    ax_map.set_title(f"{STEM}: max projection + flash footprints\nA, B: recur_zone {z_id}   C: recur_zone {c_id}")
    ax_map.axis("off")

    t = np.arange(analyzer.n_frames) / fps
    for row, (name, k) in enumerate(picks.items()):
        ax = fig.add_subplot(grid[row, 1])
        ax.plot(t, traces[k], color=COLORS[name], lw=1)
        ax.axvline((det.at[k, "frame"] - 1) / fps, color="black", ls=":", lw=1)  # this flash's own frame
        ax.set_ylabel("ALS value", fontsize=9)
        label = "r = 1 (itself)" if name == "A" else f"r with A = {r_all[a, k]:.3f}"
        ax.set_title(f"{name}: recur_zone {zone_of[name]}, frame {det.at[k, 'frame']}  |  {label}",
                     fontsize=11, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row == 2:
            ax.set_xlabel("time (s)")
    fig.suptitle("Trace = mean ALS value inside each flash's own footprint, every frame; r = Pearson, 0 lag. "
                 "Dotted line = the flash's own frame.", fontsize=11)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
