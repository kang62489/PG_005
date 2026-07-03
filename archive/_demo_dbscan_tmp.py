"""
Throwaway test: sanity-check the real RegionAnalyzer (DBSCAN clustering +
ring-trace + latency pipeline) against real CAT/MED tif segments.

Deleted after use.
"""

import sqlite3
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from classes.region_analyzer import RegionAnalyzer, _peak_offset_from_spike, eps_and_min_samples

CAT_DIR = Path("results/bk_new/categorized")
MED_DIR = Path("results/bk_new/median")
OUT_DIR = Path("output/dbscan_demo")
REC_DB = Path("data/rec_data.db")

FRAME_DURATION_MS = 100.0  # placeholder -- real value comes from AbfClip.ts_imgs, not available in this test

CAT_CMAP = ListedColormap(["#000000", "#888888", "#ffffff"])  # bg=black, dim=gray, bright=white
CLUSTER_RGBA = [
    (0.91, 0.30, 0.24, 0.45),  # red
    (0.18, 0.80, 0.44, 0.45),  # green
    (0.20, 0.60, 0.86, 0.45),  # blue
    (0.95, 0.61, 0.07, 0.45),  # orange
    (0.61, 0.35, 0.71, 0.45),  # purple
]


# ---------------------------------------------------------------------------
# OBJ lookup
# ---------------------------------------------------------------------------

def lookup_obj(tif_path: Path) -> str:
    """Query rec_data.db for the objective used in this recording."""
    stem = tif_path.stem  # 2025_10_13-0029_A1S2LC1_BIEXP_GAUSS_CAT
    base = stem.split("_BIEXP")[0].split("_MOV")[0]  # 2025_10_13-0029_A1S2LC1
    date, rest = base.split("-", 1)  # date=2025_10_13, rest=0029_A1S2LC1
    serial = rest.split("_")[0]  # 0029
    filename_key = f"{date}-{serial}.tif"  # 2025_10_13-0029.tif
    rec_table = f"REC_{date}"
    try:
        with sqlite3.connect(REC_DB) as conn:
            cur = conn.execute(f'SELECT OBJ FROM "{rec_table}" WHERE Filename = ?', (filename_key,))
            row = cur.fetchone()
        return row[0] if row else "10X"
    except Exception:
        return "10X"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_frame(ax, frame: np.ndarray, title: str) -> None:
    ax.imshow(frame, cmap=CAT_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _draw_cluster_shading(ax, label_frame: np.ndarray, centroids: list[tuple[float, float]]) -> None:
    """Semi-transparent color fill per cluster, centroid cross + index label."""
    h, w = label_frame.shape
    overlay = np.zeros((h, w, 4), dtype=float)
    n_clusters = int(label_frame.max()) + 1 if label_frame.max() >= 0 else 0
    for k in range(n_clusters):
        r, g, b, a = CLUSTER_RGBA[k % len(CLUSTER_RGBA)]
        overlay[label_frame == k] = (r, g, b, a)
    ax.imshow(overlay, interpolation="nearest")
    for k, (row, col) in enumerate(centroids):
        ax.plot(col, row, "+", color="black", markersize=20, markeredgewidth=4)
        ax.plot(col, row, "+", color="white", markersize=18, markeredgewidth=2.5)
        ax.text(col + 5, row - 5, str(k), color="white", fontsize=9, fontweight="bold")


def _draw_ring_map(ax, cat_frame: np.ndarray, clusters: list[dict], highlight: set[int], title: str) -> None:
    """Circle overlay for every cluster on one panel, each in its own color.

    1 cluster: inner (dashed) + outer (solid) ring pair, per compute_ring_traces.
    >1 clusters: single R circle (solid) per cluster, no ring split -- matches
    compute_cluster_trace, which represents each cluster as one whole region.
    """
    ax.imshow(cat_frame, cmap=CAT_CMAP, vmin=0, vmax=2, interpolation="nearest")
    h, w = cat_frame.shape
    is_single = "inner_mask" in clusters[0]

    overlay = np.zeros((h, w, 4), dtype=float)
    for i, c in enumerate(clusters):
        r, g, b, _ = CLUSTER_RGBA[i % len(CLUSTER_RGBA)]
        if is_single:
            overlay[c["inner_mask"]] = (r, g, b, 0.55)
            overlay[c["outer_mask"]] = (r, g, b, 0.28)
        else:
            overlay[c["mask"]] = (r, g, b, 0.4)
    ax.imshow(overlay, interpolation="nearest")

    for i, c in enumerate(clusters):
        edge_color = CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3]
        row_c, col_c = c["centroid"]
        R = c["R_px"]
        lw = 1.8 if i in highlight else 1.0
        rings = [(R / np.sqrt(2), "--"), (R, "-")] if is_single else [(R, "-")]
        for r_ring, ls in rings:
            circle = Circle((col_c, row_c), r_ring, fill=False, edgecolor=edge_color, linewidth=lw, linestyle=ls)
            ax.add_patch(circle)
        ax.plot(col_c, row_c, "+", color="black", markersize=12, markeredgewidth=2.5)
        ax.plot(col_c, row_c, "+", color=edge_color, markersize=10, markeredgewidth=1.5)

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _offset_label(offset: int) -> str:
    return "spike" if offset == 0 else f"spike{offset:+d}"


def save_main_figure(
    path: Path,
    stem: str,
    obj: str,
    min_samples: int,
    cat_stack: np.ndarray,
    si: int,
    ai: int,
    area_pct: np.ndarray,
    label_frame: np.ndarray,
    centroids: list[tuple[float, float]],
    n_raw_clusters: int,
) -> None:
    n_clusters = int(label_frame.max()) + 1 if int(label_frame.max()) >= 0 else 0

    fig = Figure(figsize=(20, 8), dpi=120)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 2.5], hspace=0.55, wspace=0.08)

    # --- Row 0: B+D% line plot (x = frame offset relative to spike, 0 = spike) ---
    ax_bd = fig.add_subplot(gs[0, :])
    ax_bd.plot(np.arange(len(area_pct)) - si, area_pct, color="#3498db", linewidth=1.6,
               marker="o", markersize=3.5)

    for fi, label, color in [
        (si - 1, f"spike-1: {area_pct[si-1]:.2f}%" if si > 0 else "spike-1 (OOB)", "#888888"),
        (si,     f"spike:   {area_pct[si]:.2f}%", "#e74c3c"),
        (si + 1, f"spike+1: {area_pct[si+1]:.2f}%" if si+1 < len(area_pct) else "spike+1 (OOB)", "#f39c12"),
    ]:
        if 0 <= fi < len(area_pct):
            ax_bd.axvline(fi - si, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=label)

    ax_bd.plot(ai - si, area_pct[ai], "*", color="white", markersize=14, markeredgecolor="black",
               markeredgewidth=1, zorder=5, label=f"DBSCAN run on frame {ai}")

    ax_bd.set_xlabel("Frame offset from spike (0 = spike)", fontsize=9)
    ax_bd.set_ylabel("B+D  %", fontsize=9)
    ax_bd.set_title(
        "B+D% per frame  |  star = frame used for DBSCAN (baseline+10σ threshold, spike or spike+1)",
        fontsize=9,
    )
    ax_bd.legend(fontsize=8, loc="upper right")
    ax_bd.tick_params(labelsize=8)

    # --- Row 1: spike-1 .. spike+4 panels; DBSCAN shading only on the ai panel ---
    for col_i, offset in enumerate(range(-1, 5)):
        fi = si + offset
        ax = fig.add_subplot(gs[1, col_i])
        if 0 <= fi < cat_stack.shape[0]:
            tag = "  ★" if fi == ai else ""
            title = f"{_offset_label(offset)}  {area_pct[fi]:.2f}%{tag}"
            _draw_frame(ax, cat_stack[fi], title)
            if fi == ai:
                _draw_cluster_shading(ax, label_frame, centroids)
        else:
            ax.set_title(f"{_offset_label(offset)}\n(out of range)", fontsize=9)
            ax.axis("off")

    # Row-1 title: DBSCAN settings
    _, tops, _, _ = gs.get_grid_positions(fig)
    settings_text = (
        f"RegionAnalyzer — OBJ={obj}  min_samples={min_samples}  "
        f"found {n_raw_clusters} → kept {n_clusters}"
    )
    fig.text(0.5, tops[1] + 0.015, settings_text, ha="center", va="bottom",
              fontsize=9, fontweight="bold")

    fig.suptitle(f"{stem}  |  OBJ={obj}", fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")


def save_ring_figure(
    path: Path,
    stem: str,
    obj: str,
    cat_stack: np.ndarray,
    si: int,
    ai: int,
    clusters: list[dict],
    frame_duration_ms: float,
) -> None:
    """Ring/circle maps + z-score traces for every cluster.

    1 cluster: inner/outer ring split, latency = outer peak - inner peak
    (spread within the one release site).
    >1 clusters: single R circle per cluster, latency = latest cluster peak
    time - earliest cluster peak time (asynchrony across release sites);
    the earliest and latest clusters are highlighted.
    """
    if not clusters:
        fig = Figure(figsize=(10, 4), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No cluster detected — ring analysis skipped",
                 ha="center", va="center", fontsize=12, color="#888888", transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(f"{stem}  |  OBJ={obj}  |  ring analysis", fontsize=11, fontweight="bold")
        fig.savefig(path, bbox_inches="tight")
        return

    is_single = "inner_mask" in clusters[0]

    if is_single:
        c = clusters[0]
        inner_peak_rel = _peak_offset_from_spike(c["inner_trace"], si)
        outer_peak_rel = _peak_offset_from_spike(c["outer_trace"], si)
        latency = None
        if inner_peak_rel is not None and outer_peak_rel is not None:
            latency = (outer_peak_rel - inner_peak_rel) * frame_duration_ms
        highlight = {0}
    else:
        peak_rels = [(i, _peak_offset_from_spike(c["trace"], si)) for i, c in enumerate(clusters)]
        valid = [(i, p) for i, p in peak_rels if p is not None]
        latency = None
        highlight = set()
        if len(valid) >= 2:
            earliest_i, earliest_p = min(valid, key=lambda pair: pair[1])
            latest_i, latest_p = max(valid, key=lambda pair: pair[1])
            latency = (latest_p - earliest_p) * frame_duration_ms
            highlight = {earliest_i, latest_i}

    latency_label = f"{latency:.1f} ms" if latency is not None else "n/a"

    fig = Figure(figsize=(22, 8.5), dpi=120)
    gs = fig.add_gridspec(2, 9, height_ratios=[2.2, 1.6], hspace=0.45, wspace=0.08)

    # --- Row 0: spike-4 .. spike+4 panels, every cluster's circle(s) overlaid on every panel ---
    for col_i, offset in enumerate(range(-4, 5)):
        fi = si + offset
        ax = fig.add_subplot(gs[0, col_i])
        if 0 <= fi < cat_stack.shape[0]:
            tag = "  [DBSCAN frame]" if fi == ai else ""
            _draw_ring_map(ax, cat_stack[fi], clusters, highlight, f"{_offset_label(offset)}{tag}")
        else:
            ax.set_title(f"{_offset_label(offset)}\n(out of range)", fontsize=8)
            ax.axis("off")

    # --- Row 1: full-segment z-score traces, x-axis relative to spike (0 = spike) ---
    ax_tr = fig.add_subplot(gs[1, :])
    if is_single:
        c = clusters[0]
        n_frames = len(c["inner_trace"])
        offsets = np.arange(n_frames) - si
        split_um = c["R_um"] / np.sqrt(2)
        ax_tr.plot(offsets, c["inner_trace"], color="#e74c3c", linewidth=1.8, label=f"inner (0–{split_um:.1f}µm)")
        ax_tr.plot(offsets, c["outer_trace"], color="#3498db", linewidth=1.8,
                   label=f"outer ({split_um:.1f}–{c['R_um']:.1f}µm)")
        trace_title = f"Ring z-score traces — 1 cluster (red=inner  blue=outer)  |  latency = {latency_label}"
    else:
        n_frames = len(clusters[0]["trace"])
        offsets = np.arange(n_frames) - si
        for i, c in enumerate(clusters):
            color = CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3]
            lw = 2.2 if i in highlight else 1.2
            alpha = 1.0 if i in highlight else 0.55
            ax_tr.plot(offsets, c["trace"], color=color, linewidth=lw, alpha=alpha,
                       label=f"cluster {i} (R={c['R_um']:.1f}µm)")
        trace_title = (
            f"Whole-cluster z-score traces — all {len(clusters)} clusters overlaid, "
            f"no ring split  |  latency = {latency_label}"
        )

    window_lo = max(0, si - 4) - si
    window_hi = min(n_frames - 1, si + 4) - si
    ax_tr.axvspan(window_lo, window_hi, color="#f1c40f", alpha=0.12, label="panels shown above")

    for fi, color in [(-1, "#888888"), (0, "#e74c3c"), (1, "#f39c12")]:
        if 0 <= si + fi < n_frames:
            ax_tr.axvline(fi, color=color, linestyle=":", linewidth=1.0, alpha=0.6)

    ax_tr.set_xlabel("Frame offset from spike (0 = spike)", fontsize=9)
    ax_tr.set_ylabel("Mean z-score", fontsize=9)
    ax_tr.set_title(trace_title, fontsize=9)
    ax_tr.legend(fontsize=7, loc="upper right", ncol=2)
    ax_tr.tick_params(labelsize=8)
    ax_tr.grid(True, alpha=0.3)

    fig.suptitle(f"{stem}  |  OBJ={obj}", fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict]) -> None:
    header = f"{'Recording':<35} {'OBJ':>4} {'raw':>5} {'kept':>5} {'spike%':>7} {'frame':>8} {'latency_ms':>11}"
    print(f"\n{'-'*90}")
    print(header)
    print(f"{'-'*90}")
    for r in rows:
        latency_str = f"{r['latency_ms']:.1f}" if r["latency_ms"] is not None else "n/a"
        print(
            f"  {r['stem']:<33} {r['obj']:>4} {r['n_raw']:>5} {r['n_kept']:>5} "
            f"{r['signal']:>7.2f} {r['frame']:>8} {latency_str:>11}"
        )
    print(f"{'-'*90}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = [
        CAT_DIR / "2025_10_13-0029_A1S2LC1_BIEXP_GAUSS_CAT.tif",
        CAT_DIR / "2025_11_08-0027_A1S2RC1_BIEXP_GAUSS_CAT.tif",
        CAT_DIR / "2025_06_11-0005_A1S4RC1_BIEXP_GAUSS_CAT.tif",
        CAT_DIR / "2025_12_15-0013_A1S3RC1_BIEXP_GAUSS_CAT.tif",
        CAT_DIR / "2025_04_03-0033_A1S2RC1_BIEXP_GAUSS_CAT.tif",
    ]
    picks = [p for p in picks if p.exists()]

    print(f"Analyzing {len(picks)} files:")
    for p in picks:
        print(f"  {p.name}")

    rows = []
    for tif_path in picks:
        stem = tif_path.stem
        cat_stack = tifffile.imread(tif_path)
        med_path = MED_DIR / tif_path.name.replace("_CAT.tif", "_MED.tif")
        med_stack = tifffile.imread(med_path).astype(np.float32)
        si = cat_stack.shape[0] // 2

        obj = lookup_obj(tif_path)

        try:
            analyzer = RegionAnalyzer(cat_stack, med_stack, si, obj=obj)
        except Exception as exc:
            print(f"  [skip] {stem}: {exc}")
            continue

        ai = analyzer.critical_frame_idx
        n_kept = len(analyzer.centroids)
        latency_ms = analyzer.get_peak_latency_ms(FRAME_DURATION_MS)
        _, min_samples = eps_and_min_samples(obj)

        rows.append({
            "stem":       stem.replace("_BIEXP_GAUSS_CAT", ""),
            "obj":        obj,
            "n_raw":      analyzer.n_raw_clusters,
            "n_kept":     n_kept,
            "signal":     analyzer.area_pct[ai],
            "frame":      "spike+1" if ai != si else "spike",
            "latency_ms": latency_ms,
        })

        out_png = OUT_DIR / f"{stem}_ideas_demo.png"
        save_main_figure(
            out_png, stem, obj, min_samples, cat_stack, si, ai,
            analyzer.area_pct, analyzer.label_frame, analyzer.centroids, analyzer.n_raw_clusters,
        )

        out_ring_png = OUT_DIR / f"{stem}_rings_demo.png"
        save_ring_figure(out_ring_png, stem, obj, cat_stack, si, ai, analyzer.clusters, FRAME_DURATION_MS)

    print_summary_table(rows)
    print(f"\nFigures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
