"""
Throwaway demo: validate two detection/filtering ideas using CAT.tif segments.

  Idea 1 — B+D% detection criterion:
    plot bright+dim pixel % across all frames; a real ACh event shows a clear
    elevation at spike_frame (or spike_frame+1) vs spike_frame-1 (baseline).

  Idea 2 — cluster size filter:
    after DBSCAN, keep only clusters >= MIN_CLUSTER_FRACTION of all non-bg px
    in the analysis frame.

DBSCAN is run on whichever of spike / spike+1 has the higher B+D% (delayed
signal fix): the chosen frame is marked with a star in the B+D plot and a
"[DBSCAN]" tag in the panel title.

Deleted after use.
"""

import sqlite3
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN

CAT_DIR = Path("results/bk_new/categorized")
MED_DIR = Path("results/bk_new/median")
OUT_DIR = Path("output/dbscan_demo")
REC_DB  = Path("data/rec_data.db")

EPS_UM           = 10.0   # inter-varicosity gap in µm (tunable)
MIN_DENSITY_FRAC = 0.1    # min_samples = 10% of eps-circle area
MIN_CLUSTER_FRACTION = 0.05

PIXEL_SCALE = {"10X": 0.75, "40X": 3.0, "60X": 4.5}  # px/µm

CAT_BG = 0
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
    stem = tif_path.stem                          # 2025_10_13-0029_A1S2LC1_BIEXP_GAUSS_CAT
    base = stem.split("_BIEXP")[0].split("_MOV")[0]  # 2025_10_13-0029_A1S2LC1
    date, rest = base.split("-", 1)               # date=2025_10_13, rest=0029_A1S2LC1
    serial = rest.split("_")[0]                   # 0029
    filename_key = f"{date}-{serial}.tif"          # 2025_10_13-0029.tif
    rec_table = f"REC_{date}"
    try:
        with sqlite3.connect(REC_DB) as conn:
            cur = conn.execute(f'SELECT OBJ FROM "{rec_table}" WHERE Filename = ?', (filename_key,))
            row = cur.fetchone()
        return row[0] if row else "unknown"
    except Exception:
        return "unknown"


def eps_and_min_samples(obj: str) -> tuple[int, int]:
    """Convert EPS_UM to pixels for this objective, compute min_samples."""
    px_per_um = PIXEL_SCALE.get(obj, 1.0)
    eps_px = int(EPS_UM * px_per_um)
    min_samples = max(1, int(MIN_DENSITY_FRAC * 3.14159 * eps_px ** 2))
    return eps_px, min_samples


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_bd_pct(stack: np.ndarray) -> np.ndarray:
    """Fraction of non-background pixels per frame, in %."""
    total_px = stack.shape[1] * stack.shape[2]
    return np.array([np.sum(frame > CAT_BG) / total_px * 100 for frame in stack])


def run_dbscan_filtered(
    analysis_frame: np.ndarray,
    eps_px: int,
    min_samples: int,
) -> tuple[np.ndarray, list[tuple[float, float]], int]:
    """
    DBSCAN on all non-bg pixels, then keep only clusters whose size
    >= MIN_CLUSTER_FRACTION of total non-bg pixels in this frame.

    Returns:
      label_frame   : (H, W)  cluster index (0-based, largest-first) at non-bg px,
                               -1 = noise/rejected, -2 = background
      centroids     : list of (row, col) per kept cluster
      n_raw_clusters: total clusters found by DBSCAN before size filter
    """
    coords = np.argwhere(analysis_frame > CAT_BG)
    label_frame = np.full(analysis_frame.shape, -2, dtype=int)
    if coords.shape[0] == 0:
        return label_frame, [], 0

    raw_labels = DBSCAN(eps=eps_px, min_samples=min_samples).fit_predict(coords)

    total_non_bg = coords.shape[0]
    min_cluster_px = int(total_non_bg * MIN_CLUSTER_FRACTION)
    unique, counts = np.unique(raw_labels[raw_labels >= 0], return_counts=True)
    n_raw_clusters = len(unique)

    kept = []
    for k, c in zip(unique.tolist(), counts.tolist()):
        if c / total_non_bg >= MIN_CLUSTER_FRACTION:
            kept.append((k, c))
    kept.sort(key=lambda x: x[1], reverse=True)  # largest first

    remapped = np.full_like(raw_labels, -1)
    for new_k, (old_k, _) in enumerate(kept):
        remapped[raw_labels == old_k] = new_k

    label_frame[coords[:, 0], coords[:, 1]] = remapped

    centroids = []
    for new_k in range(len(kept)):
        pts = coords[remapped == new_k]
        centroids.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))

    return label_frame, centroids, n_raw_clusters, total_non_bg, min_cluster_px


def pick_analysis_frame(bd_pct: np.ndarray, si: int) -> int:
    """Return spike or spike+1, whichever has higher B+D%."""
    next_i = si + 1
    if next_i >= len(bd_pct):
        return si
    return si if bd_pct[si] >= bd_pct[next_i] else next_i


def compute_ring_traces(
    label_frame: np.ndarray,
    centroid: tuple[float, float],
    med_stack: np.ndarray,
    cluster_k: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, bool]:
    """
    R is the enclosing-circle radius (max centroid distance among cluster pixels).
    Cluster pixels are split into equal-area inner/outer rings at R/√2:
    inner = 0 <= r <= R/√2, outer = R/√2 < r <= R. Mean z-score from med_stack
    is computed per ring per frame.

    R only sees pixels inside the frame, so a cluster touching the image edge
    is truncated and R underestimates its true extent — flagged via the
    returned `touches_boundary`.

    Returns: inner_trace, outer_trace, R, inner_mask, outer_mask, touches_boundary
    """
    coords = np.argwhere(label_frame == cluster_k)
    row_c, col_c = centroid
    dists = np.sqrt((coords[:, 0] - row_c) ** 2 + (coords[:, 1] - col_c) ** 2)
    R = float(dists.max())
    split = R / np.sqrt(2)

    H, W = label_frame.shape
    touches_boundary = bool(
        np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == H - 1)
        or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == W - 1)
    )

    inner_mask = np.zeros((H, W), dtype=bool)
    outer_mask = np.zeros((H, W), dtype=bool)
    inner_mask[coords[dists <= split, 0], coords[dists <= split, 1]] = True
    outer_mask[coords[dists >  split, 0], coords[dists >  split, 1]] = True

    n_frames = med_stack.shape[0]
    inner_trace = np.array([med_stack[f][inner_mask].mean() if inner_mask.any() else np.nan for f in range(n_frames)])
    outer_trace = np.array([med_stack[f][outer_mask].mean() if outer_mask.any() else np.nan for f in range(n_frames)])

    return inner_trace, outer_trace, R, inner_mask, outer_mask, touches_boundary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_frame(ax, frame: np.ndarray, title: str) -> None:
    ax.imshow(frame, cmap=CAT_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _draw_cluster_shading(
    ax,
    label_frame: np.ndarray,
    centroids: list[tuple[float, float]],
) -> None:
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
        ax.text(
            col + 5, row - 5, str(k), color="white", fontsize=9, fontweight="bold",
            path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"])
                          .withStroke(linewidth=2, foreground="black")],
        )


def _draw_ring_map(
    ax,
    cat_frame: np.ndarray,
    inner_mask: np.ndarray,
    outer_mask: np.ndarray,
    centroid: tuple[float, float],
    R: float,
    title: str,
) -> None:
    ax.imshow(cat_frame, cmap=CAT_CMAP, vmin=0, vmax=2, interpolation="nearest")
    h, w = cat_frame.shape
    overlay = np.zeros((h, w, 4), dtype=float)
    overlay[inner_mask] = (0.95, 0.30, 0.25, 0.6)  # red  — inner
    overlay[outer_mask] = (0.20, 0.55, 0.90, 0.6)  # blue — outer
    ax.imshow(overlay, interpolation="nearest")
    row_c, col_c = centroid
    for r_ring, ls in [(R / np.sqrt(2), "--"), (R, "-")]:
        circle = __import__("matplotlib.patches", fromlist=["Circle"]).Circle(
            (col_c, row_c), r_ring, fill=False, edgecolor="white", linewidth=1.2, linestyle=ls
        )
        ax.add_patch(circle)
    ax.plot(col_c, row_c, "+", color="black", markersize=16, markeredgewidth=3)
    ax.plot(col_c, row_c, "+", color="white", markersize=14, markeredgewidth=2)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _offset_label(offset: int) -> str:
    if offset == 0:
        return "spike"
    return f"spike{offset:+d}"


def save_main_figure(
    path: Path,
    stem: str,
    obj: str,
    min_samples: int,
    stack: np.ndarray,
    si: int,
    ai: int,
    bd_pct: np.ndarray,
    label_frame: np.ndarray,
    centroids: list[tuple[float, float]],
    n_raw_clusters: int,
) -> None:
    n_clusters = int(label_frame.max()) + 1 if int(label_frame.max()) >= 0 else 0

    fig = Figure(figsize=(20, 8), dpi=120)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 2.5], hspace=0.55, wspace=0.08)

    # --- Row 0: B+D% line plot (x = frame offset relative to spike, 0 = spike) ---
    ax_bd = fig.add_subplot(gs[0, :])
    ax_bd.plot(np.arange(len(bd_pct)) - si, bd_pct, color="#3498db", linewidth=1.6,
               marker="o", markersize=3.5)

    for fi, label, color in [
        (si - 1, f"spike-1: {bd_pct[si-1]:.2f}%"  if si > 0              else "spike-1 (OOB)", "#888888"),
        (si,     f"spike:   {bd_pct[si]:.2f}%",                                                  "#e74c3c"),
        (si + 1, f"spike+1: {bd_pct[si+1]:.2f}%"  if si+1 < len(bd_pct) else "spike+1 (OOB)",  "#f39c12"),
    ]:
        if 0 <= fi < len(bd_pct):
            ax_bd.axvline(fi - si, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=label)

    ax_bd.plot(ai - si, bd_pct[ai], "*", color="white", markersize=14, markeredgecolor="black",
               markeredgewidth=1, zorder=5, label=f"DBSCAN run on frame {ai}")

    ax_bd.set_xlabel("Frame offset from spike (0 = spike)", fontsize=9)
    ax_bd.set_ylabel("B+D  %", fontsize=9)
    ax_bd.set_title(
        "Idea 1 — B+D% per frame  |  star = frame used for DBSCAN (spike or spike+1, whichever higher)",
        fontsize=9,
    )
    ax_bd.legend(fontsize=8, loc="upper right")
    ax_bd.tick_params(labelsize=8)

    # --- Row 1: spike-1 .. spike+4 panels; DBSCAN shading only on the ai panel ---
    for col_i, offset in enumerate(range(-1, 5)):
        fi = si + offset
        ax = fig.add_subplot(gs[1, col_i])
        if 0 <= fi < stack.shape[0]:
            tag = "  ★" if fi == ai else ""
            title = f"{_offset_label(offset)}  {bd_pct[fi]:.2f}%{tag}"
            _draw_frame(ax, stack[fi], title)
            if fi == ai:
                _draw_cluster_shading(ax, label_frame, centroids)
        else:
            ax.set_title(f"{_offset_label(offset)}\n(out of range)", fontsize=9)
            ax.axis("off")

    # Row-1 title: DBSCAN settings (moved out of the figure suptitle)
    _, tops, _, _ = gs.get_grid_positions(fig)
    settings_text = (
        f"DBSCAN settings — OBJ={obj}  EPS={EPS_UM}µm  min_samples={min_samples}  "
        f"size_filter≥{MIN_CLUSTER_FRACTION*100:.0f}%  |  found {n_raw_clusters} → kept {n_clusters}"
    )
    fig.text(0.5, tops[1] + 0.015, settings_text, ha="center", va="bottom",
              fontsize=9, fontweight="bold")

    fig.suptitle(f"{stem}  |  OBJ={obj}", fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")


def save_ring_figure(
    path: Path,
    stem: str,
    obj: str,
    stack: np.ndarray,
    med_stack: np.ndarray,
    si: int,
    ai: int,
    label_frame: np.ndarray,
    centroids: list[tuple[float, float]],
) -> None:
    n_clusters = int(label_frame.max()) + 1 if int(label_frame.max()) >= 0 else 0

    fig = Figure(figsize=(22, 8), dpi=120)

    if n_clusters == 0:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No cluster detected — ring analysis skipped",
                 ha="center", va="center", fontsize=12, color="#888888",
                 transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(f"{stem}  |  OBJ={obj}  |  ring analysis", fontsize=11, fontweight="bold")
        fig.savefig(path, bbox_inches="tight")
        return

    inner_trace, outer_trace, R, inner_mask, outer_mask, touches_boundary = compute_ring_traces(
        label_frame, centroids[0], med_stack
    )
    split_r = R / np.sqrt(2)
    px_per_um = PIXEL_SCALE.get(obj, 1.0)
    R_um = R / px_per_um
    split_um = split_r / px_per_um
    boundary_tag = "  ⚠ boundary-cropped" if touches_boundary else ""

    gs = fig.add_gridspec(2, 9, height_ratios=[2.2, 1.6], hspace=0.4, wspace=0.08)

    # --- Row 0: spike-4 .. spike+4 panels, fixed ring overlay on every panel ---
    for col_i, offset in enumerate(range(-4, 5)):
        fi = si + offset
        ax = fig.add_subplot(gs[0, col_i])
        if 0 <= fi < stack.shape[0]:
            tag = "  [DBSCAN frame]" if fi == ai else ""
            _draw_ring_map(ax, stack[fi], inner_mask, outer_mask, centroids[0], R,
                            f"{_offset_label(offset)}{tag}")
        else:
            ax.set_title(f"{_offset_label(offset)}\n(out of range)", fontsize=8)
            ax.axis("off")

    # --- Row 1: full-segment ring z-score traces, with the -4..+4 window annotated ---
    ax_tr = fig.add_subplot(gs[1, :])
    n_frames = med_stack.shape[0]
    frames = np.arange(n_frames)
    ax_tr.plot(frames, inner_trace, color="#e74c3c", linewidth=1.8, label=f"inner (0–{split_um:.1f}µm)")
    ax_tr.plot(frames, outer_trace, color="#3498db", linewidth=1.8, label=f"outer ({split_um:.1f}–{R_um:.1f}µm)")

    window_lo = max(0, si - 4)
    window_hi = min(n_frames - 1, si + 4)
    ax_tr.axvspan(window_lo, window_hi, color="#f1c40f", alpha=0.15, label="panels shown above")

    for fi, color in [(si - 1, "#888888"), (si, "#e74c3c"), (si + 1, "#f39c12")]:
        if 0 <= fi < n_frames:
            ax_tr.axvline(fi, color=color, linestyle="--", linewidth=1.0, alpha=0.7)

    ax_tr.set_xlabel("Frame index", fontsize=9)
    ax_tr.set_ylabel("Mean z-score", fontsize=9)
    ax_tr.set_title("Full ring z-score traces — cluster 0  (red=inner  blue=outer)", fontsize=9)
    ax_tr.legend(fontsize=8, loc="upper right")
    ax_tr.tick_params(labelsize=8)
    ax_tr.grid(True, alpha=0.3)

    fig.suptitle(
        f"{stem}  |  OBJ={obj}  R={R_um:.1f}µm  split={split_um:.1f}µm{boundary_tag}",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict]) -> None:
    header = f"{'Recording':<35} {'OBJ':>4} {'eps':>4} {'spike%':>7} {'non_bg_px':>10} {'min_px(5%)':>11} {'raw':>5} {'kept':>5}"
    print(f"\n{'-'*93}")
    print(header)
    print(f"{'-'*93}")
    for r in rows:
        print(
            f"  {r['stem']:<33} {r['obj']:>4} {r['eps_px']:>4}px {r['signal']:>7.2f} "
            f"{r['total_non_bg']:>10,} {r['min_px']:>11,} "
            f"{r['n_raw']:>5} {r['n_clusters']:>5}"
        )
    print(f"{'-'*93}")


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
        stack = tifffile.imread(tif_path)
        med_path = MED_DIR / tif_path.name.replace("_CAT.tif", "_MED.tif")
        med_stack = tifffile.imread(med_path).astype(np.float32)
        si = stack.shape[0] // 2

        obj = lookup_obj(tif_path)
        eps_px, min_samples = eps_and_min_samples(obj)

        bd_pct = compute_bd_pct(stack)
        ai = pick_analysis_frame(bd_pct, si)
        base_pct = bd_pct[si - 1] if si > 0 else 0.0
        label_frame, centroids, n_raw, total_non_bg, min_cluster_px = run_dbscan_filtered(stack[ai], eps_px, min_samples)
        n_clusters = int(label_frame.max()) + 1 if int(label_frame.max()) >= 0 else 0

        rows.append({
            "stem":          stem.replace("_BIEXP_GAUSS_CAT", ""),
            "obj":           obj,
            "eps_px":        eps_px,
            "base":          base_pct,
            "signal":        bd_pct[ai],
            "elev":          bd_pct[ai] - base_pct,
            "frame":         "spike+1" if ai != si else "spike",
            "n_raw":         n_raw,
            "n_clusters":    n_clusters,
            "total_non_bg":  total_non_bg,
            "min_px":        min_cluster_px,
        })

        out_png = OUT_DIR / f"{stem}_ideas_demo.png"
        save_main_figure(out_png, stem, obj, min_samples, stack, si, ai, bd_pct, label_frame, centroids, n_raw)

        out_ring_png = OUT_DIR / f"{stem}_rings_demo.png"
        save_ring_figure(out_ring_png, stem, obj, stack, med_stack, si, ai, label_frame, centroids)

    print_summary_table(rows)
    print(f"\nFigures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
