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

from pathlib import Path

import numpy as np
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from skimage.measure import find_contours
from sklearn.cluster import DBSCAN

CAT_DIR = Path("results/bk_new/categorized")
OUT_DIR = Path("output/dbscan_demo")

EPS = 20
MIN_SAMPLES = 50
MIN_CLUSTER_FRACTION = 0.05   # keep cluster if >= 5% of non-bg pixels in analysis frame

CAT_BG = 0
CAT_CMAP = ListedColormap(["#111111", "#00bcd4", "#ffeb3b"])  # bg, dim, bright
CLUSTER_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_bd_pct(stack: np.ndarray) -> np.ndarray:
    """Fraction of non-background pixels per frame, in %."""
    total_px = stack.shape[1] * stack.shape[2]
    return np.array([np.sum(frame > CAT_BG) / total_px * 100 for frame in stack])


def run_dbscan_filtered(
    analysis_frame: np.ndarray,
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

    raw_labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(coords)

    total_non_bg = coords.shape[0]
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

    return label_frame, centroids, n_raw_clusters


def pick_analysis_frame(bd_pct: np.ndarray, si: int) -> int:
    """Return spike or spike+1, whichever has higher B+D%."""
    next_i = si + 1
    if next_i >= len(bd_pct):
        return si
    return si if bd_pct[si] >= bd_pct[next_i] else next_i


def get_cluster_contours(label_frame: np.ndarray, n_clusters: int) -> list[list]:
    """One list of contour arrays per cluster."""
    out = []
    for k in range(n_clusters):
        mask = (label_frame == k).astype(np.uint8)
        out.append(find_contours(mask, 0.5))
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_frame(ax, frame: np.ndarray, title: str) -> None:
    ax.imshow(frame, cmap=CAT_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _draw_contours(
    ax,
    contours_per_cluster: list[list],
    centroids: list[tuple[float, float]],
) -> None:
    for k, contours in enumerate(contours_per_cluster):
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.8)
        if k < len(centroids):
            row, col = centroids[k]
            ax.plot(col, row, "+", color="black", markersize=20, markeredgewidth=4)   # shadow
            ax.plot(col, row, "+", color="white", markersize=18, markeredgewidth=2.5) # marker
            ax.text(col + 5, row - 5, str(k), color="white", fontsize=9, fontweight="bold",
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"])
                                  .withStroke(linewidth=2, foreground="black")])


def save_figure(
    path: Path,
    stem: str,
    stack: np.ndarray,
    si: int,
    ai: int,
    bd_pct: np.ndarray,
    label_frame: np.ndarray,
    centroids: list[tuple[float, float]],
    n_raw_clusters: int,
) -> None:
    n_clusters = int(label_frame.max()) + 1 if int(label_frame.max()) >= 0 else 0
    contours_per_cluster = get_cluster_contours(label_frame, n_clusters)

    fig = Figure(figsize=(14, 8), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 2.5], hspace=0.4, wspace=0.06)

    # --- Row 0: B+D% line plot ---
    ax_bd = fig.add_subplot(gs[0, :])
    ax_bd.plot(np.arange(len(bd_pct)), bd_pct, color="#3498db", linewidth=1.6,
               marker="o", markersize=3.5)

    for fi, label, color in [
        (si - 1, f"spike-1: {bd_pct[si-1]:.2f}%"  if si > 0              else "spike-1 (OOB)", "#888888"),
        (si,     f"spike:   {bd_pct[si]:.2f}%",                                                  "#e74c3c"),
        (si + 1, f"spike+1: {bd_pct[si+1]:.2f}%"  if si+1 < len(bd_pct) else "spike+1 (OOB)",  "#f39c12"),
    ]:
        if 0 <= fi < len(bd_pct):
            ax_bd.axvline(fi, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=label)

    ax_bd.plot(ai, bd_pct[ai], "*", color="white", markersize=14, markeredgecolor="black",
               markeredgewidth=1, zorder=5, label=f"DBSCAN run on frame {ai}")

    ax_bd.set_xlabel("Frame index", fontsize=9)
    ax_bd.set_ylabel("B+D  %", fontsize=9)
    ax_bd.set_title(
        "Idea 1 — B+D% per frame  |  star = frame used for DBSCAN (spike or spike+1, whichever higher)",
        fontsize=9,
    )
    ax_bd.legend(fontsize=8, loc="upper right")
    ax_bd.tick_params(labelsize=8)

    # --- Row 1: spike-1 / spike / spike+1 panels ---
    dbscan_tag = f"[DBSCAN] raw:{n_raw_clusters} kept:{n_clusters}"
    panel_specs = [
        (si - 1, "spike-1  (baseline)"),
        (si,     f"spike  {bd_pct[si]:.2f}%{('  ' + dbscan_tag) if ai == si else ''}"),
        (si + 1, (f"spike+1  {bd_pct[si+1]:.2f}%{('  ' + dbscan_tag) if ai == si + 1 else ''}"
                  if si + 1 < len(bd_pct) else "spike+1 (OOB)")),
    ]
    for col_i, (fi, title) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[1, col_i])
        if 0 <= fi < stack.shape[0]:
            _draw_frame(ax, stack[fi], title)
            _draw_contours(ax, contours_per_cluster, centroids)
        else:
            ax.set_title(f"{title}\n(out of range)", fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"{stem}  |  DBSCAN eps={EPS}px  min_samples={MIN_SAMPLES}  "
        f"size_filter>={MIN_CLUSTER_FRACTION*100:.0f}% non-bg px  |  "
        f"DBSCAN found {n_raw_clusters}  ->  kept {n_clusters}",
        fontsize=10, fontweight="bold",
    )
    fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict]) -> None:
    header = f"{'Recording':<35} {'base%':>6} {'spike%':>7} {'elev%':>6} {'frame':>8} {'raw':>5} {'kept':>5}"
    print(f"\n{'-'*80}")
    print(header)
    print(f"{'-'*80}")
    for r in rows:
        print(
            f"  {r['stem']:<33} {r['base']:>6.2f} {r['signal']:>7.2f} "
            f"{r['elev']:>6.2f} {r['frame']:>8} {r['n_raw']:>5} {r['n_clusters']:>5}"
        )
    print(f"{'-'*80}")


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
        si = stack.shape[0] // 2

        bd_pct = compute_bd_pct(stack)
        ai = pick_analysis_frame(bd_pct, si)
        base_pct = bd_pct[si - 1] if si > 0 else 0.0
        label_frame, centroids, n_raw = run_dbscan_filtered(stack[ai])
        n_clusters = int(label_frame.max()) + 1 if int(label_frame.max()) >= 0 else 0

        rows.append({
            "stem":       stem.replace("_BIEXP_GAUSS_CAT", ""),
            "base":       base_pct,
            "signal":     bd_pct[ai],
            "elev":       bd_pct[ai] - base_pct,
            "frame":      "spike+1" if ai != si else "spike",
            "n_raw":      n_raw,
            "n_clusters": n_clusters,
        })

        out_png = OUT_DIR / f"{stem}_ideas_demo.png"
        save_figure(out_png, stem, stack, si, ai, bd_pct, label_frame, centroids, n_raw)

    print_summary_table(rows)
    print(f"\nFigures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
