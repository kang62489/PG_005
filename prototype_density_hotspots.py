import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import uniform_filter
from skimage.measure import label

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from classes import AbfClip  # noqa: E402
from classes.region_analyzer import _run_cluster_seeker, compute_eps_px  # noqa: E402
from functions import zscore_img_segs  # noqa: E402

# ── Tunables ─────────────────────────────────────────────────────────────────
WINDOW_PX = 200
DENSITY_THRESH = 0.15

CLUSTER_RGBA = [
    (0.91, 0.30, 0.24, 0.55),
    (0.18, 0.80, 0.44, 0.55),
    (0.20, 0.60, 0.86, 0.55),
    (0.95, 0.61, 0.07, 0.55),
    (0.61, 0.35, 0.71, 0.55),
    (0.10, 0.74, 0.61, 0.55),
]

OUT_DIR = PROJECT_ROOT / "output" / "density_hotspots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECORDINGS = [
    ("2025_06_11-0003_BIEXP_GAUSS.tif", "2025_06_11_0004.abf", "10X"),
    ("2025_12_15-0012_BIEXP_GAUSS.tif", "2025_12_15_0008.abf", "10X"),
]


def compute_threshold(segment: np.ndarray, spike_frame_idx: int) -> float:
    baseline_pixels = np.concatenate([segment[i].flatten() for i in range(spike_frame_idx)])
    return float(baseline_pixels.mean() + 2 * baseline_pixels.std())


def run_density_hotspot(spike_frame: np.ndarray, threshold_used: float, obj: str) -> dict:
    bright_mask = spike_frame > threshold_used
    density = uniform_filter(bright_mask.astype(float), size=WINDOW_PX, mode="constant", cval=0.0)
    qualifies = density >= DENSITY_THRESH
    hotspot_mask = bright_mask & qualifies

    eps_px = compute_eps_px(obj)
    gated_label_frame, centroids, n_raw = _run_cluster_seeker(hotspot_mask.astype(int), eps_px, z_frame=spike_frame)
    n_clusters = len(centroids)

    # _run_cluster_seeker only ever sees hotspot_mask (bright_mask & qualifies), so a real
    # bright pixel whose OWN local density dipped just under DENSITY_THRESH gets excluded
    # even when it's directly touching an accepted cluster. Expand each accepted cluster
    # back out to the full original bright_mask blob(s) it touches, so no connected bright
    # pixel gets left out just because of a marginal per-pixel density value.
    labeled_bright = label(bright_mask)
    label_frame = np.full(bright_mask.shape, -2, dtype=int)
    for cluster_id in range(n_clusters):
        cluster_mask = gated_label_frame == cluster_id
        touched_blob_ids = set(labeled_bright[cluster_mask].tolist()) - {0}
        whole_mask = np.isin(labeled_bright, list(touched_blob_ids))
        label_frame[whole_mask] = cluster_id

    return {
        "bright_mask": bright_mask,
        "density": density,
        "qualifies": qualifies,
        "hotspot_mask": hotspot_mask,
        "label_frame": label_frame,
        "centroids": centroids,
        "n_raw": n_raw,
        "n_clusters": n_clusters,
        "detected": n_clusters > 0,
    }


def save_workflow_figure(path: Path, title: str, result: dict) -> None:
    fig = Figure(figsize=(4.5, 4.5), dpi=130)
    ax = fig.subplots(1, 1)
    ax.imshow(result["bright_mask"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    label_frame = result["label_frame"]
    h, w = label_frame.shape
    overlay = np.zeros((h, w, 4))
    for k in range(result["n_clusters"]):
        overlay[label_frame == k] = CLUSTER_RGBA[k % len(CLUSTER_RGBA)]
    ax.imshow(overlay, interpolation="nearest")

    for cluster_id, (row_c, col_c) in enumerate(result["centroids"]):
        ax.plot(col_c, row_c, "+", color="black", markersize=14, markeredgewidth=3)
        ax.plot(col_c, row_c, "+", color="white", markersize=12, markeredgewidth=1.8)
        ax.text(col_c + 8, row_c - 8, str(cluster_id), color="white", fontsize=10, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\nn_clusters={result['n_clusters']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


def save_recording_montage(path: Path, rec_stem: str, seg_results: list[dict]) -> None:
    n = len(seg_results)
    ncols = 8
    nrows = int(np.ceil(n / ncols))
    fig = Figure(figsize=(ncols * 2.0, nrows * 2.0), dpi=130)
    axes = fig.subplots(nrows, ncols).flatten()

    n_detected = sum(r["detected"] for r in seg_results)
    for seg_idx, result in enumerate(seg_results):
        ax = axes[seg_idx]
        ax.imshow(result["bright_mask"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        label_frame = result["label_frame"]
        h, w = label_frame.shape
        overlay = np.zeros((h, w, 4))
        for k in range(result["n_clusters"]):
            overlay[label_frame == k] = CLUSTER_RGBA[k % len(CLUSTER_RGBA)]
        ax.imshow(overlay, interpolation="nearest")

        color = "limegreen" if result["detected"] else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"seg{seg_idx:02d} n={result['n_clusters']}", fontsize=7, color=color)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{rec_stem} — window_px={WINDOW_PX} (fixed), density>={DENSITY_THRESH}\n"
        f"reliability: {n_detected}/{n} = {n_detected / n:.1%}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    for proc_tiff_name, abf_name, obj in RECORDINGS:
        proc_tiff_path = PROJECT_ROOT / "proc_tiffs" / proc_tiff_name
        raw_abf_path = PROJECT_ROOT / "raw_abfs" / abf_name
        rec_stem = proc_tiff_path.stem

        clip = AbfClip(
            proc_tiff_path=proc_tiff_path,
            raw_abf_path=raw_abf_path,
            results_dir=PROJECT_ROOT / "results",
            detrend_mode="BIEXP",
            normalization="GAUSS",
        )
        lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)

        seg_results = []
        for segment in lst_zscore:
            spike_frame_idx = segment.shape[0] // 2
            threshold_used = compute_threshold(segment, spike_frame_idx)
            result = run_density_hotspot(segment[spike_frame_idx], threshold_used, obj)
            seg_results.append(result)

        n_detected = sum(r["detected"] for r in seg_results)
        n = len(seg_results)
        print(f"{rec_stem}: reliability = {n_detected}/{n} = {n_detected / n:.1%}")

        for seg_idx, result in enumerate(seg_results):
            save_workflow_figure(
                OUT_DIR / f"{rec_stem}_workflow_seg{seg_idx:02d}.png",
                f"{rec_stem}  seg{seg_idx:02d}  (workflow)",
                result,
            )

        save_recording_montage(OUT_DIR / f"{rec_stem}_density_prototype_montage.png", rec_stem, seg_results)

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
