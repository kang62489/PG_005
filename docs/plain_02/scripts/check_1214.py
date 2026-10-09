# ruff: noqa: INP001
"""
check_1214.py  --  Is the 99.6 % flash of 2025_12_14-0002 (60X, frame 962) real?

Steps
-----
1. ALS frames 958-966 on one gray range + the frame-962 mask outline.
2. Whole-frame mean of every frame (a frame-wide jump = artifact, not local release).

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/check_1214.py
"""
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[3]

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEM = "2025_12_14-0002_BIEXP_ALS"
FRAMES = range(958, 967)  # 1-based
FLASH_FRAME = 962
OUT_PATH = ROOT / "docs" / "plain_02" / "figures" / "check_2025_12_14-0002.png"


def main() -> None:
    """Steps 1-2."""
    stack = tifffile.imread(ROOT / "proc_tiffs" / f"{STEM}.tif").astype(np.float32)
    mask = tifffile.imread(ROOT / "output" / "plain_02" / "mask" / f"{STEM}_FLASH_MASK.tif")[FLASH_FRAME - 1] > 0
    vmin, vmax = np.percentile(stack[[f - 1 for f in FRAMES]], [1, 99])

    fig = Figure(figsize=(18, 9), layout="constrained")
    grid = fig.add_gridspec(2, len(FRAMES))
    for k, f in enumerate(FRAMES):
        ax = fig.add_subplot(grid[0, k])
        ax.imshow(stack[f - 1], cmap="gray", vmin=vmin, vmax=vmax)
        if f == FLASH_FRAME:
            ax.contour(mask.astype(float), levels=[0.5], colors="#E69F00", linewidths=1)
        ax.set_title(f"frame {f}", fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    frame_mean = stack.mean(axis=(1, 2))
    ax.plot(np.arange(1, len(frame_mean) + 1), frame_mean, color="#0072B2", lw=1)
    ax.axvline(FLASH_FRAME, color="#E69F00", ls=":", lw=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("whole-frame mean (ALS)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle(f"{STEM}: frames {FRAMES[0]}-{FRAMES[-1]} (same gray range), orange = frame-{FLASH_FRAME} mask; "
                 "bottom: whole-frame mean, dotted = frame 962")
    fig.savefig(OUT_PATH, dpi=100)
    print(f"saved {OUT_PATH}")
    print(f"frame mean at {FLASH_FRAME}: {frame_mean[FLASH_FRAME - 1]:.3f}; others median "
          f"{np.median(frame_mean):.3f}, 99.9th pct {np.percentile(frame_mean, 99.9):.3f}")


if __name__ == "__main__":
    main()
