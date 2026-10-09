# ruff: noqa: INP001
"""
arbor_hull.py  --  Axon-arbor area of the ChI in Aosaki & Kawaguchi 1996 Fig. b (Kang's screenshot).

Steps
-----
1. Crop panel b (inside the red box); dark pixels = drawn neuron.
2. Scale: length of the 100 um scale bar in px.
3. Remove the scale bar, its "100 um" text and the "b" label; keep the axon + soma.
4. Bounding box, fitted ellipse (same extent) and convex hull of the dark pixels -> um^2.

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/arbor_hull.py
"""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread
from scipy.spatial import ConvexHull

# ── CONFIG ────────────────────────────────────────────────────────────────────
FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "aosaki_kawaguchi_1996_fig1_full.png"  # full Fig. 1 page (1545 px wide)
PANEL = (slice(603, 1030), slice(240, 875))  # rows, cols of panel Ab (axon); below the tip of the Aa dendrite
BAR_BOX = (slice(596, 605), slice(680, 860))  # where the 100 um scale bar line is
CLEAR_BOXES = [(slice(590, 645), slice(680, 860))]  # scale bar + "100 um" text
DARK = 0.5  # gray < this = drawn line
OUT_PATH = Path(__file__).resolve().parents[1] / "figures" / "arbor_hull_check.png"


def measure() -> dict:
    """Steps 1-4 -> gray image, px/um, arbor pixels (xs, ys), hull, and areas in um^2."""
    img = imread(FIG_PATH)[..., :3].mean(axis=2)
    dark = img < DARK
    bar_cols = np.flatnonzero(dark[BAR_BOX].any(axis=0))
    px_per_um = (bar_cols.max() - bar_cols.min() + 1) / 100
    for rows, cols in CLEAR_BOXES:
        dark[rows, cols] = False
    panel = np.zeros_like(dark)
    panel[PANEL] = dark[PANEL]
    ys, xs = np.nonzero(panel)
    width_um, height_um = (np.ptp(xs) + 1) / px_per_um, (np.ptp(ys) + 1) / px_per_um
    hull = ConvexHull(np.c_[xs, ys])
    return {"img": img, "px_per_um": px_per_um, "xs": xs, "ys": ys, "hull": hull,
            "width_um": width_um, "height_um": height_um,
            "box_um2": width_um * height_um, "ellipse_um2": np.pi * width_um * height_um / 4,
            "hull_um2": hull.volume / px_per_um**2}  # 2-D: volume = area


def main() -> None:
    """Print the areas + save a check image."""
    m = measure()
    print(f"{m['px_per_um']:.3f} px/um; extent {m['width_um']:.0f} x {m['height_um']:.0f} um; "
          f"box {m['box_um2'] / 1e3:.1f}, ellipse {m['ellipse_um2'] / 1e3:.1f}, "
          f"convex hull {m['hull_um2'] / 1e3:.1f} x 10^3 um^2")
    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot()
    ax.imshow(m["img"], cmap="gray")
    ax.scatter(m["xs"], m["ys"], s=0.2, color="#0072B2")
    loop = np.r_[m["hull"].vertices, m["hull"].vertices[:1]]
    ax.plot(m["xs"][loop], m["ys"][loop], color="#E69F00", lw=1.5)
    ax.set_title(f"convex hull {m['hull_um2'] / 1e3:.0f} × 10³ µm² (blue = pixels used)")
    ax.axis("off")
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
