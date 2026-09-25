"""
Flow-pattern label per TV-L1 pair: linear flow-field fit inside the CAT mask (CPU, a tiny least-squares fit).

  Step 1. Blocks : u, v and keep_mask block-averaged (FLOW_PATTERN_BLOCK px, same blocks as the streamlines)
  Step 2. Fit    : flow(p) ~= A @ (p - centroid) + drift on the CAT blocks (6 numbers)
  Step 3. Label  : anisotropic if |drift| > spread (spread = |trace(A)| / 2 * RMS radius),
                   else source (trace(A) > 0, flow spreads out) or sink (trace(A) < 0, flow gathers in)

Drift angle uses the screen convention 0 = right, 90 = up.

Example:
    >>> pattern = fit_flow_pattern(pair["u"], pair["v"], pair["keep_mask"], um_per_px=1.333)
    >>> pattern["label"], pattern["drift_angle_deg"]      # 'anisotropic', 121.3
"""

## Modules
# Third-party imports
import numpy as np

FLOW_PATTERN_BLOCK = 8        # px; flow block-averaged before the fit and the streamlines
MIN_PATTERN_BLOCKS = 3        # fewer CAT blocks than this -> no fit (label None)


def block_mean(img: np.ndarray, block: int = FLOW_PATTERN_BLOCK) -> np.ndarray:
    """(H, W) -> (H // block, W // block) mean of each block."""
    h, w = img.shape[0] // block * block, img.shape[1] // block * block
    return img[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def fit_flow_pattern(u: np.ndarray, v: np.ndarray, keep_mask: np.ndarray, um_per_px: float) -> dict:
    """{label, drift_um, spread_um, drift_angle_deg}; drift / spread in µm/frame (all None if too few CAT blocks)."""
    keep = block_mean(keep_mask.astype(float)) > 0
    if keep.sum() < MIN_PATTERN_BLOCKS:
        return {"label": None, "drift_um": None, "spread_um": None, "drift_angle_deg": None}

    ys, xs = np.nonzero(keep)
    px = xs * FLOW_PATTERN_BLOCK + FLOW_PATTERN_BLOCK / 2
    py = ys * FLOW_PATTERN_BLOCK + FLOW_PATTERN_BLOCK / 2
    uu, vv = block_mean(u)[keep], block_mean(v)[keep]
    cx, cy = px.mean(), py.mean()
    design = np.column_stack([px - cx, py - cy, np.ones_like(px)])
    (a11, _, bu), *_ = np.linalg.lstsq(design, uu, rcond=None)
    (_, a22, bv), *_ = np.linalg.lstsq(design, vv, rcond=None)

    r_rms = float(np.sqrt(np.mean((px - cx) ** 2 + (py - cy) ** 2)))
    trace = float(a11 + a22)
    spread = abs(trace) / 2 * r_rms          # px/frame at the typical radius
    drift = float(np.hypot(bu, bv))

    if drift > spread:
        label = "anisotropic"
    else:
        label = "source" if trace > 0 else "sink"

    return {
        "label": label,
        "drift_um": drift * um_per_px,
        "spread_um": spread * um_per_px,
        "drift_angle_deg": float(np.degrees(np.arctan2(-bv, bu)) % 360),  # -bv: image rows grow downward
    }
