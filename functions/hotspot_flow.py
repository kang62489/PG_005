"""
Spike-aligned TV-L1 optical flow on the median stack, CAT mask applied after the flow (CPU only).

  Step 1. Flow : skimage TV-L1 on the raw (unmasked) MED pair
  Step 2. Mask : keep = union of both frames' CAT-bright pixels (plot_flow_panels draws arrows only there)

Pairs: spike-1->spike, spike->spike+1, ..., spike+3->spike+4 (FLOW_OFFSETS).

Example:
    >>> pairs = compute_flow_pairs(med_stack, cat_stack, spike_frame_idx)
    >>> pairs[1]["label"], pairs[1]["u"].shape       # 'spike -> spike+1', (H, W)
"""

## Modules
# Third-party imports
import numpy as np
from skimage.registration import optical_flow_tvl1

# Local imports
from classes.spatial_categorization import CATEGORY_BRIGHT

FLOW_OFFSETS = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)]  # (from, to) frame offsets from the spike


def _offset_label(offset: int) -> str:
    return "spike" if offset == 0 else f"spike{offset:+d}"


def mask_frame(frame: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Floor everything outside `keep` to this frame's own min."""
    return np.where(keep, frame, float(frame.min()))


def premasked_flow(
    med: np.ndarray, cat: np.ndarray, idx_from: int, idx_to: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(v, u, keep): TV-L1 flow of the raw frame pair; keep = union of both frames' CAT-bright pixels."""
    keep = (cat[idx_from] == CATEGORY_BRIGHT) | (cat[idx_to] == CATEGORY_BRIGHT)
    v, u = optical_flow_tvl1(med[idx_from], med[idx_to])
    return v, u, keep


def compute_flow_pairs(med: np.ndarray, cat: np.ndarray, spike_frame_idx: int) -> list[dict]:
    """One {label, offset_from, offset_to, idx_from, idx_to, u, v, keep_mask} dict per in-range FLOW_OFFSETS pair."""
    med = np.asarray(med, dtype=np.float32)  # same dtype as the exported MED tif
    cat = np.asarray(cat)
    n_frames = med.shape[0]

    pairs = []
    for offset_from, offset_to in FLOW_OFFSETS:
        idx_from, idx_to = spike_frame_idx + offset_from, spike_frame_idx + offset_to
        if idx_from < 0 or idx_to >= n_frames:
            continue
        v, u, keep = premasked_flow(med, cat, idx_from, idx_to)
        pairs.append({
            "label": f"{_offset_label(offset_from)} -> {_offset_label(offset_to)}",
            "offset_from": offset_from,
            "offset_to": offset_to,
            "idx_from": idx_from,
            "idx_to": idx_to,
            "u": u,
            "v": v,
            "keep_mask": keep,
        })
    return pairs
