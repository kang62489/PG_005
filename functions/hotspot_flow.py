"""
Spike-aligned TV-L1 optical flow on the median stack, CAT mask applied after the flow (CPU, one thread per pair).

  Step 1. Flow  : skimage TV-L1 on one raw (unmasked) MED pair + keep = union of both frames' CAT-bright pixels
  Step 2. Pairs : every in-range FLOW_OFFSETS pair (spike-1->spike ... spike+3->spike+4), run in parallel threads

plot_flow_panels draws arrows only inside keep.

Example:
    >>> pairs = compute_flow_pairs(med_stack, cat_stack, spike_frame_idx)
    >>> pairs[1]["label"], pairs[1]["u"].shape       # 'spike -> spike+1', (H, W)
"""

## Modules
# Standard library imports
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import numpy as np
from skimage.registration import optical_flow_tvl1

# Local imports
from classes.spatial_categorization import CATEGORY_BRIGHT

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 2: pairs ---------------------------------------------------------
FLOW_OFFSETS = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)]  # (from, to) frame offsets from the spike
N_THREADS_FLOW = len(FLOW_OFFSETS)  # one thread per pair (~3.4x faster, identical output)


# ===========================================================================
#
#   STEP 1 -- FLOW
#
# ===========================================================================


def pair_flow(
    med: np.ndarray, cat: np.ndarray, idx_from: int, idx_to: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(v, u, keep): TV-L1 flow of the raw frame pair; keep = union of both frames' CAT-bright pixels."""
    keep = (cat[idx_from] == CATEGORY_BRIGHT) | (cat[idx_to] == CATEGORY_BRIGHT)
    v, u = optical_flow_tvl1(med[idx_from], med[idx_to])
    return v, u, keep


# ===========================================================================
#
#   STEP 2 -- PAIRS
#
# ===========================================================================


def _offset_label(offset: int) -> str:
    """Frame offset -> label: 0 -> 'spike', 2 -> 'spike+2', -1 -> 'spike-1'."""
    return "spike" if offset == 0 else f"spike{offset:+d}"


def compute_flow_pairs(med: np.ndarray, cat: np.ndarray, spike_frame_idx: int) -> list[dict]:
    """One {label, offset_from, offset_to, idx_from, idx_to, u, v, keep_mask} dict per in-range FLOW_OFFSETS pair."""
    med = np.asarray(med, dtype=np.float32)  # same dtype as the exported MED tif
    cat = np.asarray(cat)
    n_frames = med.shape[0]

    in_range = [
        (offset_from, offset_to, spike_frame_idx + offset_from, spike_frame_idx + offset_to)
        for offset_from, offset_to in FLOW_OFFSETS
        if spike_frame_idx + offset_from >= 0 and spike_frame_idx + offset_to < n_frames
    ]
    with ThreadPoolExecutor(N_THREADS_FLOW) as pool:
        flows = list(pool.map(lambda p: pair_flow(med, cat, p[2], p[3]), in_range))

    pairs = []
    for (offset_from, offset_to, idx_from, idx_to), (v, u, keep) in zip(in_range, flows, strict=True):
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
