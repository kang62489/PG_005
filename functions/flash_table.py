"""
flash_table.py  --  One row per flash (= one 4-connected blob of one frame) from a flash mask.

  Step 1. Label  : blobs per frame, in-plane 4-connectivity (as the pipeline's mask cleanup), never across frames
  Step 2. Clean  : blobs < min_px dropped (0 = keep all; the pipeline's own masks are already cleaned)
  Step 3. Table  : frame (1-based), area (px, um^2), bounding box (r1 / c1 exclusive), touches_edge

Example:
    clean, table = flash_table(analyzer.mask, analyzer.um_per_px)
"""

## Modules
# Third-party imports
import numpy as np
import pandas as pd
from scipy import ndimage

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

LABEL_CHUNK = 100  # frames labelled per call (int32 labels of all frames at once would be ~5 GB)
EDGE_PX = 1  # px: a bounding box this close to the frame edge touches it (the mask's outermost 1-px ring is never on)
FLASH_COLUMNS = ["frame", "area_px", "area_um2", "r0", "c0", "r1", "c1", "touches_edge"]


# ===========================================================================
#
#   STEPS 1-3
#
# ===========================================================================

def flash_table(mask: np.ndarray, um_per_px: float, min_px: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """(mask without blobs < min_px, one row per remaining blob with FLASH_COLUMNS)."""
    _, h, w = mask.shape
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = ndimage.generate_binary_structure(2, 1)  # in-plane 4-connectivity only
    clean = np.zeros_like(mask)
    rows = []
    for start in range(0, len(mask), LABEL_CHUNK):
        labels, _ = ndimage.label(mask[start:start + LABEL_CHUNK], structure=structure)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_px
        keep[0] = False
        clean[start:start + LABEL_CHUNK] = keep[labels]
        for k, box in enumerate(ndimage.find_objects(labels), 1):
            if box is not None and keep[k]:
                rows.append({"frame": start + box[0].start + 1, "area_px": int(sizes[k]), "r0": box[1].start,
                             "c0": box[2].start, "r1": box[1].stop, "c1": box[2].stop})
    table = pd.DataFrame(rows, columns=["frame", "area_px", "r0", "c0", "r1", "c1"])
    table["area_um2"] = table["area_px"] * um_per_px**2
    table["touches_edge"] = ((table["r0"] <= EDGE_PX) | (table["c0"] <= EDGE_PX)
                             | (table["r1"] >= h - EDGE_PX) | (table["c1"] >= w - EDGE_PX))
    return clean, table[FLASH_COLUMNS]
