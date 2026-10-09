"""
zone_groups.py  --  Group recur_zones that share a centre, for the contour montages (spontaneous_analysis.py).

  Step 1. Circle   : per recur_zone, centre of mass + radius ("inner" = nearest edge, "far" = farthest pixel)
  Step 2. Groups   : smallest remaining zone = seed; zones whose centre lies inside its circle join its group
  Step 3. Overlaps : shared area of every pair of group-largest zones (measured only, zones unchanged)

Example:
    groups, circles = group_zones(analyzer.zone_masks, "far")
    overlaps = group_overlaps(analyzer.zone_masks, groups, analyzer.um_per_px)
"""

## Modules
# Standard library imports
import itertools

# Third-party imports
import numpy as np
import pandas as pd
from scipy import ndimage

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

OVERLAP_COLUMNS = ["zone_i", "zone_j", "shared_um2", "pct_of_i", "pct_of_j"]

# ===========================================================================
#
#   STEP 1 -- CIRCLE
#
# ===========================================================================

def zone_circle(mask: np.ndarray, mode: str) -> tuple[float, float, float]:
    """(row, col, radius) around the zone's centre of mass.

    "inner": distance to the nearest pixel outside the zone (largest circle at the centre inside it; 0 if outside).
    "far": distance to the zone's farthest pixel (smallest circle at the centre enclosing the whole zone).
    """
    cy, cx = ndimage.center_of_mass(mask)
    if mode == "far":
        rows, cols = np.nonzero(mask)
        return cy, cx, float(np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2).max())
    inside = ndimage.distance_transform_edt(mask)
    return cy, cx, float(inside[int(round(cy)), int(round(cx))])


# ===========================================================================
#
#   STEP 2 -- GROUPS
#
# ===========================================================================

def group_zones(masks: dict, mode: str) -> tuple[list[tuple[int, list]], dict]:
    """([(seed zone id, group ids largest first)], {zone id: circle}); every zone ends up in exactly one group."""
    circles = {z: zone_circle(m, mode) for z, m in masks.items()}
    area = {z: int(m.sum()) for z, m in masks.items()}
    remaining = sorted(masks, key=area.get)
    groups = []
    while remaining:
        seed = remaining[0]
        sy, sx, r = circles[seed]
        members = [z for z in remaining
                   if z == seed or (circles[z][0] - sy) ** 2 + (circles[z][1] - sx) ** 2 <= r**2]
        groups.append((seed, sorted(members, key=area.get, reverse=True)))
        remaining = [z for z in remaining if z not in members]
    return groups, circles


# ===========================================================================
#
#   STEP 3 -- OVERLAPS
#
# ===========================================================================

def group_overlaps(masks: dict, groups: list[tuple[int, list]], um_per_px: float) -> pd.DataFrame:
    """One row per overlapping pair of group-largest zones: shared area (um^2), % of each zone's area."""
    largest = sorted(ids[0] for _, ids in groups)
    rows = []
    for i, j in itertools.combinations(largest, 2):
        shared = int((masks[i] & masks[j]).sum())
        if shared:
            rows.append({"zone_i": i, "zone_j": j, "shared_um2": shared * um_per_px**2,
                         "pct_of_i": 100 * shared / masks[i].sum(), "pct_of_j": 100 * shared / masks[j].sum()})
    return pd.DataFrame(rows, columns=OVERLAP_COLUMNS)
