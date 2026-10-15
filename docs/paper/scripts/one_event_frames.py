# ruff: noqa: INP001
"""Scratch: 2025_06_11-0006 compartment 11 (2 units, 1 event) -- blobs overlapping its footprint, frames 998-1010."""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

SPONT = Path("results/spontaneous")
STEM, CID = "2025_06_11-0006_BIEXP_ALS", 11

coords = np.load(SPONT / "footprints" / f"{STEM}_ZONES.npz")[f"zone{CID}_footprint"]
frames = list(range(998, 1011))
pages = tifffile.imread(SPONT / "mask" / f"{STEM}_HOTSPOT_MASK.tif", key=[f - 1 for f in frames]) > 0
print(f"footprint {len(coords)} px")
for f, page in zip(frames, pages, strict=True):
    lab, _ = ndimage.label(page)
    hit = lab[coords[:, 0], coords[:, 1]]
    blobs = []
    for b in np.unique(hit[hit > 0]):
        ys, xs = np.nonzero(lab == b)
        blobs.append(f"blob {b}: {ys.size:6d} px, {np.sum(hit == b):6d} in footprint, "
                     f"centroid (y {ys.mean():5.0f}, x {xs.mean():5.0f})")
    print(f"frame {f}: {len(blobs)} blob(s)" + "".join(f"\n    {b}" for b in blobs))
