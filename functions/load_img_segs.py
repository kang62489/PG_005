## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile


def load_img_segs(
    proc_tiff_path: Path,
    lst_img_frame_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Read raw (detrended, unnormalized) per-spike segments from the proc TIFF."""
    lst_segments: list[np.ndarray] = []

    with tifffile.TiffFile(proc_tiff_path) as tif:
        for left, right in lst_img_frame_ranges:
            # Source TIFFs are float16 — numpy has no native float16 arithmetic and
            # emulates it in software, making downstream math ~6x slower than float32.
            segment = tif.asarray(key=slice(left, right + 1)).astype(np.float32)
            lst_segments.append(segment)

    return lst_segments
