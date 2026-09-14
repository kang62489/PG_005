"""Stack just the spike frame from each already-exported per-segment tiff in
output/als_segments_0012/ into one multi-page tiff, for quick side-by-side
comparison across all 20 trials.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

SEGMENTS_DIR = PROJECT_ROOT / "output" / "als_segments_0012"
OUT_PATH = PROJECT_ROOT / "output" / "als_segments_0012_spike_frame_stack.tif"


def main() -> None:
    seg_paths = sorted(SEGMENTS_DIR.glob("seg*.tif"))
    if not seg_paths:
        msg = f"No segment tiffs found in {SEGMENTS_DIR}"
        raise FileNotFoundError(msg)

    spike_frames = []
    for seg_path in seg_paths:
        segment = tifffile.imread(seg_path)
        spike_frame_idx = segment.shape[0] // 2
        spike_frames.append(segment[spike_frame_idx])

    stack = np.stack(spike_frames).astype(np.float32)
    tifffile.imwrite(OUT_PATH, stack)
    print(f"Stacked {len(spike_frames)} spike frame(s) from {SEGMENTS_DIR.name}/ -> {OUT_PATH}")


if __name__ == "__main__":
    main()
