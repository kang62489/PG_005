# ruff: noqa: INP001, E402
"""
obj_coverage.py  --  plain_02 Figs. 1-2, part 1: flash masks + per-flash table at 60X and 40X (GACh3.0).

Steps
-----
1. Select: proc_20260922_000.txt -> recordings with an ALS tiff, OBJ 40X / 60X, SENSOR GACh3.0.
2. Detect only (SpontaneousZoneAnalyzer.detect: background threshold + mask cleanup incl. blobs < TH_SMALL_OBJ px
   removed, no grouping; giant flashes still in).
3. Flash = one 4-connected blob of one frame; coverage = blob area / frame area
   -> output/plain_02/per_recording/{stem}.csv (resumable), output/plain_02/edge/{stem}.csv (bounding boxes)
   + output/plain_02/flash_coverage.csv.

Part 2 (flash-area histograms, Figs. 1-2, recordings table): flash_area_obj.py.

Usage:
    .venv/Scripts/python.exe docs/plain_02/scripts/obj_coverage.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from functions import check_cuda

_CUDA, _MSG = check_cuda()  # must run before anything imports numba

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import CROSSOVER_RATIO, TH_SMALL_OBJ
from spontaneous_analysis import select_recordings

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROC_LIST = ROOT / "data" / "proc_20260922_000.txt"
DATA_DIR = ROOT / "output" / "plain_02"  # regenerable caches
OBJS = ["60X", "40X"]
SENSOR = "GACh3.0"
CHUNK = 100  # frames labelled per call


def clean_and_table(mask: np.ndarray, min_px: int = TH_SMALL_OBJ) -> tuple[np.ndarray, list[dict]]:
    """Drop blobs < min_px and list the rest: (clean mask, rows of frame / area_px / bounding box).

    Blobs = 4-connected within a frame (as the pipeline's mask cleanup), never across frames. Labelled
    CHUNK frames at a time (one call per chunk instead of one per frame; int32 labels of all frames would be ~5 GB).
    """
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = ndimage.generate_binary_structure(2, 1)  # in-plane 4-connectivity only
    clean = np.zeros_like(mask)
    rows = []
    for start in range(0, len(mask), CHUNK):
        labels, _ = ndimage.label(mask[start:start + CHUNK], structure=structure)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_px
        keep[0] = False
        clean[start:start + CHUNK] = keep[labels]
        for k, box in enumerate(ndimage.find_objects(labels), 1):
            if box is not None and keep[k]:
                rows.append({"frame": start + box[0].start + 1, "area_px": int(sizes[k]), "r0": box[1].start,
                             "c0": box[2].start, "r1": box[1].stop, "c1": box[2].stop})
    return clean, rows


def main() -> None:
    """Steps 1-3; recordings with a per-recording CSV already in DATA_DIR/per_recording are skipped (resumable)."""
    print(_MSG)
    recordings = select_recordings(PROC_LIST, ROOT / "data" / "rec_data.db", ROOT / "data" / "exp_info.db", True)
    recordings = recordings.filter(recordings["OBJ"].is_in(OBJS) & (recordings["SENSOR"] == SENSOR))
    print(f"{len(recordings)} recordings: {recordings['OBJ'].value_counts().sort('OBJ').rows()}")
    (DATA_DIR / "per_recording").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "edge").mkdir(parents=True, exist_ok=True)

    columns = ["recording", "obj", "frame", "area_px", "coverage_pct"]
    for i, row in enumerate(recordings.iter_rows(named=True), 1):
        stem = Path(row["proc_tiff_path"]).stem
        rec_csv = DATA_DIR / "per_recording" / f"{stem}.csv"
        if rec_csv.exists():
            continue
        analyzer = SpontaneousZoneAnalyzer(tifffile.imread(row["proc_tiff_path"]), obj=row["OBJ"],
                                           sigma_ratio=CROSSOVER_RATIO, cuda_available=_CUDA)
        analyzer.detect()
        mask, blobs = clean_and_table(analyzer.mask)  # pipeline already dropped < TH_SMALL_OBJ; this lists blobs
        edge = pd.DataFrame(blobs, columns=["frame", "area_px", "r0", "c0", "r1", "c1"])
        edge.to_csv(DATA_DIR / "edge" / f"{stem}.csv", index=False)  # read by edge_touch / flash_area_obj
        table = edge[["frame", "area_px"]].assign(recording=stem, obj=row["OBJ"])
        table["coverage_pct"] = 100 * table["area_px"] / (mask.shape[1] * mask.shape[2])
        table[columns].to_csv(rec_csv, index=False)
        print(f"[{i}/{len(recordings)}] {stem} ({row['OBJ']}): {len(table)} flashes")

    table = pd.concat([pd.read_csv(DATA_DIR / "per_recording" / f"{Path(p).stem}.csv")
                       for p in recordings["proc_tiff_path"]], ignore_index=True)
    csv_path = DATA_DIR / "flash_coverage.csv"
    table.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
