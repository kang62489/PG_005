"""
flash_size_analysis.py  --  Flash size per recording, detection only (no zones, no maps); for 40X / 60X.

  Step 1. Select : proc list -> recordings with an ALS tiff, OBJ in --objs (and SENSOR = --sensor, if given)
  Step 2. Detect : SpontaneousZoneAnalyzer.detect() (background threshold + mask cleanup, blobs < TH_SMALL_OBJ out)
  Step 3. Export : {results_dir}/flash_size/{stem}.csv, one row per flash (functions/flash_table.py):
                   recording, obj, sensor, frame, area_px, area_um2, bounding box, touches_edge.
                   Recordings with a CSV already there are skipped (resumable).

Usage:
    python flash_size_analysis.py --proc_list data/proc_20260922_000.txt
        [--results_dir results] [--objs 40X 60X] [--sensor GACh3.0] [--sigma 2.0]
"""

## Modules
# Standard library imports
import argparse
import time
from pathlib import Path

# Third-party imports
import tifffile
from rich.console import Console

# Local imports
from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import CROSSOVER_RATIO
from functions import check_cuda
from functions.flash_table import flash_table
from spontaneous_analysis import select_recordings

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

DEFAULT_OBJS = ["40X", "60X"]  # 10X flashes come from spontaneous_analysis.py


# ===========================================================================
#
#   RUN
#
# ===========================================================================

def run(proc_list_path: Path, results_dir: Path = Path("results"), objs: list[str] | None = None,
        sensor: str | None = None, sigma: float = CROSSOVER_RATIO, cuda_available: bool = False,
        db_path: Path = Path("data/rec_data.db"), exp_db_path: Path = Path("data/exp_info.db")) -> None:
    """Steps 1-3 for every selected recording in a proc list."""
    run_t0 = time.time()
    out_dir = results_dir / "flash_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    objs = objs or DEFAULT_OBJS

    # Step 1. Select
    recordings = select_recordings(proc_list_path, db_path, exp_db_path, all_obj=True)
    keep = recordings["OBJ"].is_in(objs)
    if sensor is not None:
        keep &= recordings["SENSOR"] == sensor
    recordings = recordings.filter(keep)
    console.log(f"{len(recordings)} recording(s) selected ({', '.join(objs)}{f', {sensor}' if sensor else ''})")

    for i, row in enumerate(recordings.iter_rows(named=True), 1):
        stem = Path(row["proc_tiff_path"]).stem
        out_path = out_dir / f"{stem}.csv"
        if out_path.exists():
            console.log(f"[dim][{i}/{len(recordings)}] {stem}: already done, skipped[/dim]")
            continue
        t0 = time.time()

        # Step 2. Detect
        analyzer = SpontaneousZoneAnalyzer(tifffile.imread(row["proc_tiff_path"]), obj=row["OBJ"], sigma_ratio=sigma,
                                           cuda_available=cuda_available)
        analyzer.detect()

        # Step 3. Export
        _, table = flash_table(analyzer.mask, analyzer.um_per_px)
        table.insert(0, "sensor", row["SENSOR"])
        table.insert(0, "obj", row["OBJ"])
        table.insert(0, "recording", stem)
        table.to_csv(out_path, index=False)
        console.log(f"[{i}/{len(recordings)}] {stem} ({row['OBJ']}, {row['SENSOR']}): {len(table)} flashes, "
                    f"{int(table['touches_edge'].sum())} touching the edge, {time.time() - t0:.1f}s "
                    f"-> {out_path.resolve()}")
    console.rule(f"[dim]Total time: {time.time() - run_t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash size per recording (detection only)")
    parser.add_argument("--proc_list", required=True, type=Path, help="Proc list (proc_*.txt) naming the recordings")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Outputs go to <results_dir>/flash_size/")
    parser.add_argument("--objs", nargs="+", default=DEFAULT_OBJS, help="Objectives to analyze")
    parser.add_argument("--sensor", default=None, help="Only this sensor (default: all)")
    parser.add_argument("--sigma", type=float, default=CROSSOVER_RATIO,
                        help="Threshold = background peak + this many sigmas")
    parser.add_argument("--db", type=Path, default=Path("data/rec_data.db"), help="Path to rec_data.db")
    parser.add_argument("--exp_db", type=Path, default=Path("data/exp_info.db"), help="Path to exp_info.db")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda()  # must run before anything imports numba
    console.log(_cuda_msg)
    run(args.proc_list, args.results_dir, args.objs, args.sensor, args.sigma, _cuda_available, args.db, args.exp_db)
