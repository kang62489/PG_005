"""
Recordings of a proc / ana list file (every pipeline reads its list through this class).

  Step 1. Read   : filelist_parser -> table (every 'Picked:' column) + dir_* paths
  Step 2. Dirs   : dir_* paths, checked (missing -> ValueError)
  Step 3. Select : rows in rec_data.db, with their processed tiff on disk, OBJ / SENSOR filters
                   (reads the db only when called)
  Step 4. Detection parameters per selected row: th_small_obj (removal_suggestion.csv, else REMOVAL_FALLBACK)
                   -> connect_radius

Example:
    >>> recs = RecordingList(Path("data/proc_20260922_000.txt"), "proc")
    >>> raw_dir, proc_dir = recs.dirs()
    >>> recordings = recs.select(objs=["10X"], sensor="GACh3.0")   # raw_tiff_name, proc_tiff_path, OBJ, SENSOR
    >>> recordings = recs.set_connect_radius(recs.set_removal_suggestion(recordings, Path("results")))
"""

## Modules
# Standard library imports
import math
from pathlib import Path

# Third-party imports
import pandas as pd
import polars as pl
from rich.console import Console

# Local imports
from functions import filelist_parser, lookup_rec_from_db

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

REQUIRED_DIRS = {  # dir_* keys every list of this type must have
    "proc": ("dir_raw_tiffs", "dir_proc_tiffs"),
    "ana": ("dir_proc_tiffs", "dir_raw_abfs", "dir_results"),
}
ALS_SUFFIX = "_BIEXP_ALS.tif"  # default processed tiff of select()

# --- detection parameters ---
REMOVAL_SUGGESTION_CSV = "removal_suggestion.csv"  # permutation_test.py output: recording, th_small_obj_px
REMOVAL_FALLBACK = {"10X": 1900, "40X": 1700, "60X": 1500}  # px: OBJ median th_small_obj_px (results/permutation/)
CONNECT_ROUND_PX = 5  # px: connect radius sqrt(th_small_obj / pi) rounded to the nearest multiple of this


class RecordingList:
    """Table + dir_* paths of one list file; select() keeps the recordings to analyze.

    Results:
        table, io_dirs                                   (__init__)
        raw_tiff_name, proc_tiff_path, OBJ, SENSOR rows  (select)
    """

    def __init__(self, list_path: Path, list_type: str = "proc", db_path: Path = Path("data/rec_data.db"),
                 exp_db_path: Path = Path("data/exp_info.db")) -> None:
        self.list_path = list_path
        self.list_type = list_type
        self.db_path = db_path
        self.exp_db_path = exp_db_path
        self.table, self.io_dirs = filelist_parser(list_path)

    # -----------------------------------------------------------------------
    # dirs: checked dir_* paths
    # -----------------------------------------------------------------------

    def dirs(self, *keys: str) -> tuple[Path, ...]:
        """Paths of the given dir_* keys (default: REQUIRED_DIRS of the list type); missing -> ValueError."""
        keys = keys or REQUIRED_DIRS[self.list_type]
        missing = [key for key in keys if key not in self.io_dirs]
        if missing:
            msg = f"Missing {', '.join(missing)} in {self.list_path}"
            raise ValueError(msg)
        return tuple(Path(self.io_dirs[key]) for key in keys)

    # -----------------------------------------------------------------------
    # select: recordings in rec_data.db with their processed tiff, OBJ / SENSOR filters
    # -----------------------------------------------------------------------

    def select(self, objs: list[str] | None = None, sensor: str | None = None,
               suffix: str = ALS_SUFFIX) -> pl.DataFrame:
        """Rows (raw_tiff_name, proc_tiff_path, OBJ, SENSOR) that are in rec_data.db, have {stem}{suffix} in
        dir_proc_tiffs, and pass the filters (objs / sensor None = all)."""
        (proc_dir,) = self.dirs("dir_proc_tiffs")
        rec_info = lookup_rec_from_db(self.table.select("raw_tiff_name"), self.db_path, self.exp_db_path)
        selected: list[dict] = []
        for name in self.table["raw_tiff_name"].to_list():
            proc_tiff_path = proc_dir / f"{Path(name).stem}{suffix}"
            match = rec_info.filter(pl.col("Filename") == name) if not rec_info.is_empty() else rec_info
            if match.is_empty():
                console.log(f"[yellow]Skipped {name}: not in rec_data.db[/yellow]")
                continue
            obj, rec_sensor = match["OBJ"].item(), match["SENSOR"].item()
            if not proc_tiff_path.exists():
                console.log(f"[yellow]Skipped {name}: {proc_tiff_path.name} not found[/yellow]")
                continue
            if objs is not None and obj not in objs:
                console.log(f"[yellow]Skipped {name}: OBJ={obj} (only {', '.join(objs)})[/yellow]")
                continue
            if sensor is not None and rec_sensor != sensor:
                continue
            selected.append({"raw_tiff_name": name, "proc_tiff_path": str(proc_tiff_path), "OBJ": obj,
                             "SENSOR": rec_sensor})
        return pl.DataFrame(selected)

    # -----------------------------------------------------------------------
    # detection parameters: th_small_obj, connect_radius per selected row
    # -----------------------------------------------------------------------

    @staticmethod
    def set_removal_suggestion(recordings: pl.DataFrame, results_dir: Path) -> pl.DataFrame:
        """Add th_small_obj (px) + th_small_obj_flag to every recording, once before the loop.

        th_small_obj = own th_small_obj_px from <results_dir>/permutation/{OBJ}/removal_suggestion.csv (flag empty),
        else REMOVAL_FALLBACK[OBJ] (flag says why).
        """
        if recordings.is_empty():
            return recordings
        own: dict[tuple[str, str], int] = {}  # (OBJ, raw stem) -> th_small_obj_px
        for obj in recordings["OBJ"].unique().sort().to_list():
            path = results_dir / "permutation" / obj / REMOVAL_SUGGESTION_CSV
            if not path.exists():
                continue
            table = pd.read_csv(path)
            console.log(f"th_small_obj: {len(table)} {obj} recording(s) in {path.resolve()}")
            own.update({(obj, stem): int(px) for stem, px in zip(table["recording"], table["th_small_obj_px"],
                                                                 strict=True)})

        th_small_obj, flags = [], []
        for name, obj in zip(recordings["raw_tiff_name"], recordings["OBJ"], strict=True):
            key = (obj, Path(name).stem)
            if key in own:
                th_small_obj.append(own[key])
                flags.append("")
            else:
                th_small_obj.append(REMOVAL_FALLBACK[obj])
                flags.append(f"{REMOVAL_FALLBACK[obj]} px, {obj} fallback, not in {REMOVAL_SUGGESTION_CSV}")
        return recordings.with_columns(pl.Series("th_small_obj", th_small_obj), pl.Series("th_small_obj_flag", flags))

    @staticmethod
    def set_connect_radius(recordings: pl.DataFrame) -> pl.DataFrame:
        """Add connect_radius (px): same-frame blobs this close are one flash.

        = radius of a circle of th_small_obj px (the largest chance-size piece from the permutation test), rounded to
        the nearest CONNECT_ROUND_PX, e.g. th_small_obj 2,100 px -> 25.9 -> 25 px.
        """
        if recordings.is_empty():
            return recordings
        radius = [CONNECT_ROUND_PX * round(math.sqrt(th / math.pi) / CONNECT_ROUND_PX)
                  for th in recordings["th_small_obj"]]
        return recordings.with_columns(pl.Series("connect_radius", radius))
