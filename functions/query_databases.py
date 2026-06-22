## Modules
# Standard library imports
import sqlite3
from pathlib import Path

# Third-party imports
import polars as pl
from rich.console import Console

# Local application imports
from utils import ColumnSorter

console = Console()


def _sort_rec_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop IGNORE_COLUMNS, then reorder the rest by ColumnSorter group priority.

    Priority: CORE_COLUMNS, IMG_COND_COLUMNS, EPHY_COND_COLUMNS, OTHER_COLUMNS,
    FLUIDIC_COLUMNS, MEMO_COLUMNS, each kept in its own defined order. Any column
    not covered by a group (e.g. a new field added to rec_data.db later) is
    appended alphabetically at the end.
    """
    kept_columns = [c for c in df.columns if c not in ColumnSorter.IGNORE_COLUMNS]

    priority_groups = (
        ColumnSorter.CORE_COLUMNS,
        ColumnSorter.IMG_COND_COLUMNS,
        ColumnSorter.EPHY_COND_COLUMNS,
        ColumnSorter.OTHER_COLUMNS,
        ColumnSorter.FLUIDIC_COLUMNS,
        ColumnSorter.MEMO_COLUMNS,
    )
    ordered: list[str] = []
    for group in priority_groups:
        ordered.extend(column for column in group if column in kept_columns)

    # Columns present in the data but not covered by any priority group.
    leftover = sorted(column for column in kept_columns if column not in ordered)

    return df.select(ordered + leftover)


def populate_animal_id_values(df: pl.DataFrame, exp_db_path: Path) -> pl.DataFrame:
    """Fill missing ANIMAL_ID values using the DOR -> Animal_ID mapping in exp_info.db.

    Derives each row's DOR from the date prefix of Filename (e.g. '2024_02_15-0000.tif'
    -> '2024_02_15'), looks up every candidate Animal_ID recorded for that DOR in
    BASIC_INFO, then fills missing (null or empty-string) ANIMAL_ID values per DOR:
      - exactly one candidate -> fill every missing row with it.
      - exactly two candidates, one of them already used among that DOR's filled
        rows -> fill the missing rows with the other (still-unused) candidate.
    Any DOR that doesn't reduce to one of those two cases (0 or >=3 candidates, or
    two candidates with neither/both already used) is left as-is.
    """
    if "ANIMAL_ID" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("ANIMAL_ID"))

    df = df.with_columns(pl.col("Filename").str.split("-").list.first().alias("_dor"))
    dors = df["_dor"].unique().to_list()

    conn = sqlite3.connect(exp_db_path)
    try:
        placeholders = ", ".join("?" * len(dors))
        rows = conn.execute(
            f"SELECT DOR, Animal_ID FROM BASIC_INFO WHERE DOR IN ({placeholders})", dors
        ).fetchall()
    finally:
        conn.close()

    candidates_by_dor: dict[str, list[str]] = {}
    for dor, animal_id in rows:
        candidates_by_dor.setdefault(dor, []).append(animal_id)

    fill_value_by_dor: dict[str, str] = {}
    for dor, candidates in candidates_by_dor.items():
        animal_ids = df.filter(pl.col("_dor") == dor)["ANIMAL_ID"]
        missing_mask = animal_ids.is_null() | (animal_ids == "")
        if not missing_mask.any():
            continue

        if len(candidates) == 1:
            fill_value_by_dor[dor] = candidates[0]
        elif len(candidates) == 2:
            used = set(animal_ids.filter(~missing_mask).to_list())
            unused = [c for c in candidates if c not in used]
            if len(unused) == 1 and len(used) <= 1:
                fill_value_by_dor[dor] = unused[0]

    is_missing = pl.col("ANIMAL_ID").is_null() | (pl.col("ANIMAL_ID") == "")
    df = df.with_columns(
        pl.when(is_missing & pl.col("_dor").is_in(fill_value_by_dor.keys()))
        .then(pl.col("_dor").replace(fill_value_by_dor))
        .otherwise(pl.col("ANIMAL_ID"))
        .alias("ANIMAL_ID")
    )
    return df.drop("_dor")


def lookup_rec_from_db(table: pl.DataFrame, db_path: Path, exp_db_path: Path) -> pl.DataFrame:
    """Batch-query rec_data.db and compile the results into a single DataFrame.

    Groups `table`'s raw_tiff_name values by date prefix (e.g. '2025_12_15-0042.tif'
    -> table 'REC_2025_12_15') and issues one query per date table for all of that
    date's filenames, instead of one query per row. Each REC_* table's column set
    is read via PRAGMA table_info first, since different dates can have different
    (or missing) columns; tables that don't exist are skipped. Per-date results
    are diagonally concatenated, so columns absent from a given date become null
    rather than raising.

    The compiled result is then column-sorted (`_sort_rec_columns`) and has any
    missing ANIMAL_ID values backfilled from exp_info.db (`populate_animal_id_values`).

    Returns the compiled DataFrame (one row per matched Filename) -- not yet
    joined back onto `table`.
    """
    table = table.with_columns(pl.col("raw_tiff_name").str.split("-").list.first().alias("_date"))

    frames: list[pl.DataFrame] = []
    conn = sqlite3.connect(db_path)
    try:
        # group_by on a single column yields a 1-element key tuple per group.
        for (date,), rows_for_date in table.group_by("_date"):
            rec_table = f"REC_{date}"
            db_columns = [row[1] for row in conn.execute(f"PRAGMA table_info({rec_table})").fetchall()]
            if not db_columns:
                console.log(f"[yellow]Skipped {rec_table}: table not found in rec_data.db[/yellow]")
                continue

            raw_tiff_names = rows_for_date["raw_tiff_name"].to_list()
            placeholders = ", ".join("?" * len(raw_tiff_names))
            select_cols = ", ".join(db_columns)
            query = f"SELECT {select_cols} FROM {rec_table} WHERE Filename IN ({placeholders})"
            frames.append(
                pl.read_database(query=query, connection=conn, execute_options={"parameters": raw_tiff_names})
            )
    finally:
        conn.close()

    if not frames:
        return pl.DataFrame()

    result = _sort_rec_columns(pl.concat(frames, how="diagonal_relaxed"))
    return populate_animal_id_values(result, exp_db_path)


def _bright_excluded_expr(col: str = "bright_area_um2") -> pl.Expr:
    """True where `col` shows no real bright-region detection (null or zero)."""
    return pl.col(col).is_null() | (pl.col(col) == 0)


def _read_experiments(results_db_path: Path) -> pl.DataFrame:
    """Read the full experiments table from results.db into a polars DataFrame."""
    conn = sqlite3.connect(results_db_path)
    try:
        return pl.read_database(query="SELECT * FROM experiments", connection=conn)
    finally:
        conn.close()


def get_excluded_recordings(results_db_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recordings/cells with no detected bright region -- the ones compute_region_stats() excludes.

    Returns:
        (excluded_images, excluded_cells):
            excluded_images: one row per individual recording where bright_area_um2 is
                null/0 (columns: exp_date, abf_serial, img_serial, ANIMAL_ID, SLICE, AT).
            excluded_cells: cells (ANIMAL_ID, SLICE, AT) where every one of that cell's
                recordings failed to detect a bright region.
    """
    id_cols = ["exp_date", "abf_serial", "img_serial", "ANIMAL_ID", "SLICE", "AT"]
    df = _read_experiments(results_db_path)

    if df.is_empty():
        empty = pl.DataFrame(schema=dict.fromkeys(id_cols, pl.Utf8))
        return empty, empty.select("ANIMAL_ID", "SLICE", "AT")

    excluded_images = df.filter(_bright_excluded_expr()).select(id_cols)

    per_cell = df.group_by(["ANIMAL_ID", "SLICE", "AT"]).agg(pl.col("bright_area_um2").mean())
    excluded_cells = per_cell.filter(_bright_excluded_expr()).select("ANIMAL_ID", "SLICE", "AT")

    return excluded_images, excluded_cells


def get_cell_recording_status(results_db_path: Path) -> pl.DataFrame:
    """Per-recording bright-region detection status, for building per-cell file lists.

    Returns one row per recording (sorted by cell, then filename), columns:
    ANIMAL_ID, SLICE, AT, med_filename, detected (True if a bright region was
    found in that recording's spike frame).
    """
    df = _read_experiments(results_db_path)

    if df.is_empty():
        return pl.DataFrame(
            schema={"ANIMAL_ID": pl.Utf8, "SLICE": pl.Utf8, "AT": pl.Utf8, "med_filename": pl.Utf8, "detected": pl.Boolean}
        )

    return df.select(
        "ANIMAL_ID", "SLICE", "AT", "med_filename", (~_bright_excluded_expr()).alias("detected")
    ).sort(["ANIMAL_ID", "SLICE", "AT", "med_filename"])


def compute_region_stats(results_db_path: Path) -> pl.DataFrame:
    """Mean +/- std of bright/dim area and peak latency, averaged per unique cell.

    Multiple recordings of the same cell (ANIMAL_ID, SLICE, AT) are first
    collapsed to one row per cell (per-cell mean across its own recordings).
    Cells with no detected bright region (bright_area_um2 null/0 across every
    one of that cell's recordings -- i.e. spike-frame categorization never
    found a bright blob) are excluded before computing mean/std, since there's
    no real measurement to average for a cell that was never actually
    recorded. n_detected/n_total let you see how many cells were usable out of
    how many were attempted.

    Returns one row per metric (bright_area_um2, dim_area_um2, total_area_um2,
    peak_latency_ms) with columns: metric, mean, std, n_detected, n_total.
    total_area_um2 is bright_area_um2 + dim_area_um2, so it's null wherever dim
    wasn't detected -- a cell with no dim measurement has no real total to report.
    """
    metric_cols = ["bright_area_um2", "dim_area_um2", "total_area_um2", "peak_latency_ms"]
    agg_cols = ["bright_area_um2", "dim_area_um2", "peak_latency_ms"]
    df = _read_experiments(results_db_path)

    if df.is_empty():
        return pl.DataFrame(
            schema={"metric": pl.Utf8, "mean": pl.Float64, "std": pl.Float64, "n_detected": pl.Int64, "n_total": pl.Int64}
        )

    per_cell = df.group_by(["ANIMAL_ID", "SLICE", "AT"]).agg([pl.col(c).mean().alias(c) for c in agg_cols])
    per_cell = per_cell.with_columns((pl.col("bright_area_um2") + pl.col("dim_area_um2")).alias("total_area_um2"))
    n_total = per_cell.height

    detected = per_cell.filter(~_bright_excluded_expr())
    n_detected = detected.height

    return pl.DataFrame({
        "metric": metric_cols,
        "mean": [detected[c].mean() for c in metric_cols],
        "std": [detected[c].std() for c in metric_cols],
        "n_detected": [n_detected] * len(metric_cols),
        "n_total": [n_total] * len(metric_cols),
    })


def count_unique_cells(ref_df: pl.DataFrame) -> pl.DataFrame:
    """Reduce ref_df rows to one row per unique cell, listing each cell's filenames.

    A cell is identified by its (ANIMAL_ID, SLICE, AT) triple -- e.g.
    ('neoChAT-677', '2R', 'CELL_1'). Multiple raw_tiff_name entries in an ana
    list commonly share the same cell (repeated recordings of the same slice).

    Returns one row per unique cell with columns: ANIMAL_ID, SLICE, AT,
    Filenames (sorted list of Filename values for that cell), n_images.
    """
    return (
        ref_df.group_by(["ANIMAL_ID", "SLICE", "AT"])
        .agg(pl.col("Filename").sort().alias("Filenames"), pl.len().alias("n_images"))
        .sort(["ANIMAL_ID", "SLICE", "AT"])
    )
