## Modules
# Standard library imports
import sqlite3
from pathlib import Path

# Third-party imports
import polars as pl
from rich.console import Console

# Local application imports
from utils.params import ColumnSorter

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
                console.print(f"[yellow]Skipped {rec_table}: table not found in rec_data.db[/yellow]")
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
