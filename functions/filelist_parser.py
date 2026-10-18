"""
filelist_parser.py  --  Read a pick / proc / ana list (*.txt) into a table + its dir_* paths.

  Step 1. Header : 'Picked: [col1, col2, ...]' line -> column names
  Step 2. Rows   : following '[v1, v2, ...]' lines -> one row each (field count checked)
  Step 3. Dirs   : every 'dir_xxx: path' line -> io_dirs

Every field is read, so columns are looked up by name downstream instead of by position.

Example:
    >>> table, io_dirs = filelist_parser(Path("data/proc_20260922_000.txt"))
    >>> io_dirs["dir_proc_tiffs"]
"""

## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import polars as pl


def _df_builder(columns: list[str], rows: list[list[str]]) -> pl.DataFrame:
    """Validate every row's field count against columns, then build the DataFrame."""
    for row in rows:
        if len(row) != len(columns):
            msg = f"Row {row} has {len(row)} fields, expected {len(columns)} ({columns})"
            raise ValueError(msg)
    return pl.DataFrame(rows, schema=columns, orient="row")


def filelist_parser(path: Path) -> tuple[pl.DataFrame, dict[str, str]]:
    """Parse a pick / proc / ana list file into its bracket-row table and dir_* paths.

    Returns:
        (table, io_dirs): one column per 'Picked:' field; io_dirs maps dir_raw_tiffs, dir_proc_tiffs, ... to paths.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # Step 1. Header
    picked_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Picked:"))
    columns = [field.strip() for field in lines[picked_idx].split(":", 1)[1].strip().strip("[]").split(",")]

    # Step 2. Rows
    rows: list[list[str]] = []
    for line in lines[picked_idx + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("["):
            break
        rows.append([field.strip() for field in stripped.strip("[]").split(",")])
    table = _df_builder(columns, rows)

    # Step 3. Dirs
    io_dirs = {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in lines)
        if key.strip().startswith("dir_")
    }

    return table, io_dirs
