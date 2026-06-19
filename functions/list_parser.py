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


def list_parser(path: Path) -> tuple[pl.DataFrame, dict[str, str]]:
    """Parse a pick/proc/ana list file into its bracket-row table and dir_* paths.

    The 'Picked: [col1, col2, ...]' line supplies column names; every following
    '[v1, v2, ...]' line supplies one row of values for those columns -- every
    field is read, so columns are looked up by name downstream instead of by
    position.

    Returns:
        (table, io_dirs) where table has one column per header field, and
        io_dirs maps each 'dir_xxx:' line (dir_raw_tiffs, dir_proc_tiffs,
        dir_raw_abfs, dir_results, ...) to its path value.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    picked_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Picked:"))
    columns = [p.strip() for p in lines[picked_idx].split(":", 1)[1].strip().strip("[]").split(",")]

    rows: list[list[str]] = []
    for line in lines[picked_idx + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("["):
            break
        rows.append([p.strip() for p in stripped.strip("[]").split(",")])

    table = _df_builder(columns, rows)

    io_dirs = {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in lines)
        if key.strip().startswith("dir_")
    }

    return table, io_dirs
