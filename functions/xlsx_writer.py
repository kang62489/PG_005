"""Write analysis summary tables to xlsx files."""

from pathlib import Path

import polars as pl
from openpyxl import Workbook


def write_cell_summary_xlsx(cell_df: pl.DataFrame, output_path: Path) -> None:
    """Write a count_unique_cells() result to output_path as a single-sheet xlsx.

    Filenames lists are joined into comma-separated strings, since xlsx cells
    can't hold a real list.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Cells"
    ws.append(["ANIMAL_ID", "SLICE", "AT", "Filenames", "n_images"])
    for row in cell_df.iter_rows(named=True):
        ws.append([row["ANIMAL_ID"], row["SLICE"], row["AT"], ", ".join(row["Filenames"]), row["n_images"]])
    wb.save(output_path)
