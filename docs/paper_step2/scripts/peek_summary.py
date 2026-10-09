# ruff: noqa: INP001
"""Scratch: overview of results/spontaneous/spontaneous_summary.xlsx (sheets, columns, a few rows)."""

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

path = Path("results/spontaneous/spontaneous_summary.xlsx")
for name, df in pd.read_excel(path, sheet_name=None).items():
    print(f"\n=== sheet '{name}': {df.shape[0]} rows x {df.shape[1]} cols")
    print(", ".join(df.columns))
    print(df.head(5).to_string())
