# ruff: noqa: INP001
"""Scratch: hotspots detected per recording (ZONES.xlsx 'counts' row 'detected'), summarised per day (GACh3.0)."""

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
rec = pd.read_excel("results/spontaneous/spontaneous_summary.xlsx", sheet_name="recordings")
rec = rec[rec["sensor"] == "GACh3.0"]
rows = []
for r in rec.itertuples():
    counts = pd.read_excel(Path("results/spontaneous") / f"{r.recording}_ZONES.xlsx", sheet_name="counts")
    detected = counts.loc[counts["stage"] == "detected", "n_hotspots"]
    rows.append({"date": r.recording[:10], "recording": r.recording, "detected": int(detected.iloc[0]),
                 "n_compartments": r.n_compartments})
df = pd.DataFrame(rows)
print(df.groupby("date").agg(recordings=("recording", "size"), detected_median=("detected", "median"),
                             detected_min=("detected", "min"), detected_max=("detected", "max"),
                             compartments=("n_compartments", "sum")).to_string())
