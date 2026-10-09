# ruff: noqa: INP001
"""Scratch: which GACh3.0 compartment has only 1 event, and why is it a compartment (not an NR zone)?"""

import ast
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 200)
SPONT = Path("results/spontaneous")

comp = pd.read_excel(SPONT / "spontaneous_summary.xlsx", sheet_name="compartments")
one = comp[(comp["sensor"] == "GACh3.0") & (comp["n_events"] < 2)]
print("=== compartments with < 2 events\n", one.to_string(index=False))

for r in one.itertuples():
    book = pd.read_excel(SPONT / f"{r.recording}_ZONES.xlsx", sheet_name=None)
    row = book["compartments"].query("compartment_id == @r.compartment_id")
    print(f"\n=== {r.recording} compartment {r.compartment_id}: 'compartments' row\n", row.to_string(index=False))
    frames = ast.literal_eval(row["active_frames"].iloc[0])
    print("active_frames (as stored, with repeats):", frames)
    tracks = ast.literal_eval(row["track_ids"].iloc[0])
    fit = book["step4_fit"]
    print("units in step4_fit:\n", fit[fit["unit"].isin(tracks)].to_string(index=False))
    print("step3_merge rows touching its source:\n",
          book["step3_merge"][book["step3_merge"].isin([r.source]).any(axis=1)].to_string(index=False))
