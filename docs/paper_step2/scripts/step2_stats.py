# ruff: noqa: INP001
"""Scratch: step-2 numbers from results/spontaneous/spontaneous_summary.xlsx (GACh3.0 only).

  1. join recordings -> slice (rec_data.db SLICE, FRAMES) and animal (exp_info.db BASIC_INFO, DOR -> Animal_ID)
  2. counts: recordings / slices / animals, with >= 1 compartment
  3. compartments per animal / slice / recording (mean ± SD, median [IQR])
  4. days with no compartment
  5. frequency, option b: pooled compartments (GACh3.0, recordings with compartments, >= 2 events)
"""

import sqlite3

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

rec = pd.read_excel("results/spontaneous/spontaneous_summary.xlsx", sheet_name="recordings")
comp = pd.read_excel("results/spontaneous/spontaneous_summary.xlsx", sheet_name="compartments")
rec["date"] = rec["recording"].str[:10]
rec["filename"] = rec["recording"].str.replace("_BIEXP_ALS", "") + ".tif"
# paper rule: a compartment needs >= 2 separate events (recurring in time) -> drops 2025_06_11-0006 #11
print("excluded (< 2 events):", comp.loc[comp["n_events"] < 2, ["recording", "compartment_id"]].to_numpy().tolist())
comp = comp[comp["n_events"] >= 2]
rec["n_compartments"] = rec["recording"].map(comp.groupby("recording").size()).fillna(0).astype(int)

# --- 1. slice / frames / animal ---
con = sqlite3.connect("data/rec_data.db")
meta = []
for date in sorted(rec["date"].unique()):
    df = pd.read_sql(f"SELECT Filename, SLICE, FRAMES FROM 'REC_{date}'", con)
    meta.append(df)
con.close()
meta = pd.concat(meta).rename(columns={"Filename": "filename", "SLICE": "slice", "FRAMES": "frames_db"})
con = sqlite3.connect("data/exp_info.db")
animals = pd.read_sql("SELECT DOR, Animal_ID FROM BASIC_INFO", con).rename(columns={"DOR": "date", "Animal_ID": "animal"})
con.close()
dup = animals[animals.duplicated("date", keep=False)]
if len(dup):
    print("!! dates with more than one animal in BASIC_INFO:\n", dup)
rec = rec.merge(meta, on="filename", how="left").merge(animals.drop_duplicates("date"), on="date", how="left")
rec["slice_id"] = rec["date"] + "_" + rec["slice"].astype(str)

print("=== sensors (all 105):", rec["sensor"].value_counts().to_dict())
print("=== n_frames (analyzed):", rec["n_frames"].value_counts().to_dict(), "| FRAMES in rec_data.db:",
      rec["frames_db"].value_counts().to_dict())
g = rec[rec["sensor"] == "GACh3.0"].copy()
print("missing slice / animal:", g["slice"].isna().sum(), "/", g["animal"].isna().sum())


def describe(x: pd.Series, label: str) -> None:
    x = x.astype(float)
    q1, q3 = np.percentile(x, [25, 75])
    print(f"  {label:<42} n={len(x):3d}  mean {x.mean():6.2f} ± {x.std(ddof=1):5.2f} SD | "
          f"median {x.median():6.2f} [IQR {q1:.2f}–{q3:.2f}] | range {x.min():.2f}–{x.max():.2f}")


# --- 2. counts ---
has = g[g["n_compartments"] > 0]
print("\n=== 2. counts (GACh3.0)")
for label, df in (("all", g), ("with >= 1 compartment", has)):
    print(f"  {label:<24} recordings {len(df):3d} | slices {df['slice_id'].nunique():3d} | "
          f"animals {df['animal'].nunique():3d} | compartments {int(df['n_compartments'].sum())}")

# --- 3. compartments per animal / slice / recording ---
print("\n=== 3. compartments per unit (GACh3.0)")
for scope, df in (("all", g), ("with compartments only", has)):
    print(f" [{scope}]")
    describe(df["n_compartments"], "per recording")
    describe(df.groupby("slice_id")["n_compartments"].sum(), "per slice (sum of its recordings)")
    describe(df.groupby("animal")["n_compartments"].sum(), "per animal (sum of its recordings)")
    describe(df.groupby("animal")["slice_id"].nunique(), "slices per animal")
    describe(df.groupby("animal").size(), "recordings per animal")

per_animal = g.groupby(["date", "animal"]).agg(slices=("slice_id", "nunique"), recordings=("recording", "size"),
                                               rec_with_comp=("n_compartments", lambda s: int((s > 0).sum())),
                                               compartments=("n_compartments", "sum"),
                                               thr_median=("background_threshold", "median"))
print("\n=== per animal (GACh3.0)\n", per_animal.to_string())

# --- 4. no-compartment days ---
print("\n=== 4. recordings with 0 compartments, by day")
zero = g[g["n_compartments"] == 0]
print(zero.groupby("date").size().rename("n_zero").to_frame().join(g.groupby("date").size().rename("n_all")).to_string())
print(zero[["recording", "slice", "n_frames", "frames_db", "background_threshold", "n_non_recur_zone_type_1",
            "n_non_recur_zone_type_2", "n_dropped_units"]].to_string())

# --- 5. frequency, pooled (option b) ---
c = comp[comp["sensor"] == "GACh3.0"]
cf = c[c["n_events"] >= 2]
print(f"\n=== 5. frequency, pooled GACh3.0 compartments: {len(c)} compartments, {len(cf)} with >= 2 events")
describe(cf["mean_freq_hz"], "mean_freq_hz")
describe(cf["mean_period_s"], "mean_period_s")
describe(c["n_events"], "n_events (all compartments)")
