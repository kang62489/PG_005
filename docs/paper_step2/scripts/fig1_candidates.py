# ruff: noqa: INP001
"""Scratch: candidate recordings for Fig. 1A (GACh3.0) + page 1 of their ZONE_MAPS.tif as PNG.

  keep    : typical compartment count (IQR 7-12), >= 1 compartment with an event >= 80 % (leave-one-out),
            striatum outline present
  rank    : distance to the step-2 medians (compartments 9, rate 0.15 Hz) -- closest = most typical
  -> output/paper_step2/candidates/{rank}_{recording}_page1.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

pd.set_option("display.width", 220)
SPONT = Path("results/spontaneous")
OUT = Path("output/paper_step2/candidates")
OUT.mkdir(parents=True, exist_ok=True)
N_SHOW = 6

rec = pd.read_excel(SPONT / "spontaneous_summary.xlsx", sheet_name="recordings")
rec = rec[(rec["sensor"] == "GACh3.0") & (rec["n_compartments"] > 0)].copy()
ev = pd.read_excel(Path(__file__).parents[1] / "tables" / "event_coverage.xlsx", sheet_name="events")
big = ev[ev["coverage_loo"] >= 0.8].groupby("recording")["compartment_id"].nunique().rename("n_comp_ge80")
rec = rec.merge(big, left_on="recording", right_index=True, how="left").fillna({"n_comp_ge80": 0})

keep = rec[rec["n_compartments"].between(7, 12) & (rec["n_comp_ge80"] > 0) & rec["striatum_coverage"].notna()
           & (rec["striatum_area_um2"] > 0)].copy()
keep["score"] = (abs(keep["n_compartments"] - 9) / 9 + abs(np.log(keep["median_freq_hz"] / 0.15)))
keep = keep.sort_values("score")
cols = ["recording", "n_compartments", "n_non_recur_zone_type_1", "n_non_recur_zone_type_2", "median_freq_hz",
        "n_comp_ge80", "striatum_coverage", "score"]
print(f"{len(keep)} of {len(rec)} recordings pass the filters; top {N_SHOW}:")
print(keep[cols].head(N_SHOW).to_string(index=False))

for rank, stem in enumerate(keep["recording"].head(N_SHOW), 1):
    page = tifffile.imread(SPONT / f"{stem}_ZONE_MAPS.tif", key=0)
    path = OUT / f"{rank}_{stem}_page1.png"
    Image.fromarray(page).save(path)
    print(f"saved {path.resolve()}")
