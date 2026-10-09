# ruff: noqa: INP001
"""Scratch: largest single hotspot per GACh3.0 compartment (µm²), results/spontaneous (latest Saion run).

  per active frame : label the binary _HOTSPOT_MASK.tif page; the compartment's hotspot = the blob with the
                     largest overlap with its footprint; size = the whole blob (px -> µm²)
  per compartment  : the largest of these over its active frames (+ the frame it came from);
                     compartments with < 2 events left out (paper rule: recurring = >= 2 separate events)
  -> docs/paper_step2/tables/largest_hotspot.xlsx
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

SPONT = Path("results/spontaneous")
OUT = Path(__file__).parents[1] / "tables" / "largest_hotspot.xlsx"

rec = pd.read_excel(SPONT / "spontaneous_summary.xlsx", sheet_name="recordings")
rec = rec[(rec["sensor"] == "GACh3.0") & (rec["n_compartments"] > 0)]

rows = []
for stem in rec["recording"]:
    comps = pd.read_excel(SPONT / f"{stem}_ZONES.xlsx", sheet_name="compartment_stats")
    um2_per_px = float((comps["area_um2"] / comps["area_px"]).iloc[0])
    recurring = set(comps.loc[comps["n_events"] >= 2, "compartment_id"])  # paper rule: >= 2 separate events
    frames_by = {int(c.compartment_id): sorted(set(ast.literal_eval(c.active_frames)))
                 for c in pd.read_excel(SPONT / f"{stem}_ZONES.xlsx", sheet_name="compartments").itertuples()
                 if c.compartment_id in recurring}
    footprints = np.load(SPONT / "footprints" / f"{stem}_ZONES.npz")
    all_frames = sorted({f for fr in frames_by.values() for f in fr})
    pages = tifffile.imread(SPONT / "mask" / f"{stem}_HOTSPOT_MASK.tif", key=[f - 1 for f in all_frames]) > 0
    if pages.ndim == 2:
        pages = pages[None]
    labels = {f: ndimage.label(pages[i])[0] for i, f in enumerate(all_frames)}  # blobs per frame

    for cid, frames in frames_by.items():
        coords = footprints[f"zone{cid}_footprint"]
        best_px, best_frame = 0, None
        for f in frames:
            lab = labels[f]
            hit = lab[coords[:, 0], coords[:, 1]]
            hit = hit[hit > 0]
            if not hit.size:
                continue
            blob = np.bincount(hit).argmax()  # blob with the largest overlap with the footprint
            size = int((lab == blob).sum())
            if size > best_px:
                best_px, best_frame = size, f
        rows.append({"recording": stem, "compartment_id": cid, "footprint_um2": len(coords) * um2_per_px,
                     "largest_hotspot_px": best_px, "largest_hotspot_um2": best_px * um2_per_px,
                     "frame": best_frame})

df = pd.DataFrame(rows)
df.to_excel(OUT, index=False)
x = df["largest_hotspot_um2"]
q1, med, q3 = np.percentile(x, [25, 50, 75])
print(f"{len(rec)} recordings, {len(df)} compartments")
print(f"largest hotspot per compartment: median {med:,.0f} µm² [IQR {q1:,.0f}–{q3:,.0f}], "
      f"range {x.min():,.0f}–{x.max():,.0f}")
for th in (50_000, 100_000, 200_000):
    print(f"  compartments with a hotspot >= {th:,} µm²: {(x >= th).sum()} ({(x >= th).mean():.0%})")
ex = df[(df["recording"] == "2025_06_11-0008_BIEXP_ALS") & (df["compartment_id"] == 2)]
print("check 0008 compartment 2:\n", ex.to_string(index=False))
print(f"saved -> {OUT.resolve()}")
