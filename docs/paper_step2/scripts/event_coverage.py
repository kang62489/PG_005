# ruff: noqa: INP001
"""Scratch: per-event coverage of every GACh3.0 compartment (results/spontaneous).

  per active frame : coverage = hotspot px inside the compartment footprint / footprint px
                     (hotspot mask = all hotspots of that frame, mask/{stem}_HOTSPOT_MASK.tif)
  per event        : run of consecutive active frames -> its max frame coverage
  per compartment  : max event coverage; "large" = >= 1 event with coverage >= COVER_TH
  -> docs/paper_step2/tables/event_coverage.xlsx (events, compartments)
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

SPONT = Path("results/spontaneous")
OUT = Path(__file__).parents[1] / "tables" / "event_coverage.xlsx"
COVER_TH = (0.8, 0.9)

rec = pd.read_excel(SPONT / "spontaneous_summary.xlsx", sheet_name="recordings")
rec = rec[(rec["sensor"] == "GACh3.0") & (rec["n_compartments"] > 0)]

event_rows, comp_rows = [], []
for stem in rec["recording"]:
    comps = pd.read_excel(SPONT / f"{stem}_ZONES.xlsx", sheet_name="compartments")
    footprints = np.load(SPONT / "footprints" / f"{stem}_ZONES.npz")
    frames_by = {int(c.compartment_id): sorted(set(ast.literal_eval(c.active_frames))) for c in comps.itertuples()}
    all_frames = sorted({f for fr in frames_by.values() for f in fr})
    pages = tifffile.imread(SPONT / "mask" / f"{stem}_HOTSPOT_MASK.tif", key=[f - 1 for f in all_frames]) > 0
    if pages.ndim == 2:
        pages = pages[None]
    page_of = {f: i for i, f in enumerate(all_frames)}

    for cid, frames in frames_by.items():
        coords = footprints[f"zone{cid}_footprint"]
        inside = {f: pages[page_of[f]][coords[:, 0], coords[:, 1]] for f in frames}  # footprint px hit per frame
        cover = {f: inside[f].mean() for f in frames}
        runs = np.split(np.array(frames), np.flatnonzero(np.diff(frames) > 1) + 1)  # consecutive frames = 1 event
        # leave-one-out footprint: union of all OTHER events' pixels (removes "the event defines its own footprint")
        ev_union = np.array([np.any([inside[f] for f in run], axis=0) for run in runs])
        n_hit = ev_union.sum(axis=0)
        best = []
        for k, run in enumerate(runs):
            c = max(cover[f] for f in run)
            best.append(c)
            loo = (n_hit - ev_union[k]) > 0
            c_loo = max(inside[f][loo].mean() for f in run) if len(runs) > 1 and loo.any() else np.nan
            event_rows.append({"recording": stem, "compartment_id": cid, "event": k + 1, "first_frame": int(run[0]),
                               "n_frames": len(run), "coverage": float(c), "coverage_loo": float(c_loo),
                               "loo_px": int(loo.sum())})
        comp_rows.append({"recording": stem, "compartment_id": cid, "footprint_px": len(coords),
                          "n_events_runs": len(runs), "max_coverage": float(max(best)),
                          "median_coverage": float(np.median(best)),
                          **{f"n_events_ge_{th}": int(sum(b >= th for b in best)) for th in COVER_TH}})

events, compartments = pd.DataFrame(event_rows), pd.DataFrame(comp_rows)
with pd.ExcelWriter(OUT) as writer:
    events.to_excel(writer, sheet_name="events", index=False)
    compartments.to_excel(writer, sheet_name="compartments", index=False)


def iqr(x: pd.Series) -> str:
    q1, q3 = np.percentile(x, [25, 75])
    return f"median {x.median():.2f} [IQR {q1:.2f}–{q3:.2f}], mean {x.mean():.2f} ± {x.std():.2f}"


print(f"{len(rec)} recordings, {len(compartments)} compartments, {len(events)} events (runs of consecutive frames)")
print(f"event coverage        : {iqr(events['coverage'])}")
print(f"compartment max cover : {iqr(compartments['max_coverage'])}")
for th in COVER_TH:
    n_ev = int((events["coverage"] >= th).sum())
    n_c = int((compartments[f"n_events_ge_{th}"] > 0).sum())
    per_rec = compartments.groupby("recording")[f"n_events_ge_{th}"].apply(lambda s: (s > 0).mean())
    print(f">= {th:.0%}: events {n_ev}/{len(events)} ({n_ev / len(events):.1%}) | compartments with >= 1 such event "
          f"{n_c}/{len(compartments)} ({n_c / len(compartments):.1%}) | per recording, fraction of its compartments: "
          f"{iqr(per_rec)} | recordings with >= 1: {(per_rec > 0).sum()}/{len(per_rec)}")
loo = events.dropna(subset=["coverage_loo"])
loo_max = loo.groupby(["recording", "compartment_id"])["coverage_loo"].max()
print(f"\nleave-one-out ({len(loo)} events, {len(loo_max)} compartments with >= 2 events)")
print(f"event coverage_loo    : {iqr(loo['coverage_loo'])}")
print(f"compartment max loo   : {iqr(loo_max)}")
for th in COVER_TH:
    n_ev = int((loo["coverage_loo"] >= th).sum())
    n_c = int((loo_max >= th).sum())
    n_r = loo_max[loo_max >= th].index.get_level_values(0).nunique()
    print(f">= {th:.0%}: events {n_ev}/{len(loo)} ({n_ev / len(loo):.1%}) | compartments {n_c}/{len(loo_max)} "
          f"({n_c / len(loo_max):.1%}) | recordings with >= 1: {n_r}/{loo_max.index.get_level_values(0).nunique()}")
print("event count check (runs vs n_events in summary):",
      int(compartments["n_events_runs"].sum()), "vs",
      int(pd.read_excel(SPONT / "spontaneous_summary.xlsx", sheet_name="compartments")
          .query("sensor == 'GACh3.0'")["n_events"].sum()))
print(f"saved -> {OUT.resolve()}")
