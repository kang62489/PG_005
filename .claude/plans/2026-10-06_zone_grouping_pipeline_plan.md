# Plan: revised zone grouping → pipeline (2026-10-06)

Story TODO 8 (`docs/continue_from_here.md`). Rules agreed with the user and tested in scratch
(`output/test01/all_zones_circle.py`, 0003 + 0012). Documented in `docs/paper_discussion_2026-10-02.md`
Step 2 "Zone grouping". Phase by phase; user checks after each phase. **No code touched before approval.**

## Rules to port

| # | Grouping | Rule | Constant |
|---|---|---|---|
| 0 | Detect | peak + 2σ; mask cleanup drops blobs < 4,000 px (`TH_SMALL_OBJ`); **no 1 % lower limit**, 80 % upper kept | `MIN_HOTSPOT_FRAC` removed |
| 1 | Consecutive frames → units | hotspot n+1 joins hotspot n if its centroid is inside n's circle (centroid → farthest pixel) | `MAX_CENTROID_DEVIATION` removed |
| 2 | Trace correlation | unit-unit r = best r among their hotspots' own traces (no averaging); complete linkage, cut r ≥ 0.95 | `MIN_GROUP_CORR` |
| 3 | Merge zones | zone A ≥ 95 % inside zone B (shared px / A px) → merged, best pair first, smaller into larger, repeat | new `MIN_INSIDE = 0.95` |
| 4 | Fit leftovers | leftover unit → best zone if its best hotspot is ≥ 90 % inside (shared / hotspot px); zone masks fixed after step 3 | new `MIN_FIT = 0.90` |
| 5 | Leftovers that overlap | isolated split (A) 0 px with every zone / (B) the rest; within each set, units sharing ≥ 1 px chained → zone (≥ 2 units); single units dropped | — |

Proximity grouping (115 px) removed. Known minor issue: step 5 chains (TODO 9, accepted for now).

## Status (2026-10-07, Session 85) -- phases 1–4 done and committed

- Phase 1 ✅ ported, identical to scratch at merge 0.95 (`TH_MERGE_ZONES = 0.95`, `TH_FIT_ZONES = 0.90`; 0.9 tried, back to 0.95).
- Phase 2 ✅ + **Rule B**: grouping-5 zones = **NR zones** (`non_recur_zones` sheet, light gray NR1.. on ZONE_MAPS, NR hotspots hatched `////`), out of every stat; no all-contours page, no extra PNG.
- Phase 3 ✅ `area_comparison.py` reads the `compartments` sheet (`compartment_id`, `active_frames`).
- Phase 4 ✅ neat-refactor of the 4 touched files, identical (`output/test03/`); then naming zone → compartment (sheets, columns, logs) + `mask/_HOTSPOT_MASK.tif`, values identical (`output/test03/compare_renamed.py`).
- TODO 9 resolved by Rule B.
- **Phase 5 pending** -- but first TODO 10 (paper statistics, `docs/continue_from_here.md`), since `area_comparison.py` may change.
- 2026-10-07 (Session 86): Phase 5 spontaneous part ✅ (Saion, all proc TIFFs re-made with the latest code → `results/spontaneous/`). `area_comparison.py` run = step 3, waits for TODO 23 (logic first).

## Phases (original plan)

1. **`classes/sp_zone_analyzer.py`**: constants; `group()` 2a (no lower limit) + 2b (circle linking) + 2c/2d (best-r
   trace-corr); new step 3 / 4 / 5 functions; `map()` builds zones from the new groups; `save()` sheets:
   counts, zone_stats, zones (members / fitted / merged), step3_merge, step4_fit, step5_overlap (replacing
   proximity_groups / isolated_tracks). Remove `second_grouping`, `track_mean_traces` if unused.
2. **`spontaneous_analysis.py`**: `export_zone_maps` page 2 = all zone contours; frame pages tolerate dropped
   hotspots ("+ n dropped"); summary columns `n_trace_corr` / `n_proximity` / `n_isolated` → new source counts
   (+ n dropped units / hotspots).
3. **`area_comparison.py`**: `GROUP_SHEETS` reads active frames from the old sheet names → read from the new
   `zones` sheet instead.
4. **Verify** on `output/test9/proc_test9.txt` (0009, 0003, 0011, 0012) → `output/test*/`: 0003 / 0012 must match the
   scratch numbers (11 / 11 zones, 552 / 554 and 100 / 103 hotspots in zones). ruff + neat-refactor.
5. **Formal run** (user) → `results/spontaneous/`, then re-run `area_comparison.py`; refresh story step-2 results
   (R1 map, R2 frequency) and the numbers in the docs.

## Open for the user
- Step 5 chaining (TODO 9): keep, or stricter rule (shared / smaller ≥ 0.5, or every two units overlap).
- Very large trace-corr zones in 0003 (up to ~30 % of the frame) — come from step 2, not the merge; not discussed further.
