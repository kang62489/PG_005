# Spontaneous striatum coverage (2026-09-26)

Scope: `spontaneous_analysis.py`, `classes/sp_zone_analyzer.py`, `functions/plot_results.py` (`plot_zone_overview`),
reads `data/bd_{date}_{serial}.json` from the Striatum Boundary popup.
Executed phase by phase -- user checks after each phase. No code is touched before approval.

## Decisions (user, 2026-09-26)

| # | Decision |
|---|---|
| R1 | Frequency stats: only 1-event zones are left out (`MIN_EVENTS_FOR_FREQ` 3 -> 2). 1-event zones are still used for coverage |
| R2 | No high-frequency flag / exclusion: `HIGH_FREQ_FLAG_HZ`, `high_freq_flag`, `n_high_freq_zones` removed; every zone with >= 2 events is in the freq stats, every zone is in the coverage |
| C1 | Coverage = area(union of ALL zone masks ∩ striatum) / area(striatum). Union, not sum: zones nest (TODO J) |
| C2 | No bd file / recording not in it / shape mismatch -> coverage NaN + yellow warning; rest of the run unchanged |
| C3 | Striatum outline drawn on the ZONE_MAPS overview page (page 1) |
| C4 | CLI option `--stbd <path>` (default: `bd_export_path(proc_list)`) |

Example (numbers): striatum 1,000,000 px; zone A 200,000 px, zone B 80,000 px inside A -> union 200,000 px
-> coverage 0.20 (a sum would say 0.28).

---

## Phase 1 -- Frequency rules R1 + R2 (DONE, uncommitted)

Test set `output/test9/proc_test9.txt`: `output/test12/baseline/` vs `output/test12/rules/` (`compare_rules.py`):
per-zone sheets identical except the dropped flag column; summary changes only in the freq stats
(0003: 14 -> 20 freq zones, q1 0.106 -> 0.090, CV 0.26 -> 0.59; 0011: 0 -> 1 freq zone, 0.426 Hz; 0012 unchanged).
(`plot_zone_stats()`, not exported, still tolerates a missing `high_freq_flag` column -- left as is.)

## Phase 2 -- Coverage from the bd file

1. CLI `--stbd <path>` (default `bd_export_path(proc_list)`, e.g. `data/bd_20260922_000.json` for
   `data/proc_20260922_000[_saion].txt`); test lists pass `--stbd data/bd_20260922_000.json`.
2. Step 1 (select): load the bd once; per recording `striatum = outline_mask(outline, image_shape)` or None + warning.
3. Per recording: `union = OR of all zone_masks`; summary columns `striatum_area_um2`, `zone_area_in_striatum_um2`,
   `striatum_coverage` (0-1), NaN without a striatum. Shape mismatch -> NaN + warning.
4. Log line per recording: `coverage 0.23 (striatum 0.88 of FOV)`.

Verify: test9 with `--stbd data/bd_20260922_000.json`; hand-check one recording in a scratch script;
test9 without `--stbd` -> NaN + warning, nothing else changes.

## Phase 3 -- Striatum outline on ZONE_MAPS page 1

`plot_zone_overview(..., striatum_outline=None)`: closed outline (white dashed, lw 1.5) over the zone fills;
frame pages unchanged. Verify: page 1 by eye; pages 2+ pixel-identical to Phase 2.

## Notes

- saion: copy `data/bd_20260922_000.json` to the cluster `data/` next to `proc_20260922_000_saion.txt`.
- Neat-refactor pass on touched files after Phase 3.
