# Refined TODOs (2026-09-24, replaces Session 61's six-item list)
Plan: `.claude/plans/2026-09-24_als_spontaneous_flow_plan.md` -- executed phase by phase, user checks after each phase.
All analysis now focuses on `*_BIEXP_ALS.tif` in `proc_tiffs/`.

| # | TODO | Absorbs old item(s) | Phase | Status |
|---|------|---------------------|-------|--------|
| A | Default the pipeline (CLI + GUI) to ALS | new | 0 | [x] |
| B | Merge `../PG_010/sp_ach_zones.py` into PG_005 as classes/functions: 10X only, sensor-aware (iAChSnFR / GACh3.0 / rACh1h), numba CPU+CUDA acceleration, readable xlsx names/columns, zone size/frequency/period stats, output to `results/spontaneous/` | #1 wave vs hotspots, #4 spontaneous side | 1 | [x] |
| C | SpikeReliabilityChecker: merge `prototype_reliability_group_analysis.py` -- success/failure Vm (±50 ms) PNGs in `reliability/`, with AP threshold voltages marked | #4 induced side, spike zoom-in | 2 | [x] committed (`325e09f`) |
| D | RegionAnalyzer: remove ring peak-latency analysis, merge `prototype_flow_analysis.py` (pre-masked TV-L1 flow), `flow/` replaces `latency/` | #5 locality argument | 3 | [x] tuned (Session 66): thresholds fixed; flow now UNMASKED TV-L1 then CAT mask; FLOW.png 4 rows (quivers + µm/s speed, masked / full field). Committed `3d59aca` |
| E | Further flow analysis -> `flow_pairs` table | #5 locality argument | 4 | [x] Session 68: flow threaded (~44 s -> ~15 s, identical); scope changed to per-pair pattern source / sink / anisotropic (CAT-mask linear fit) in `flow_pairs` + new `*_STREAMLINES.png`. Radial outflow / divergence / coherence / centre location dropped by user (uncommitted) |
| F | Neatness refactor of touched scripts (`sp_ach_zones.py` style: short docstrings, step-banner blocks), behavior-identical, done before each phase's feature change | new | 0-4 | [x] Session 69: all 7 files refactored (`hotspot_flow`, `spatial_categorization`, `plot_results`, `ach_domain_analysis`, `spike_reliability`, `region_analyzer`, `abf_clip`), each verified byte-identical in `output/test7/` (uncommitted -- user commits) |

Still open, outside this plan:
- [ ] Sort and mark the dataset used, separate results by objective, put it in the bucket for Jeff (old #6, merged with old #2 on 2026-09-25)
- ~~Compare the median with its segments (old #3)~~ -- dropped 2026-09-25

# Spontaneous refinement TODOs (2026-09-26)
Plan: `.claude/plans/2026-09-26_spontaneous_refine_plan.md` -- phase by phase, user checks after each phase. G-J done + 2σ threshold, committed `6a4c265`.
Test set: `output/test9/proc_test9.txt` (4 recordings, see below); baseline -> `output/test9/before/`.

| # | TODO | Phase | Status |
|---|------|-------|--------|
| G | Output layout: `_ZONE_MASK.tif` -> `mask/` + `--save_mask` toggle (default off) + zlib (was 1.26 GB = 1200×1024×1024 uint8 uncompressed); `_ZONES.npz` -> `footprints/`; xlsx + ZONE_MAPS directly in `spontaneous/`; stop exporting `spontaneous_stats.png` | 1 | [x] 2026-09-26 accepted: mask saved by DEFAULT (zlib, ~2 MB; `--no_mask` skips), npz -> `savez_compressed` (~0.5-1.3 MB); all contents identical to `output/test9/before/` (`compare_g.py`) |
| H | Drop per-frame hotspots > 80 % of the frame (838,861 px) after fragment merge, before grouping; log only (no summary column). Also explore user's idea: quantile clip of hotspot areas (0.001, 0.999) vs speckles / over-exposed first frames -- scratch plots first | 2 | [x] 2026-09-26 accepted: keep merged hotspots 1 %-80 % of frame (10,486-838,861 px); `TH_SMALL_HOTSPOTS` removed; no quantile clip (scratch `hotspot_areas.py`). Frame-1 giants dropped in 0009 / 0011 |
| I | ZONE_MAPS redesign: z-score with one shared min/max; page 1 = max proj (all frames) + all zones; then one page per detection frame with its zone contours + white outline of the actual hotspot; titles show `thr = peak + nσ` and max z of the frame's hotspots (no "meanproj"); remove `--proj` / `--color` (approved) | 3 | [x] 2026-09-26 accepted: `fit_background()` split out; gray z range = `MAP_Z_MIN` 1.0 -> median of detections' max z (min/max was stretched by bright specks); titles `thr = c + k × σ = thr`, `max z`; pages streamed to TIFF. (0011 early zone / 0003 frame-1 hotspot: user said not a problem, dropped) |
| J | Close/overlapping zones: explore centroid distance / IoU / overlap coefficient in scratch first, user picks rule, then `merge_close_zones()` Step 3b | 4 | [x] 2026-09-26 user chose NO MERGE for now: zones are unions of footprints, big zones contain small ones (oc ~1, IoU 0.06-0.2); any oc rule chains a recording into 1-3 zones (`zone_overlap.py`, `merge_preview.py`). Idea for later: "core" zone masks. Frame-page titles now also list zone source names |
| - | Threshold sigma 1.5 -> 2.0 (`CROSSOVER_RATIO`, CLI `--sigma` default) -- user 2026-09-26; test run `output/test9/after_sigma2/`: zones 0009 4->0, 0003 34->26, 0011 12->6, 0012 19->12. Saion re-run postponed by user | - | [x] |
| - | 1-event zones -> NaN freq / period (was 1/60 Hz floor); summary `n_low_freq_zones`, `n_freq_zones`, median / IQR / CV -- Session 73. Session 74: `MIN_EVENTS_FOR_FREQ` 3 -> 2 (only 1-event zones out of freq stats) | - | [x] |
| - | High-frequency flag removed (Session 74, user): no `HIGH_FREQ_FLAG_HZ` / `high_freq_flag` / `n_high_freq_zones`; every zone is used in freq stats (>= 2 events) and coverage | - | [x] |
| K | Striatum Boundary popup (plan `.claude/plans/2026-09-26_st_boundary_popup_plan.md`) + spontaneous coverage (plan `.claude/plans/2026-09-26_spontaneous_coverage_plan.md`) | - | [x] popup done + `data/bd_20260922_000.json` exported (201 rec, 105 outlines); coverage + ZONE_MAPS outline / direction labels done. Left: flow direction in DV / ML |

Test set (2026-09-26): `2024_10_11-0009`, `2025_06_11-0003`, `2025_11_27-0011`, `2025_12_15-0012` -> `output/test9/proc_test9.txt` (only the list is kept; spontaneous tests pass `--stbd data/bd_20260922_000.json`).

Spike-aligned follow-ups from the 2026-09-26 results check (not started):
- [ ] 60X decay fit fails on 43/62 significant recordings (median R² 0.4) -- look at a few SPATIAL.png.
- [ ] 23 no-peak entries in the deigo run -- spontaneous-only recordings? drop from the ana list if so.
- [ ] Copy back `data/ana_20260922_000_deigo.txt` from deigo (has the `[SKIPPED]` / stats block).

---

# Log of the project progress 2026-09-26 Sat (Session 74)
Last working file: `controllers/ctrl_st_boundary.py`
Last working line: 488-510 (`on_load_bd` / `_convert_bd` -- Load bd file in a worker thread, moved after Export by the neat-refactor)

## List of modified files (UNCOMMITTED except the popup base `89cd1db` -- user commits)
- `functions/st_boundary.py` (new in `89cd1db`, extended) — orientation (`medial_from`, `direction_labels`), raw preview, anchor Catmull-Rom curve + `snap_ends` + regions (`MIN_REGION_PX = 100`), draft storage, export (`bd_export_path`, `export_entry` incl. anchors / seeds), re-edit (`draft_from_export`, `outline_to_anchors`, fast `_deepest_point` seed)
- `controllers/ctrl_st_boundary.py`, `views/view_st_boundary.py` — popup: Unchecked / Confirmed lists, dorsal combo, anchor tool (click / drag / Finish line / Undo / Clear), right-click striatum, Shift+right-click cortex check, Confirm -> `data/st_bd_draft.json`, Export -> `bd_{date}_{serial}.json`, Load bd file (worker thread)
- `classes/dialog_get_path.py` — `get_bd_file()`
- `classes/model_from_dataframe.py`, `controllers/ctrl_align_spike.py`, `controllers/ctrl_img_proc.py` — bold red missing rows / gray `N/A` unpaired ABF (committed `89cd1db`)
- `classes/sp_zone_analyzer.py` — `MIN_EVENTS_FOR_FREQ = 2`, high-freq flag removed
- `spontaneous_analysis.py` — STEP 2b coverage (`striatum_of`, `zone_coverage`), `--stbd` (default `bd_{date}_{serial}.json` next to the proc list), summary `striatum_area_um2` / `zone_area_in_striatum_um2` / `striatum_coverage`, outline + direction labels on ZONE_MAPS page 1
- `functions/plot_results.py` — `plot_zone_overview(..., striatum_outline, axis_labels)`
- `functions/__init__.py` — registry
- `data/bd_20260922_000.json` (new, final product), `data/st_bd_draft.json` (holds 201 entries from the real-GUI Load-bd test)
- Plans: `.claude/plans/2026-09-26_st_boundary_popup_plan.md`, `.claude/plans/2026-09-26_spontaneous_coverage_plan.md`
- Scratch: `output/` cleaned (~12 GB); kept `output/test9/proc_test9.txt`, `output/test12/` (checks), `output/test13/` (neat-refactor before / after)

## Summary of current progress
- Striatum boundary: 201 recordings oriented (dorsal set by hand, medial from SLICE L / R), 105 10X striatum outlines; export 521 KB; outline -> mask IoU >= 0.9999.
- Re-edit: old exports without anchors rebuild all 105 lines (IoU min 0.991, median 0.9995, ~12 anchors); full file loads in 0.7 s (was a 38 s GUI freeze).
- Coverage = union of ALL zones ∩ striatum / striatum (union, not sum: 0003 sum would be 2.42). Test set: 0009 0.000, 0003 0.643, 0011 0.559, 0012 0.604; hand-check identical to 6 decimals.
- Freq rules: 1-event zones out of freq stats but in coverage; no high-frequency exclusion (19 / 2,971 old-run zones at 6.67-10 Hz look like split events -- left in, per user).
- Neat-refactor of touched files: behavior identical (19 checks, `output/test13/compare.py`).

## Completed TODOs/Tasks (before new wrap-up)
- ✅ TODO K popup (design -> anchor tool -> export / re-edit) and `bd_20260922_000.json`
- ✅ Coverage in `spontaneous_analysis.py` + outline / direction labels on ZONE_MAPS page 1
- ✅ Frequency rules (>= 2 events, no high-freq flag)
- ✅ Neat-refactor, `output/` cleanup

## Before the saion run (user)
- Commit + push; `git pull` on saion; `data/bd_20260922_000.json` must be in `$PROJECT/data/` next to `proc_20260922_000_saion.txt` (found automatically, no `--stbd`); then `sbatch run_on_saion.slm`.

## What should we do next? (TODOs)
- [ ] Check the saion results: no coverage NaN warnings expected for the 105 10X recordings; look at the coverage spread and at the freq stats now that 2-event zones count.
- [ ] Reset `data/st_bd_draft.json` to `{}` (201 entries from the Load-bd test) unless the boundaries are going to be edited.
- [ ] Flow direction in DV / ML: use `dorsal_vec` / `medial_vec` from `data/bd_20260922_000.json` in the spike-aligned flow analysis (rest of TODO K).

## Last Session Recap
※ recap: Built the Striatum Boundary popup (anchor tool, export / re-edit of `bd_20260922_000.json`), added striatum coverage + outline / direction labels to spontaneous ZONE_MAPS, simplified freq rules, neat-refactored. Pending: commit, saion run, result check, flow direction in DV / ML.

---

# Log of the project progress 2026-09-26 Sat (Session 73)
Last working file: `spontaneous_analysis.py`
Last working line: 225-236 (summary row: `n_low_freq_zones`, `n_freq_zones`, `median_freq_hz`, `freq_q1_hz`, `freq_q3_hz`, `freq_cv`, `median_period_s`)

## List of modified files (UNCOMMITTED -- user commits)
- `classes/sp_zone_analyzer.py` — `MIN_EVENTS_FOR_FREQ = 3` (line 70); `zone_event_stats()` (line 518): 1-event zones -> period / freq NaN (was whole recording = 1/60 Hz floor), unused `n_frames` arg removed from it and `zone_stats_table()`
- `spontaneous_analysis.py` — summary: `n_low_freq_zones` (1 event = "< 1 per recording"), `n_freq_zones` (>= 3 events), median freq / period + `freq_q1_hz` / `freq_q3_hz` / `freq_cv` from >= 3-event zones only
- `export_als_segments.py`, `export_zscore_segments.py` — deleted (`git rm`, staged; old diagnosis scripts, not imported anywhere)
- Scratch: `output/test9/` cleaned (~6.6 GB freed); kept `proc_test9.txt`, `after_sigma2/` (old baseline), new `after_freq/` run, `count_1event.py`, `compare_freq.py`

## Summary of current progress
- 1-event zones = 16 / 44 in the test set (0003 6/26, 0011 5/6, 0012 5/12); count equals `n_isolated` in every recording.
- 1/60 Hz floor pulled the medians down (0011 median = floor); freq-vs-period median mismatch comes from even-count medians.
- `after_freq/` vs `after_sigma2/`: all xlsx sheets identical except 1-event period / freq -> NaN (`compare_freq.py`).
- New summary: 0003 median 0.117 Hz (IQR 0.106-0.141, CV 0.26, 14 zones); 0011 0 measurable zones; 0012 0.103 Hz (IQR 0.085-0.198, CV 1.88 -- driven by zone 7, 2.2 Hz, likely one split event).
- Paper-claim notes: "not in phase" needs an event-timing (synchrony) measure, not frequency spread; "covers striatum" needs the TODO K boundary; sensor comparison per recording, needs full-dataset run.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Clean `output/test9/` scratch runs
- ✅ Remove unused `export_*.py` from the root
- ✅ 1-event zones -> NaN freq / period, `n_low_freq_zones`, summary freq spread (>= 3 events)

## What should we do next? (TODOs)
- [ ] TODO K: user posts the GUI design PNG; then settle (1) button location (img-proc view vs new spontaneous view), (2) image shown (ALS max proj vs raw mean/frame), (3) polyline divider vs closed striatum polygon, (4) ROI save path (e.g. `spontaneous/roi/{stem}_ROI.json`) read by `spontaneous_analysis.py` for coverage.
- [ ] Open question (user, end of Session 73): high-freq flagged zones (> 1 Hz) are currently INCLUDED in the summary freq stats (only `n_events >= 3` filters). Decide: exclude flagged zones, or first check whether 0012 zone 7 (2.2 Hz) is one event split by a gap frame.

## Last Session Recap
※ recap: Cleaned `output/test9/` and removed the unused `export_*.py`; 1-event zones now get NaN frequency and the summary reports median / IQR / CV from >= 3-event zones (verified vs `after_sigma2/`). Pending: commit, then TODO K GUI design from the user's PNG.

---

# Log of the project progress 2026-09-26 Sat (Session 72)
Last working file: `spontaneous_analysis.py`
Last working line: 149-151 (frame-page title: 2nd line `zones <id> (<source>)`, wrapped at `MAP_TITLE_WIDTH`)

## List of modified files (all committed by user: `8133ff8` backup, `6a4c265` "fixed/improved almost all spontaneous_analysis.py")
- `spontaneous_analysis.py` — new layout, `--no_mask`, stats PNG dropped, ZONE_MAPS redesign (streamed pages, z range `MAP_Z_MIN` 1.0 -> median detection max z, value titles + zone source names), `--proj` / `--color` removed, `--sigma` default = `CROSSOVER_RATIO`
- `classes/sp_zone_analyzer.py` — `save()` -> `footprints/` + `mask/` (zlib, default on), `savez_compressed`; hotspot band `MIN_HOTSPOT_FRAC` 0.01 / `MAX_HOTSPOT_FRAC` 0.8 (drop logs); `TH_SMALL_HOTSPOTS` removed; stores `bg_center` / `bg_sigma`; `CROSSOVER_RATIO` 1.5 -> 2.0
- `functions/fit_hist.py` — `fit_background()` split out of `find_background_threshold()` (same value)
- `functions/plot_results.py` — `plot_zone_overview()` / `plot_frame_zones()` replace `plot_zone_overlay()` / `plot_single_zone()` / `_tint_background()`
  - 2026-09-26 (uncommitted): `plot_frame_zones()` -> generator `frame_zone_figures()` (one reused Figure, zone contours built once + toggled, hotspot contour on a cropped bbox). ZONE_MAPS export 74.0 s -> 47.9 s on 2025_06_11-0003 (345 pages), pixel-identical; bench in `output/test14/`
- `functions/__init__.py` — exports updated
- `run_on_saion.slm`, `PIPELINE_FLOW.md` — output-layout comments
- Scratch (not tracked): `output/test9/` — `proc_test9.txt`, runs `before/ after_g/ after_g2/ after_h/ after_i*/ after_j/ after_sigma2/`, scripts `compare_g.py`, `hotspot_areas.py` (stale `TH_SMALL_HOTSPOTS` import), `check_i.py`, `zone_pages.py`, `max_z_spread.py`, `zone_overlap.py`, `merge_preview.py`

## Summary of current progress
- Test set changed to `2024_10_11-0009`, `2025_06_11-0003`, `2025_11_27-0011`, `2025_12_15-0012`.
- G: mask 1.26 GB -> ~2 MB (zlib), npz 20-65 MB -> 0.4-1.3 MB; contents byte-identical to baseline.
- H: merged hotspots kept only if 1 %-80 % of frame; frame-1 over-exposed giants (100 % / 86 %) removed; quantile clip rejected (drops real hotspots).
- I: ZONE_MAPS = max-proj overview + one page per detection frame; plain min/max range was stretched by bright specks (z 74 / 60) -> median of detection max z.
- J: no merge — zones are footprint unions, big zones nest small ones; any overlap rule chains a recording into 1-3 zones.
- Threshold 1.5σ -> 2σ (test zones: 0009 4->0, 0003 34->26, 0011 12->6, 0012 19->12).

## Completed TODOs/Tasks (before new wrap-up)
- ✅ TODO G, H, I, J (spontaneous refinement plan)
- ✅ Group (source) names on frame pages
- ✅ Threshold sigma 2.0

## What should we do next? (TODOs)
- [ ] TODO K: discuss the GUI popup design (raw TIFF preview + mouse-drawn striatum / cortex boundary -> coverage + direction).
- [ ] Clean `output/test9/` scratch runs (`before/` alone holds ~4.8 GB of uncompressed masks).
- [ ] 60X decay fit fails on 43/62 significant recordings (median R² 0.4) -- look at a few SPATIAL.png.
- [ ] 23 no-peak entries in the deigo run -- spontaneous-only recordings? drop from the ana list if so.
- [ ] Copy back `data/ana_20260922_000_deigo.txt` from deigo (has the `[SKIPPED]` / stats block).
- [ ] Dataset bucket for Jeff: sort/mark the dataset used + separate results by objective.
- (Postponed by user, not scheduled: saion re-run of spontaneous with the new code.)

## Last Session Recap
※ recap: Finished the spontaneous refinement (G-J): compressed mask/npz layout, 1-80 % hotspot band, z-scored per-frame ZONE_MAPS with source names, no zone merge, threshold 2σ; committed `6a4c265`. Pending: GUI ROI popup (K), test9 cleanup.

---

# Log of the project progress 2026-09-26 Sat (Session 71)
Last working file: `.claude/plans/2026-09-26_spontaneous_refine_plan.md`
Last working line: -- (plan written, no code edited)

## Summary of current progress
- Checked cluster results in `results/`:
  - deigo (`results/logs/ach_44668799.out`, ran pre-acceleration code, 3.9 h): 201 entries -> 163 DB rows (10X 92 / 71 sig, 40X 9 / 9, 60X 62 / 62); 38 skipped = 23 no ABF peaks + 15 spikes too close for a baseline.
  - flow_pairs 704 rows: mostly anisotropic; source only around the spike; sinks peak at spike+1 -> +2.
  - saion (`results/logs/spont_4731852.out`, 1.7 h): 105 recordings, 2,971 zones. Oddities: 12 recordings with 0 zones (2025_11_27 / 12_14 / 12_18), whole-FOV zones (2025_11_27-0005/7/8/9), median freq vs median period mismatch (2025_11_27-0007, 2025_12_14-0020), 1/60 Hz floor, 2025_11_27-0026 threshold 0.095.
- Wrote the spontaneous refinement plan (TODO G-J).

## What should we do next? (TODOs)
- [ ] Review the plan, then start Phase 1 (TODO G) -- first run the current code on the test set -> `output/test9/before/`.

## Last Session Recap
※ recap: Checked deigo/saion results (163 DB rows, 38 skips; 105 spontaneous recordings with whole-FOV / 0-zone oddities) and wrote the spontaneous refinement plan (mask layout + toggle, 80 % hotspot filter, ZONE_MAPS redesign, close-zone merge). Pending: TODO G-J, no code edited.

---

# Log of the project progress 2026-09-26 Sat (Session 70, backfill of commits `d12cf37` + `30fd9f7`)
Last working file: `classes/spatial_categorization.py`
Last working line: 42 (unused `NDIM_SINGLE_FRAME` removed)

## List of modified files
Committed by user (`d12cf37` "try speed up ach_domain_analysis with numba.jit", `30fd9f7` "complete acceleration", 2026-09-25):
- `functions/tvl1_flow.py` (new) — `optical_flow_tvl1_numba()`: numba port of skimage 0.26 TV-L1 (prange over rows, float32, same early stop) -> bit-identical flow.
- `functions/cluster_kernels.py` (new) — `binary_open_close_square()` (separable square open + close) and `within_distance()` (exact integer EDT <= radius); serial + nogil, so thread-safe.
- `functions/hotspot_flow.py` — uses `optical_flow_tvl1_numba`; flow pairs now run one after another (numba is parallel inside), `N_THREADS_FLOW` / ThreadPoolExecutor removed.
- `classes/spatial_categorization.py` — morphological cleanup via `binary_open_close_square` (replaces scipy erosion/dilation).
- `classes/region_analyzer.py` — cluster seeker eps zone via `within_distance` (replaces `distance_transform_edt`).
- `classes/spike_reliability.py` — per-segment check split into `_check_segment()`, run in `N_THREADS_RELIABILITY = 8` threads.
- `classes/abf_clip.py` — `export_plot` flag; `_export_spike_plot` -> public `export_spike_plot()`.
- `ach_domain_analysis.py` — all PNGs (spikes / reliability / spatial / flow) are built + saved as figure jobs on one background thread, which now overlaps the next entry's analysis.
- `pyproject.toml`, `uv.lock` — CUDA 12 nvrtc / nvJitLink pins (Session 69 item, committed here).

UNCOMMITTED:
- `run_on_deigo.slm` / `run_on_saion.slm` — log file prefixes `ach_` / `spont_` -> `deigo_` / `saion_`.
- `classes/spatial_categorization.py` — unused `NDIM_SINGLE_FRAME` removed (ruff OK).

## Summary of current progress
- `ach_domain_analysis.py` accelerated end to end with numba (TV-L1, morphology, EDT) + threaded reliability + overlapped figure thread. Still CPU-only -> deigo.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ numba acceleration of `ach_domain_analysis.py` (committed `d12cf37`, `30fd9f7`)
- ✅ Remove unused `NDIM_SINGLE_FRAME`

## What should we do next? (TODOs)
- [ ] Copy and check the results from the clusters (saion `/work/WickensU/kang/results/spontaneous/`, deigo `/flash/WickensU/kang/results/` + `[SKIPPED]` lines / stats block in `data/ana_20260922_000_deigo.txt`). If the deigo run predates `30fd9f7`, re-run it with the accelerated code.
- [ ] Dataset bucket for Jeff: sort/mark the dataset used + separate results by objective.

## Last Session Recap
※ recap: Backfilled the numba acceleration of ach_domain_analysis (numba TV-L1, morphology/EDT kernels, threaded reliability, background figure thread) and removed the unused NDIM_SINGLE_FRAME. Pending: check cluster results, Jeff dataset bucket.

---

# Log of the project progress 2026-09-25 Fri (Session 69)
Last working file: `pyproject.toml`
Last working line: 29-30 (`nvidia-cuda-nvrtc-cu12` / `nvidia-nvjitlink-cu12` pins via `uv add`)

## List of modified files
Committed by user (`c3e6ff9`, then `spike_reli` fast-forwarded into `main`):
- TODO F neat-refactor, each verified byte-identical in `output/test7/` (spike-aligned + spontaneous runs): `functions/hotspot_flow.py`, `classes/spatial_categorization.py`, `functions/plot_results.py`, `ach_domain_analysis.py` (+ log text "skipping z-score step" -> "skipping this entry"), `classes/spike_reliability.py`, `classes/region_analyzer.py`, `classes/abf_clip.py` (+ typo `trailing_interval_frames`).
- Prototype cleanup: `prototype_flow_analysis.py`, `prototype_reliability_group_analysis.py` -> `archive/`; deleted `prototype_rose_map_batch.py`, `prototype_directional_change.py`, `prototype_hotspot_direction_analysis.py`, `prototype_med_height_map.py`, `prototype_reaction_diffusion.py`, `classes/directional_change.py`, `classes/hotspot_direction_analysis.py`, and `plot_results.py` Step 5b (plot_directional_change, _direction_map, 7 plot_hotspot_*). Kept: `prototype_peak_tracking.py`, `prototype_med_kymographs.py`, `prototype_full_als_waves.py` (+ `classes/wave_profiles.py`, Step 5a plots).
- Cluster scripts: `run_ach.slm` -> `run_on_deigo.slm` (compute, CPU, /flash logs, `data/ana_20260922_000_deigo.txt`); new `run_on_saion.slm` (spontaneous, GPU, /work logs + results, `data/proc_20260922_000_saion.txt`).
- New lists: `data/proc_20260922_000_saion.txt` (footer -> /work/WickensU/kang), `data/ana_20260922_000_deigo.txt` (built from proc_20260922_000, 201 rows, flags set YES unchecked, footer -> /flash/WickensU/kang).

UNCOMMITTED:
- `run_on_saion.slm` — `--partition=gpu` (account can't use `gpu-v100`); resource lines now exactly the user's srun flags (`gpu`, `gpu:1`, `saion-gpu21`, 16G, 12 h) -- my extra `--cpus-per-task=16` / `--nodes` / `--ntasks` removed (the 16 CPUs had kept the job at PD Resources).
- `pyproject.toml`, `uv.lock` — `nvidia-cuda-nvrtc-cu12>=12.9.86`, `nvidia-nvjitlink-cu12>=12.9.86` (keeps nvrtc / nvJitLink on CUDA 12 for the V100; local RTX 5070 Ti kernel test OK).

Scratch: `output/test6` (centre_refine + flow_streamlines scripts/PNGs) and `output/test7` run folders deleted; `output/test7/compare_test7.py <before> <after>` kept (reusable). `output/test4/directional_change/`, `med_height_map/` deleted.

## Summary of current progress
- TODO F finished: 7 files neat, behaviour-identical (20 PNGs, 6 TIFFs, `experiments` + `flow_pairs`, spontaneous zone TIFFs / stats PNG all identical).
- `ach_domain_analysis.py` confirmed CPU-only (no CUDA module imported) -> runs on deigo; `spontaneous_analysis.py` uses the GPU -> runs on saion.
- Both `.slm` jobs ran successfully on the clusters (saion job 4731852 on saion-gpu21).

## Completed TODOs/Tasks (before new wrap-up)
- ✅ TODO F neat-refactor (7 files)
- ✅ test6 / test7 / test4 scratch cleanup, prototype archive + removal
- ✅ spike_reli merged into main (fast-forward)
- ✅ Cluster run setup: slm scripts + `_saion` / `_deigo` lists; both jobs ran

## What should we do next? (TODOs)
- [ ] Copy and check the results from the clusters (saion `/work/WickensU/kang/results/spontaneous/`, deigo `/flash/WickensU/kang/results/` + `[SKIPPED]` lines / stats block in `data/ana_20260922_000_deigo.txt`).
- [ ] Dataset bucket for Jeff: sort/mark the dataset used + separate results by objective.
- [ ] Remove unused `NDIM_SINGLE_FRAME` from `classes/spatial_categorization.py`.

## Last Session Recap
※ recap: Finished TODO F (7 files refactored, all byte-identical), archived/removed old prototypes, merged spike_reli into main, set up run_on_saion.slm / run_on_deigo.slm + cluster lists (both ran), added CUDA 12 pins. Pending: check cluster results, Jeff dataset, NDIM_SINGLE_FRAME.

---

# Log of the project progress 2026-09-25 Fri (Session 68)
Last working file: `functions/plot_results.py`
Last working line: 375 (`_draw_angle_crosshair` -> 0/90/180/270° crosshair on anisotropic STREAMLINES panels)

## List of modified files
Committed by user as `35dec17`:
- `functions/hotspot_flow.py` — `compute_flow_pairs()` runs the 5 `pair_flow()` calls in parallel threads (`N_THREADS_FLOW`).

UNCOMMITTED:
- `functions/flow_pattern.py` (new) — `fit_flow_pattern()`: CAT-mask linear fit flow ~= A(p - centroid) + drift on 8x8 blocks; anisotropic if |drift| > spread (|trace A|/2 * RMS radius), else source (trace > 0) / sink (trace < 0); returns label, drift_um, spread_um (µm/frame), drift_angle_deg (0 = →, 90 = ↑). `block_mean()` shared with the plots.
- `functions/plot_results.py` — new `plot_flow_streamlines()` (row 1 CAT-mask streamlines, row 2 full field; CAT titles `· source` / `· sink` / `· anisotropic 53°`; cyan 0/90/180/270° crosshair on anisotropic panels). FLOW.png (quiver) unchanged.
- `classes/region_analyzer.py` — `compute_flow()` attaches `pair["pattern"]`.
- `classes/results_exporter.py` — new `flow_pairs` table (pair_label, offsets, pattern, drift_um, spread_um, drift_angle_deg; UNIQUE per recording + pair) + `export_flow_pairs()` (deletes the recording's rows first, so non-significant re-runs clear stale rows).
- `ach_domain_analysis.py` — exports `flow/*_STREAMLINES.png` + `flow_pairs` rows.
- `functions/__init__.py` — registers `plot_flow_streamlines`, `fit_flow_pattern`.

Scratch: `output/test8/` deleted (user OK). `output/test6/`: `flow_streamlines.py` (streamline + direction-colour diag), `flow_pattern_fit.py` (+ csv), `centre_refine.py` (+ `centre_refine/` PNGs; no longer runs), `compare_test6.py`, `ana_test6.txt` + pipeline outputs.

## Summary of current progress
- Flow threads: ~44 s -> ~15 s per recording, bit-identical.
- Tried and dropped (user): direction-colour row, streamlines replacing FLOW.png quivers, off-view / no-flow / global rules, full-field fit, source/sink centre markers (linear-fit centre 85-230 µm off; local-score centre biased to line ends on 0012's line source) -> centre removed from code + table.
- Labels on 0003 / 0012 / 0018 match the eye: 0003 anisotropic x4 + sink; 0012 source -> sink x4; 0018 anisotropic, anisotropic, sink, anisotropic x2.
- Verified (`output/test6` vs `output/test5`): FLOW / SPATIAL / reliability / spike PNGs, MED/CAT tifs, `experiments` identical; `flow_pairs` 15 rows match `flow_pattern_fit.csv`.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Flow speed-up (threads), committed `35dec17`
- ✅ STREAMLINES.png + source / sink / anisotropic label + `flow_pairs` table (TODO E, reduced scope)

## What should we do next? (TODOs)
- [x] TODO F via `neat-refactor` (done Session 69, all identical): `functions/hotspot_flow.py`, `classes/spatial_categorization.py`, `functions/plot_results.py`, `ach_domain_analysis.py`, `classes/spike_reliability.py`, `classes/region_analyzer.py`, `classes/abf_clip.py` (light).
- [ ] Dataset bucket for Jeff: sort/mark the dataset used + separate results by objective.
- [x] Clean `output/test6` scratch (`centre_refine.py` + `centre_refine/`, `flow_streamlines.py` + `*_FLOW_STREAM_DIR.png` deleted, Session 69).

## Last Session Recap
※ recap: Threaded TV-L1 flow (~3x faster), added STREAMLINES.png with per-pair source/sink/anisotropic labels + angle crosshair and a `flow_pairs` table (centre markers dropped). Pending: TODO F refactors, Jeff dataset, test6 scratch cleanup.

---

# Log of the project progress 2026-09-25 Fri (Session 67, after commit `3d59aca`)
Last working file: `spontaneous_analysis.py`
Last working line: 127 (`export_zone_maps` -> `tifffile.imwrite(..., compression="zlib")`)

## List of modified files (UNCOMMITTED)
- `functions/hotspot_flow.py` — `premasked_flow` -> `pair_flow`; unused `mask_frame()` removed.
- `functions/__init__.py` — registry: `premasked_flow` -> `pair_flow`.
- `classes/region_analyzer.py`, `ach_domain_analysis.py`, `functions/plot_results.py` — "pre-masked" wording in docstrings/comments updated (no logic change).
- `spontaneous_analysis.py` — zone-map TIFF written with lossless zlib (0003: 230 MB -> 42 MB, pixels identical).
- `.claude/skills/neat-refactor/SKILL.md` (new) — TODO F workflow: one file at a time, plan -> approval -> before run -> edit -> ruff -> after run -> compare identical.
- Scratch: `output/test3/run_test3.py` fixed (local `FLOW_QUIVER_SCALE = 0.8`); stale PNG folders in `output/test5/spontaneous/zone_maps/` deleted.

## Summary of current progress
- Cleanups 4-6 done; old #7 merged into #9 (dataset bucket), old #8 dropped.
- `prototype_flow_analysis.py` left as-is for now (still says "pre-masked"; retired prototype).

## Completed TODOs/Tasks (before new wrap-up)
- ✅ User committed Session 66 work (`3d59aca`)
- ✅ neat-refactor skill created
- ✅ Cleanups: pair_flow rename, run_test3.py fix, stale zone-map folders
- ✅ zlib compression for zone-map TIFFs

## What should we do next? (TODOs)
- [ ] TODO F via `neat-refactor`: start with `functions/hotspot_flow.py`, then `ach_domain_analysis.py`, `classes/spike_reliability.py`, `classes/region_analyzer.py`, `classes/abf_clip.py` (light).
- [ ] Flow speed-up (5 TV-L1 pairs in parallel threads, identical to serial), then Phase 4 / TODO E `flow_pairs` metrics table.
- [ ] Dataset bucket for Jeff: sort/mark the dataset used + separate results by objective.

## Last Session Recap
※ recap: After committing 3d59aca, renamed premasked_flow -> pair_flow, fixed run_test3.py, removed stale zone-map folders, zlib-compressed zone-map TIFFs (230 -> 42 MB), and created the neat-refactor skill. Pending: TODO F refactors, flow speed-up + Phase 4, Jeff dataset.

---

# Log of the project progress 2026-09-25 Fri (Session 66)
Last working file: `functions/plot_results.py`
Last working line: 309 (`plot_flow_panels` -> shared `intensity_range` = p1-p99 of MED for rows 1 and 3)

## List of modified files (all UNCOMMITTED -- user will commit)
- `functions/fit_hist.py` — spontaneous threshold: `ZONE_HIST_BINS = 512`, new `smoothed_peak()` (Savitzky-Golay derivative = 0, window `ZONE_SMOOTH_FRAC = 0.05` of range), `fit_left_gaussian(center=)`; `find_background_threshold()` = smoothed peak + 1.5 * left-fit sigma (seed `ZONE_SIGMA_SEED` kept). `fit_hist_sigma()` (img_proc z-scoring) unchanged.
- `classes/spatial_categorization.py` — `compute_baseline_threshold()`: baseline pixels trimmed to `BASELINE_TRIM_PCT = (0.1, 99.9)`, then mean + 1.5*std (float64). Docstrings "2*std" fixed.
- `functions/hotspot_flow.py` — `premasked_flow()` now runs TV-L1 on the RAW MED pair; `keep_mask` = union of both frames' CAT (applied after, in the plot).
- `functions/plot_results.py` — `plot_flow_panels(med, flow_pairs, title_info, frame_duration_ms)`: 4 rows (1 quivers in CAT mask, 2 speed µm/s in mask, 3 quivers full field, 4 speed full field); red auto-scaled arrows (p95 = 0.9 * 24 px), scale bars, shared gray range, shared speed color scale; `PIXEL_SCALE` imported for µm/px (10X/40X/60X). New `_draw_flow_quivers()`. `FLOW_QUIVER_SCALE` -> `FLOW_AUTO_ARROW_FRAC`. Reliability montage + zone maps (`plot_zone_overlay`, `plot_single_zone`) got scale bars (`obj` / `um_per_px` args).
- `ach_domain_analysis.py` — `plot_flow_panels()` call: no `cat_stack`, + `frame_duration_ms`.
- `classes/spike_reliability.py` — passes `self.obj` to the montage.
- `spontaneous_analysis.py` — zone maps: one RGB TIFF stack `zone_maps/{stem}_ZONE_MAPS.tif` (page 1 all zones, then one page per zone; single series), no PNG subfolders.

Scratch (output/, not code): `output/test1` (spontaneous threshold test + fitting PNGs), `output/test2/{1.5,2}_sigma` (CAT trim; `AVG/`, `AVG_run/` = mean-instead-of-median), `output/test3` (unmasked flow tests; `1.5_sigma_CATMASK_AUTO_clean` = adopted look), `output/test5` (verification of the pipeline edits, `compare_test5.py`). Deleted: `output/test6`, `output/test7`, `output/test2/{2.5,3}_sigma`, `output/test2/and_mask`.

## Summary of current progress
- **Thresholds decided + implemented:** spontaneous = 512 bins / smoothed-peak center / left fit / k 1.5 (0003 0.19113, 0012 0.18630); CAT = p0.1-p99.9 trim + mean + 1.5*std (trim changes thresholds <= 0.6 %). N = 1.5 kept (2 / 2.5 / 3 cut 0003 to 57 / 35 / 25 of 82). MED kept (not AVG).
- **Verified (output/test5):** zone masks + ZONES.npz identical to test1; reliability 75/82, 5/5, 8/8 and MED/CAT tifs identical to test2/1.5_sigma; FLOW arrows match test3 clean.
- **Findings:** AND of detected-segment CATs is empty for 0003 (max count 27/75) -> dropped. AVG shows a bright bottom blob in 0003 already at spike-1 -> dirty baselines in some segments (median hides it). Unmasked TV-L1 mostly measures brightness change (0012 arrows converge on the bright line; 0018 16-68 px/frame).
- **Units:** speed = |flow| px/frame * (1/PIXEL_SCALE) µm/px * fps -> 10X @ 20 fps: 1 px/frame = 26.7 µm/s.
- GPU: skimage TV-L1 is CPU only (no cv2/cupy/torch installed); flow ~45-55 s per recording.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Compared spontaneous vs CAT thresholding; chose and implemented both
- ✅ Docstring fix (mean + 2*std -> BASELINE_SIGMA_MULT)
- ✅ Flow: unmasked TV-L1 -> CAT mask, auto arrow scale, µm/s speed rows, scale bars, red arrows, unified gray range
- ✅ Scale bars on reliability montage + zone maps; zone maps as one TIFF stack
- ✅ Verification run in output/test5

## What should we do next? (TODOs)
- [x] Commit today's edits -- user committed `3d59aca` "completed phase 3 and part of phase 4 (flow speed)".
- [ ] TODO F: run the `neat-refactor` skill file by file (behavior-identical, verified before/after).
- [ ] Flow speed-up: 5 TV-L1 pairs in parallel threads first (measure; must be identical to serial). GPU = own numba-cuda TV-L1, big job.
- [ ] Phase 4 / TODO E flow metrics -> `flow_pairs` table (speed µm/s, radial outflow, divergence, coherence), on the unmasked-then-masked flow.
- [x] Cleanup: `premasked_flow` -> `pair_flow`, unused `mask_frame()` removed, registry + "pre-masked" docstrings updated (uncommitted).
- [x] Cleanup: `output/test3/run_test3.py` fixed (local `FLOW_QUIVER_SCALE = 0.8`).
- [x] Cleanup: stale zone-map PNG folders in `output/test5/spontaneous/zone_maps/` deleted.

## Last Session Recap
※ recap: Compared and fixed thresholds (spontaneous 512-bin smoothed-peak fit; CAT p0.1-p99.9 trim, k 1.5), switched flow to unmasked TV-L1 + CAT mask with µm/s speed rows, scale bars, zone-map TIFF stack; verified in output/test5. Pending: commit, flow speed-up, Phase 4 metrics, cleanups.

---

# Log of the project progress 2026-09-25 Fri (Session 65)
Last working file: `classes/spatial_categorization.py` (no code edited this session)
Last working line: 200 (`compute_baseline_threshold` -> back to mean + 1.5*std; docstring still says 2*std)

## List of modified files
No pipeline code edited. Git history this session:
- `2d3b791` "backup before revert" -- snapshot of all Session 64 work (incl. the threshold unification).
- User reverted `classes/spatial_categorization.py`, `functions/fit_hist.py`, `prototype_flow_analysis.py` to their `1ae47b7` state.
- `325e09f` "Adopted phase 3 but need tuning" -- Phase 2 + Phase 3 kept, threshold unification dropped. Working tree clean.

Scratch (output/, not code):
- `output/test6/ana_after.txt` -- copy of `output/test7/phase3/ana_after.txt`, `dir_results` -> `output/test6`.
- `output/test6/compare_test7.py` -- compares results.db (all tables, `timestamp` skipped) + pixel-compares all PNGs vs test7.
- `output/test6/` -- full 6-recording pipeline run (415.6 s).

## Summary of current progress
- **Session 64's threshold unification is UNDONE.** Current thresholds:
  - CAT (median + per-segment reliability): baseline **mean + 1.5*std** (`spatial_categorization.compute_baseline_threshold`).
  - Spontaneous zones: `fit_hist.find_background_threshold` = **tallest-bin** peak, **1000 bins**, `ZONE_SIGMA_SEED = 0.0016`, k = 1.5 (`sp_zone_analyzer.CROSSOVER_RATIO`).
- **Post-revert run == Phase 3 reference (`output/test7/phase3/after`):** 12 MED/CAT TIFFs byte-identical, results.db `experiments` 6x35 identical, all 34 PNGs pixel-identical, stats block identical.
- **0003 reliability back to 75/82 (91.5 %)** -- the 82/82 regression came from the unified CAT threshold.
- Per-recording CAT thresholds: 0002 0.1026 / 0003 0.0992 / 0017 0.0923 / 0018 0.1310 / 0012 0.1212 / 0013 0.1074.
- Known issue carried over: tallest-bin zone threshold jumps with bin count (0012) -- the reason the unification was attempted.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Confirmed real state after the revert (Phases 2 + 3 kept, threshold change dropped)
- ✅ Re-verified Phase 3 output is identical to `output/test7/phase3/after`
- ✅ 0003 reliability regression gone (75/82)
- ✅ Committed (`325e09f`)

## What should we do next? (TODOs)
- [ ] **Check and fix ALL thresholding methods** -- spontaneous analysis, per-segment CAT, median CAT, optical flow (pre-mask). **Pre-spike (baseline) frames are dirty** -> mean + 1.5*std baselines are contaminated; decide one robust method and re-verify 0003 reliability does not jump (82/82 trap).
- [ ] Small fixes:
  - `spatial_categorization.py` docstrings say "mean + 2*std" but `BASELINE_SIGMA_MULT = 1.5`.
  - Reliability montage title shows the constant, not the k actually used.
  - Flow step is slow (50-100 s per recording; 6-rec run 415.6 s).
- Still open (not selected this time): flow interpretation (pre-mask edge pinning, 0003 sub-pixel flow, spike-1->spike = brightening), Phase 4 / TODO E, zone merge rules, formal spontaneous re-run, GUI smoke test.

## Last Session Recap
※ recap: Confirmed the revert kept Phases 2+3 and dropped the threshold unification; re-ran 6 recordings into output/test6, identical to test7 (0003 back to 75/82), committed 325e09f. Pending: fix all thresholds (dirty pre-spike frames), small fixes.

---

# Log of the project progress 2026-09-25 Fri (Session 64)
Last working file: `classes/spatial_categorization.py`
Last working line: 202 (`compute_baseline_threshold` -> now calls `fit_hist.find_background_threshold`, k = 1.5)

## List of modified files (all UNCOMMITTED)
Phase 2 (TODO C) -- reliability Vm + AP threshold:
- `functions/ap_threshold.py` (new) — `find_ap_threshold()` (first dV/dt >= 20 mV/ms before the peak, 5 ms search window), `baseline_vm()` (mean Vm -20..-5 ms).
- `classes/spike_reliability.py` — neatness refactor (step banners, logic unchanged) + new `export_vm_groups()` -> `reliability/{stem}_VM_SUCCESS_FAIL.png`.
- `functions/plot_results.py` — new `plot_vm_success_vs_failure()` (±50 ms, threshold dots, mean±SD threshold + "above baseline" line in titles).

Phase 3 (TODO D) -- ring latency removed, pre-masked flow added:
- `functions/hotspot_flow.py` (new) — `premasked_flow()`, `compute_flow_pairs()` (5 pairs spike-1->spike .. spike+3->spike+4).
- `classes/region_analyzer.py` — refactor (CONFIG + step banners); removed `compute_ring_traces`, `compute_cluster_trace`, `_peak_offset_from_spike`, `get_peak_latency_ms`, `get_temporal_traces`; kept centroid + R_lat (`cluster_radius()`); new `compute_flow()`.
- `functions/plot_results.py` — removed `plot_full_trace` + 5 ring/trace helpers; new `plot_flow_panels()` -> `flow/{stem}_FLOW.png`.
- `ach_domain_analysis.py` — `run()` split into `analyze_entry()` (1 Clip -> 2 Reliability -> 3 Median -> 4 Categorize -> 5 Region+Flow -> 6 Export); `flow/` replaces `latency/`; stats block drops `peak_latency_ms`.
- `classes/results_exporter.py`, `functions/database_ops.py` — `peak_latency_ms` no longer written/summarized (column kept as legacy).
- `functions/__init__.py` — registry: + `find_ap_threshold`, `plot_vm_success_vs_failure`, `plot_flow_panels`, `compute_flow_pairs`, `premasked_flow`; − `plot_full_trace`.

Threshold method (spontaneous zones AND CAT masks now share one function):
- `functions/fit_hist.py` — `find_background_threshold()`: center = **smoothed histogram peak** (new `smoothed_peak()`: Savitzky-Golay derivative = 0, window 5% of range), sigma = Gaussian fitted left of the peak (center pinned), **256 bins** (`ZONE_HIST_BINS`), 0.1-99.9 pct range, sigma seed from p50-p16. `fit_left_gaussian()` got an optional `center=`. `ZONE_SIGMA_SEED` removed; `N_HIST_BINS = 1000` kept for img_proc `fit_hist_sigma()` (unchanged).
- `classes/spatial_categorization.py` — `compute_baseline_threshold()` (median CAT **and** per-segment reliability check) now = `find_background_threshold(baseline, BASELINE_SIGMA_MULT=1.5)` instead of mean + 1.5*std.
- `classes/sp_zone_analyzer.py` — net unchanged (comment/log text edited and restored).
- `prototype_flow_analysis.py` — reads `output/test7/phase3/after`, uses `compute_flow_pairs`, 3-color CAT-pair background, auto arrow scale -> `output/test8/flow/optical_flow/`.

## Summary of current progress
- **Phase 2 verified:** σ=2 parity with Session 61 (0003 48/34 split identical); AP threshold dots at the upstroke knee; relative threshold ≈ +4 mV above baseline in spontaneous-like cells (0003 success +3.9 vs failure +3.8 -> threshold does not explain failures).
- **Phase 3 verified behavior-identical:** all results.db columns (except peak_latency_ms) + all MED/CAT TIFFs identical before/after refactor; flow bit-identical to the prototype. Flow step costs 50-100 s per recording (6-rec run 163 s -> 441 s).
- **Threshold story (read in order):** (1) wanted fit_hist's left-Gaussian method for CAT baseline frames -> (2) 1000 bins: float16 value grid makes a comb, "tallest bin" center jumps -> (3) 256 bins stabilised CAT -> (4) same function in spontaneous: 0012 still jumped even at 256 (flat-topped histogram) -> (5) median-pinned center fixed bin dependence but is pulled right by a bright tail (CAT 0002) -> (6) **derivative-zero (smoothed peak) center** = true noise peak, nearly bin-stable -> user chose it for both, **k = 1.5** (zones: N@2 removed too many events; N@1.5 ≈ previous zone maps).
- Production thresholds now: spontaneous 0003 0.1903 / 0018 0.1823 / 0012 0.1888; CAT baseline (6 rec) 0.0749 / 0.0915 / 0.0886 / 0.1245 / 0.1185 / 0.1050 -- verified equal to the tests.
- **Check run (0003/0012/0018, new CAT):** 0012 and 0018 fine (+3-6 % area). **0003 problem:** per-segment reliability 75/82 -> **82/82**; segments that failed with A now "pass" on scattered speckle (density gate tuned for the old threshold); median uses all 82 segments -> spike hotspot area ×2. The N@2 patched run gave 64/82 (its montage titles wrongly say 1.5σ -- title uses the constant).
- **Optical flow findings (scratch):** pre-mask pins flow to ~0 at the mask edge (0012 spike->spike+1: 0.28 px edge vs 5.1 px core); flow-then-mask has no edge gradient but moves background too -> absolute level unknown; 0003 flow is sub-pixel (arrows invisible at fixed scale); spike-1->spike "flow" is mostly brightening, not motion.
- **Zones:** near-duplicate zones found (trace-corr groups never proximity-merged; e.g. 0012 zones 1/5: 16 px, IoU 0.73); first merge rule (IoU>=0.4 or <50 px, union-find) over-merged -> parked.
- `output/` cleaned 32 GB -> ~1.3 GB (test6, reliability_group, stale test7/test8 folders deleted; test4 kept). `output/test8/README.txt` describes every remaining folder.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ TODO C (Phase 2) — Vm success/failure + AP threshold (+ relative-to-baseline line)
- ✅ TODO D (Phase 3) code — flow replaces latency, region_analyzer / ach_domain_analysis refactor, verified identical
- ✅ Threshold method unified (smoothed-peak center, 256 bins, k=1.5) for spontaneous + CAT
- ✅ output/ cleanup + test8 README

## What should we do next? (TODOs)
- [ ] **Review what files and pipelines were changed this session** (list above) before committing anything.
- [ ] **Re-check what Phase 3 really fixed** (flow vs old latency; pre-mask edge artifact; is the flow output meaningful).
- [ ] Optical flow -- next steps to be determined (candidates: noise-floor test on baseline pairs, pre-mask vs flow-then-mask, arrows inside the mask only, 5-pair parallel speed-up).
- Open problems observed, not yet prioritized: per-segment reliability k (0003 82/82 with N@1.5 -- N@2 / old A / density gate); montage title shows the constant, not the k used; bump at ~0 in spontaneous histograms; zone merge rules (parked); formal spontaneous re-run (threshold changed, PG_010 parity gone); GUI smoke test; commit.

## Last Session Recap
※ recap: Finished Phase 2 (Vm + AP threshold) and Phase 3 (flow replaces latency, verified identical), and unified the spontaneous + CAT thresholds on a smoothed-peak left-Gaussian fit (256 bins, k=1.5); all uncommitted. Pending: review changes, 0003 reliability now 100%, optical-flow direction.

---

# Log of the project progress 2026-09-24 Thu (Session 63)
Last working file: `functions/plot_results.py`
Last working line: 1203 (`plot_zone_stats` -> `panels = (...)`, frequency panel switched to log scale)

## List of modified files
- `ach_domain_analysis.py` — Phase 0: ALS is the default (`use_als=True` in `parse_ana_list()`/`run()`); CLI flag `--use_als` replaced by `--use_gauss`.
- `views/view_align_spike.py` — GUI radio button defaults to "ALS Corrected".
- `views/view_als_correct.py` — ALS lambda line edit default `100` -> `11` (matches `als_correct.py --lam` default).
- `classes/sp_zone_analyzer.py` (new) — `SpontaneousZoneAnalyzer`, port of `../PG_010/sp_ach_zones.py` (detect -> group -> map, `save()`), `timed()` sub-step timer, 0-hotspot guard, `CROSSOVER_RATIO = 1.5`, `HIGH_FREQ_FLAG_HZ = 1.0` -> `high_freq_flag` column.
- `spontaneous_analysis.py` (new) — CLI runner: proc list -> 10X-only (OBJ/SENSOR from rec_data.db) -> per-recording xlsx/tif/npz + zone-map PNGs -> `spontaneous_summary.xlsx` + `spontaneous_stats.png`, all under `<results_dir>/spontaneous/` (default `results/`). Flags: `--sigma 1.5`, `--proj`, `--color`, `--all_obj` (testing), `--debug`.
- `functions/fit_bg_hist.py` -> `functions/fit_hist.py` (git mv) — restructured into Step 1 histogram / Step 2 left-Gaussian fit / Step 3 consumers; `fit_hist_sigma()` (img_proc) unchanged (CPU bit-identical, GPU within its own 1e-11 atomic-add noise); new `find_background_threshold()` for zones (1000 bins, 0.1–99.9 pct range, out-of-range dropped) with a float16 fast path (`float16_code_counts` / `percentiles_from_counts` / `rebin_code_counts`, bit-identical to `np.percentile` path, ~10 s -> 0.3 s).
- `functions/zone_kernels.py` (new) — numba CPU + CUDA: `zone_mask()` (threshold/open/close + fill holes/small blobs, pixel-identical to scipy, 56 s -> 4.5 s) and `footprint_traces()` (114 s -> ~1 s, float16 last-digit rounding only, zones identical); float16 lookup-table trick.
- `functions/plot_results.py` — new headless figures `plot_zone_overlay`, `plot_single_zone`, `zone_colors`, `plot_zone_stats` (zone area + frequency per sensor, both log scale, fixed sensor colors validated with the dataviz validator, hollow dots = high-frequency flag).
- `functions/__init__.py`, `classes/__init__.py` — lazy-import registry updates (fit_hist, find_background_threshold, zone plots, SpontaneousZoneAnalyzer).
- `functions/detrend.py` — docstring reference `fit_bg_hist.py` -> `fit_hist.py`.
- `data/proc_20260922_000.txt`, `data/pick_20260922_000.txt` — removed red-control recordings `2025_11_08-0032` / `-0033` (tdTomato, 1202p with ABF); `Total` line corrected to the actual 201 rows (was "214" but held 203).
- `docs/knowledgebase/zone_analysis_speedup_techniques.md` (new) — speed-up write-up (learn-from-code skill).
- `.claude/plans/2026-09-24_als_spontaneous_flow_plan.md` (new) — the approved phase plan.
- **Not from this session's edits:** `classes/spatial_categorization.py` `BASELINE_SIGMA_MULT 2 -> 1.5` (user's own IDE edit; affects spike-aligned bright-pixel thresholds), `data/pick_list.json` (GUI state).

## Summary of current progress
- **Phase 0 done:** pipeline defaults to `*_BIEXP_ALS.tif` (CLI + GUI). Verified via `--help` and `parse_ana_list` resolving ALS paths.
- **Phase 1 done (spontaneous zones):**
  - Parity: at 512 bins / σ=1.5 the port reproduces PG_010 results **exactly** on all 7 PG_010 stacks (threshold, mask TIF, zone count, area, events, freq). PG_010's results turned out to have been made with `--sigma 1.5`, not the script default 2 (solved k = 1.500) -> 1.5 is now the default.
  - Switched to 1000 bins via the shared `fit_hist` kernel (user's choice). Batch A shifts: 0003 43 -> 41 zones, 0018 35 -> 40, 0012 32 -> 25 (threshold +13% on 0012) — user decided 1000 is fine, no stability check needed.
  - Speed: ~212 s -> 36–50 s per recording (fast vs slow run: masks + all 4 xlsx sheets identical); fast threshold added afterwards (bit-identical), expected ~30 s (not yet measured end-to-end).
  - `proc_20260922_000.txt` dry run: 203 rows -> 107 10X selected (GACh3.0 103, iAChSnFR 2, tdTomato 2) -> red controls removed -> 105 remain. Formal run deferred by user.
  - Frequency calculation kept as in PG_010 (user's decision). Zones > 1 Hz are flagged, not removed (user: trials contain several manually induced hotspots with higher frequencies). Single-event zones show as ~0.017 Hz (= 1/60 s recording length, a lower bound by the PG_010 convention).
- Test outputs only in `output/test6/` (17.1 GB, scratch): `spontaneous/` (512-bin parity, 7 rec), `bins1000/` (slow, 3 rec), `fast/` (latest code, 3 rec), `spontaneous/spontaneous_stats_preview.png`.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ TODO A — ALS default (CLI + GUI) + ALS lambda GUI default 11
- ✅ TODO B — sp_ach_zones merged, 10X/sensor-aware, readable outputs, 1000-bin shared histogram, numba/CUDA speed-up, high-frequency flags, stats figure
- ✅ Red-control recordings removed from the 20260922 proc/pick lists
- ✅ Knowledge-base note on the speed-up techniques

## What should we do next? (TODOs)
- [ ] Phase 2 (TODO C): merge `prototype_reliability_group_analysis.py` into `SpikeReliabilityChecker` — success/failure Vm ±50 ms PNGs in `reliability/`, AP threshold voltages (dV/dt ≥ 20 mV/ms proposed).
- [ ] Phases 3 + 4 (TODOs D + E): replace ring peak-latency with pre-masked TV-L1 flow in `RegionAnalyzer` (`flow/` replaces `latency/`), then flow metrics (speed, radial outflow, divergence, coherence).
- [ ] Delete scratch `output/test6/` (17.1 GB) once the commit is done.
- [ ] Optional: GUI entry for the spontaneous analysis (CLI only for now).

## Last Session Recap
※ recap: Made ALS the pipeline default and merged PG_010's spontaneous zone analysis into PG_005 (`sp_zone_analyzer.py` + `spontaneous_analysis.py`), verified exact parity, sped it up ~5× with numba/CUDA (identical results), added high-frequency flags and a per-sensor stats figure; formal run and Phase 2 pending.

---

# Log of the project progress 2026-09-24 Thu (Session 62)
Last working file: `prototype_flow_analysis.py`
Last working line: end of file (`run()`)

## List of modified files
- `prototype_flow_analysis.py` (new, final/kept) — TV-L1 optical flow on production blurred MED/CAT (`output/test4/`), pre-masked (mask applied to the input frames BEFORE calling `optical_flow_tvl1`, not after), masked display only, 5 panels per recording: `spike-1->spike`, `spike->spike+1`, ... `spike+3->spike+4`. Output: `output/test5/optical_flow/{tag}_FLOW_PREMASKED.png`, one per recording, all 6 regenerated and confirmed working.
- Five earlier scratch scripts from this session (`prototype_noblur_optical_flow.py`, `prototype_export_masked_med.py`, `prototype_compare_cat_masks.py`, `prototype_noblur_vs_blurred_batch.py`, `prototype_premask_optical_flow.py`) were written, used to reach the conclusions below, then **deleted** at the user's explicit request once scope narrowed — see "how we got here" below. Only `prototype_flow_analysis.py` remains.
- `output/test5/` cleaned up from 29GB down to 9.1MB — removed all no-blur detrend/ALS/median/CAT tiffs, the old masked/unmasked/cumulative/blurred-vs-noblur flow PNGs, the CAT-mask comparison PNGs, and the `spikes/` side-export folder. Only the 6 final `*_FLOW_PREMASKED.png` files remain.
- None of today's other tracked-file changes (ALS `LAM` tuning, `prototype_med_height_map.py` extensions) were made this session -- those were already committed by the user (`ee08289`, `21954b9`) before this session's work began.

## Summary of current progress (how we got here -- read in order, conclusions were revised twice)
1. **Motivating question**: does removing the Gaussian blur (a smooth, textureless blob has no interior structure for TV-L1 optical flow to lock onto -- the root cause of every previously-invalidated PIV/optical-flow attempt) let optical flow say something useful about post-spike spatial dynamics?
2. **No-blur reprocessing built and compared against blurred, all 6 recordings** (now superseded, see #4): no-blur CAT masks looked far more spatially localized than the blurred production CAT (which over-thresholds 50-90% of the frame on 4/6 recordings -- the known density/threshold-calibration TODO since Session 57, now shown clearly side-by-side). No-blur optical flow also looked more spatially varied/plausible than blurred flow, which showed either a suspiciously uniform "clean" field or near-zero flow. **User's initial verdict at this point: no-blur works, merge it later.**
3. **User then asked a sharper question**: was the mask applied before or after computing flow? Answer: after -- flow was always computed on the FULL frame, and CAT only filtered which arrows got drawn. Since TV-L1 is globally regularized, background pixels outside the nominal mask can still pull the "masked" region's flow estimate.
4. **Pre-masking test (mask applied to input frames before `optical_flow_tvl1`, not after) overturned part of #2**: on `2025_12_15-0012`/`-0013` blurred data, the previously "clean" spike->spike+1 flow collapsed to near-zero once background was floored out first -- confirming that result was mostly background leakage, not real signal. The one pattern that *survived* pre-masking was a symmetric bilateral divergent flow at spike-1->spike (arrows pointing away from the centerline on both sides).
5. **User's final scope call**: no-blur reprocessing, cumulative (direct spike->spike+4) flow, and unmasked display are all dropped. Kept: blurred (production) data only, pre-masked flow, masked-only display, per-step pairs including the newly-added pre-spike pair. This is what `prototype_flow_analysis.py` implements, confirmed running clean on all 6 recordings.
6. **Process note**: a mid-session scare where `output/test5`'s entire prior contents vanished turned out to be the user manually deleting them, not a bug.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Explored no-blur reprocessing as a way around TV-L1's blurred-blob limitation, then correctly abandoned it once pre-masking showed the real fix was in how masking interacts with flow computation, not in the input blur itself
- ✅ Diagnosed and fixed a real methodological issue: masking after flow computation lets background influence the "masked" result; masking before computation doesn't
- ✅ Landed on a final, narrowed analysis: blurred data, pre-masked, masked-display, per-step + pre-spike pairs -- confirmed across all 6 recordings
- ✅ Consolidated 5 scratch scripts down to one (`prototype_flow_analysis.py`) and deleted the superseded ones at the user's request
- ✅ Cleaned up `output/test5` (29GB -> 9.1MB), keeping only the final 6 result PNGs

## What should we do next? (TODOs)
- None from this session -- user explicitly decided not to pursue the pre-spike pattern interpretation, the pipeline-merge decision, or the CAT-threshold recalibration as follow-ups.

## Last Session Recap
※ recap: Explored no-blur reprocessing to fix TV-L1 optical flow's blurred-blob problem, initially looked promising across all 6 recordings, but a follow-up test (masking before vs after flow computation) showed much of that "clean" result was background leakage from not masking early enough -- corrected by pre-masking, which collapsed most per-step flow to near-zero but left one surviving pattern (symmetric divergent flow at spike-1->spike) unexplained. Final scope narrowed to blurred-only, pre-masked, masked-display, per-step+pre-spike pairs, consolidated into a single script (`prototype_flow_analysis.py`) after deleting 5 superseded scratch scripts; `output/test5` cleaned from 29GB to 9.1MB. User decided not to pursue this line further (no interpretation, no pipeline merge, no threshold recalibration) -- this optical-flow investigation is closed.

---

# Log of the project progress 2026-09-22 Tue (Session 61)
Last working file: `prototype_reliability_group_analysis.py`
Last working line: end of file (`print(f"  Saved: ...")`)

## List of modified files
- `prototype_reliability_group_analysis.py` — iterated at user request across the session:
  - Monkey-patched `AbfClip._export_spike_plot` to a no-op before the loop, so the automatic `results_dir/spikes/*.png` export is skipped (prototype-only patch; `classes/abf_clip.py` itself untouched).
  - Commented out the B% histogram block (`bp_detected` is still collected in case it's restored later).
  - Dropped per-recording subfolders — output now goes flat into `output/reliability_group/` as `<recording_stem>_vm_success_vs_failure.png`.
  - Success/failure Vm axes: `grid(True)` added, x-limits iterated 200 → 100 → final **±50 ms**.
- `docs/continue_from_here.md` — this log entry.

## Summary of current progress
- **Todo "reprocess all selected raw TIFFs" is done** — all recordings reprocessed to both GAUSS and ALS this session (earlier attempts had been blocked mid-session waiting on the GAUSS pass to finish).
- Regenerated all 6 recordings' `vm_success_vs_failure.png` against the freshly reprocessed GAUSS TIFFs — flat into `output/reliability_group/`, ±50 ms window, grid on, no B% histogram, no spikes/ subfolder.
- This zoomed-in success-vs-failure Vm view also satisfies the **"zoom in to see the details of successful and failed spikes"** todo.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Recorded and refined the 8-item project TODO list in this log (Matityahu/DA wave check, reprocess TIFFs, separate by objective, median vs. segment comparison, induced vs. spontaneous reliability, success/failure spike zoom-in, Astra/Claude rosemap locality argument, dataset sort for Jeff)
- ✅ Reprocessed all selected raw TIFFs to both GAUSS and ALS
- ✅ Updated `prototype_reliability_group_analysis.py` (skip spikes/ export, bypass B% histogram, flat output, ±50ms Vm x-limits, grid on) — ruff clean
- ✅ Regenerated all 6 `vm_success_vs_failure.png` outputs into `output/reliability_group/`
- ✅ Zoom in on successful and failed spikes (via the ±50ms Vm success/failure plots above)

## What should we do next? (TODOs)
- [ ] Large spacing and multiple/different frequencies of spontaneous ACh hotspots may suggest ACh propagation is not wavelike but a series of adjacent hotspots occurring — need to double check against Matityahu and the original DA wave paper.
- [ ] Separate results based on objective.
- [ ] Compare the difference between the median and its segments.
- [ ] Discuss the reliability between induced and spontaneous.
- [ ] Think about Astra's model analysis and Claude's rosemap for the argument that major hotspots are not "diffusively" transmitted across large space but are local and restricted.
- [ ] Properly sort and mark the dataset used, then put it in the bucket for Jeff.

## Last Session Recap
※ recap: Finished reprocessing all raw TIFFs to GAUSS and ALS. Iterated `prototype_reliability_group_analysis.py` (skip spikes/ export, bypass B% histogram, flat output layout, ±50ms Vm window with grid) and regenerated all 6 recordings' success-vs-failure Vm plots into `output/reliability_group/` — this also covers the "zoom in on successful/failed spikes" todo.

---

# Log of the project progress 2026-09-17 Thu (Session 60)
Last working file: `views/view_align_spike.py`
Last working line: 120 (`self.le_status.setReadOnly(True)`)

## List of modified files
- `functions/gaussian_blur.py` — real bug fix: both `_cpu_conv` and `_gpu_conv` used to under-smooth pixels near image edges/corners (CPU skipped out-of-bounds kernel taps and renormalized over fewer samples; GPU clamped to the edge pixel). Replaced both with reflect padding (mirrors the image across its own edge), verified bit-exact against `scipy.ndimage.gaussian_filter(mode="reflect")` (max diff ~1.5e-8). User regenerated all 6 recordings' proc TIFFs with the fix afterward (confirmed via file timestamps + a before/after corner-noise check: 0.375 → 0.134, much closer to the ~0.10-0.12 interior baseline).
- `prototype_med_height_map.py` (new) — standalone 3D "intensity height-map" (surface plot) of spike→spike+4 MED frames, modeled after `prototype_rose_map_batch.py`'s pattern. Iterated through: CAT-mask floor (non-bright pixels floored to vmin — this is what actually cleaned up the background speckle), a per-pixel baseline z-score attempt (`zscore_by_baseline`, tried per user request referencing the old removed `zscore_img_segs.py` — rejected, no denoising benefit, amplified some corner spikes), then switched from recomputing everything via `AbfClip` to loading the pipeline's own already-exported MED/CAT tiffs directly from `output/test4/`. Final state: raw MED intensity + CAT mask, `use_mask=True`, `RECORDINGS` covers only `2025_06_11-0002`/`-0003`.
- `classes/spatial_categorization.py` — added a `BASELINE_WINDOW_FRAMES=5` cap on `_calculate_global_threshold`'s baseline window, then reverted it later in the same session per explicit user request. File confirmed back to its exact pre-session state via `git diff` (no residual changes).
- `views/view_align_spike.py` — real bug fix: `le_current_total`/`le_status` in the Align Spike popup window were editable `QLineEdit`s (missing `setReadOnly(True)`), unlike every other display-only line edit in the project (`view_img_proc.py`, `view_als_correct.py`, `view_dor_query.py` all set it). Added `setReadOnly(True)` to both.
- Everything above (plus other untracked prototype files already present at session start) was committed by the user as `b6266dc "fix some bugs"`.

## Summary of current progress
- **FlowRegSuite/pyflowreg investigated and closed as a dead end**: its motion-detection core is the same TV-L1-family variational optical flow already ruled out earlier (assumes conserved texture/brightness-constancy; MED's spike-aligned signal grows/spreads without conserved texture — same failure mode as the already-tested `skimage.optical_flow_tvl1` shrink-only control test).
- **3D MED height-map visualization built and iterated** (see `prototype_med_height_map.py` above) — settled on raw MED intensity + CAT-mask floor as the cleanest rendering; z-score normalization tried and rejected.
- **Found and fixed the Gaussian-blur edge-noise bug** (see modified files above) — this was the root cause of "sharp, unsmoothed edges" the user noticed in the height-map plots. Fix verified both synthetically (vs scipy) and against real proc-tiff corner-noise measurements before and after.
- **Audited `AbfClip` spike-to-frame and segment alignment end-to-end**: verified `spike_frame_indices_raw` math against real ABF peak times by hand for both a `diff=1` recording (`2025_06_11-0002`) and a `diff=2` recording (`2025_12_15-0012`/`2025_12_15_0008.abf`) — exact match both times. Checked all 8 segments of the `diff=2` recording for length/centering/overlap consistency — all clean. Found `diff=2` occurs in 4 of the 6 recordings (not just 1 as first assumed); user confirmed this matches their own manual raw-TIFF trimming convention (1 frame off the front for `diff=1`, 1 off each end for `diff=2`), which is exactly what `abf_clip.py`'s hardcoded `first_dropped=1` assumption already encodes. No alignment bug found — `last_dropped` remains computed-but-unused, confirmed harmless for this dataset.
- **Fixed the Align Spike popup's read-only display fields** (see modified files above).

## Completed TODOs/Tasks (before new wrap-up)
- N/A — no open TODO list carried into this session; all work originated from today's conversation.

## What should we do next? (TODOs)
- None — user explicitly confirmed nothing needs to carry over (FlowRegSuite line closed, categorizer change reverted, blur fix already regenerated by user, alignment audit resolved with no bug found).

## Last Session Recap
※ recap: Fixed a real Gaussian-blur edge-noise bug (reflect padding, verified vs scipy) and user regenerated all proc TIFFs. Audited AbfClip spike/frame alignment — confirmed correct. Reverted a categorizer experiment. Fixed read-only line-edit bug in Align Spike popup. Nothing pending.

---

# Log of the project progress 2026-09-16 Wed (Session 59)
Last working file: `prototype_front_velocity.py` (scratch-only session, no pipeline files touched)
Last working line: end of file (`run()`)

## List of modified files
- None in `classes/`/`functions/`/`ach_domain_analysis.py`. Entire session was scratch prototyping in the repo root (all new, untracked `prototype_*.py` files) plus this log:
  - `prototype_reliability_group_analysis.py`, `prototype_rose_map_batch.py` — copied in from prior sessions' OS-temp scratchpads, per user request, so they survive in the repo instead of a temp dir.
  - `prototype_piv.py`, `prototype_piv_sanity_check.py`, `prototype_piv_sanity_check2.py`, `prototype_piv_streamline.py`, `prototype_piv_alt_methods.py`, `prototype_diff_maps.py`, `prototype_front_velocity.py` — new this session (PIV exploration + alternatives).

## Summary of current progress
- **PI's request**: group spikes into detected ("success") vs. failure segments and compare Vm shape and spatial (B%) density. Built `prototype_reliability_group_analysis.py`: confirmed `seg_results`/`df_picked_spikes`/`clip.get_vm_segments()` all share the same per-segment order by construction, so grouping is a clean index-match. Iterated through user feedback: pooled → per-recording folders (`output/reliability_group/<recording>/`) → peak-aligned Vm traces (each trace's own spike peak at t=0, via `argmax(vm)`). Final result across all 6 recordings in `ana_20260915_000.txt`: reliability ranged 56.1%–100%, B% histogram of detected segments heavily skewed toward 0–0.25, and peak-aligned success/failure Vm shapes look visually indistinguishable (FWHM/shape comparison itself deferred — user said "later").
- **PIV exploration on `2025_06_11-0002`** (spike..spike+4 MED frames): built a windowed cross-correlation PIV prototype (`prototype_piv.py`). Result was chaotic/noisy regardless of masking strategy (raw CAT bright mask, density-gated hotspot mask, or MED×CAT pixel-masking — exported as `output/piv/med_and_cat_0002.tif` at the user's request). Diagnosed two real, compounding causes: (1) the CAT/density-gate threshold is far too permissive on this recording (47–77% of the frame reads "bright" at spike+0/+1/+2 — the same open Session-57 calibration TODO), so masking barely restricted anything; (2) even where correctly masked, a smooth Gaussian-blurred blob's interior has no texture for cross-correlation to lock onto (the aperture problem).
- **Validated the PIV code itself is correct** using two real seeded-particle benchmark datasets the user provided (`output/piv/test_image_set/`, `output/piv/test_image_set_2/`): `test_image_set` gave a highly consistent 0.100 px/pair displacement across all 199 consecutive frame pairs (a synthetic subpixel-accuracy calibration set); `test_image_set_2` (real bubble/particle images) gave a genuine, physically-plausible recirculation pattern once windowed small enough (20px) and cleaned with a median-filter outlier pass + `plt.streamplot` (`output/piv/streamline_wedge_zoom.png`) — user's initial reaction to the raw quiver version was "are you crazy... I would rather ask you plot streamline instead of this."
- **Found and fixed a real quiver sign bug**: every quiver call this session (carried over incorrectly from last session's rose-map polar-plot convention) passed `V = -dv_row` instead of `V = dv_row` on `imshow`+`quiver` axes, vertically mirroring every flow field shown. Caught by the user directly comparing `test_image_set_2`'s quiver against the image's visually-obvious up-left flow. Verified the correct sign empirically (a plain `V=+30` test) before fixing all three affected scripts and regenerating every plot.
- At the user's explicit redirect ("I have a feeling now you kind of being restricted by the idea of PIV... what's the main purpose... do you have any solutions?"), stepped back from PIV specifically and tried two texture-free alternatives on `2025_06_11-0002`, per user request, exported to their own `output/` subfolders:
  - **`prototype_diff_maps.py`** → `output/diff_maps/diffmap_0002.png`: plain `frame[n+1]-frame[n]` diverging heatmaps for both MED and CAT. Directly showed a retreating margin moving frame to frame (CAT: +250,235px gained at spike+0→+1, then −174,999/−321,471/−224,100px lost over the next three pairs) — no motion/correspondence assumption, so nothing to fabricate.
  - **`prototype_front_velocity.py`** → `output/front_velocity/frontvel_0002.png`: front-normal velocity via the level-set relation `v = -(dI/dt)*grad(I)/|grad(I)|^2`, masked to only trust blocks with sufficient local gradient (so it honestly reports "no information" in the flat interior instead of fabricating a vector, unlike optical flow — see below). spike+0→+1 showed a real, coherent fan-like pattern; later pairs (fewer bright pixels remaining) were much noisier.
  - **Optical flow (TV-L1) was tried and explicitly invalidated**: it initially looked deceptively clean (a single uniform downward direction across the whole masked region), but a synthetic control test (a blob that only *shrinks* in place, zero true motion) produced a spurious median "speed" of 22.9px from `optical_flow_tvl1` — proving it doesn't escape the shrink-vs-move confound, it just hides it behind a smooth-looking field. This result was not trusted/reported as a finding.
  - **Gradient-magnitude PIV** (Sobel-filtered images fed into the same windowed correlation) was also tried at two window sizes (15µm, 45µm) — stayed chaotic at both, with the larger window actually showing *higher* median speed (12-15 vs 5.8-6 µm/pair), which is backwards from what real signal averaging would do.
- User's verdict on all of the above: "still not what I want... tired." Session paused there (interrupted further by an unplanned Windows update restart, but no work was lost — verified via `git status` and `ls output/` after reconnecting).
- **Discovered mid-session** (before the restart) two unfamiliar, more-thorough output folders (`output/spatial_change_preview_20260916_023908/`, `output/spatial_change_refined_20260916_025431/`) covering all 6 recordings with forward/backward-consistency-checked optical flow, explicit signal-support masking, and stated caveats in `HOW_TO_READ.txt`. Flagged directly to the user rather than assumed — confirmed by the user afterward: **this is separate, parallel work they're doing with another AI tool ("Astra"/ChatGPT) on the same problem**, not something to merge into or take credit for. User explicitly said they're continuing that exploration elsewhere and will discuss with this session later.
- Cleaned up `output/piv/` at the user's request (removed pre-sign-fix stale files and debug-only previews, renamed quiver outputs to simple `quiver_*.png` names) after being told directly: "why every time you just don't show where the fuck you export the file and I have to keep asking" — saved as a standing memory rule (`feedback_always_state_export_paths.md`) to always state full export paths immediately, unprompted, going forward.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Built and delivered the PI-requested reliability-group analysis (detected vs. failure Vm traces + B% histogram), per-recording, peak-aligned, across all 6 recordings in `ana_20260915_000.txt`
- ✅ Exhaustively tested PIV (raw MED, CAT-masked, gradient-magnitude, multiple window sizes) on `2025_06_11-0002` — confirmed not viable for this data (no texture)
- ✅ Validated the PIV/quiver code itself is correct against two independent real benchmark datasets, including a proper streamline-based cleanup of a noisy-but-real result
- ✅ Found, root-caused, and fixed a real quiver vertical-sign bug across all affected scripts and outputs
- ✅ Tried and honestly evaluated two texture-free alternatives (diff maps, front-normal velocity) per the user's explicit redirect away from PIV-specific thinking
- ✅ Caught and invalidated a deceptively-clean optical-flow result via a synthetic shrink-only control test, rather than reporting it as a false win
- ✅ Copied two scratch prototype scripts into the repo root (from prior sessions' OS-temp scratchpads) at the user's request, so they persist
- ✅ Cleaned up `output/piv/` naming/clutter per user request

## What should we do next? (TODOs)
- [ ] **Spatial-change method for spike..spike+4 is still unresolved.** Neither PIV/optical-flow/gradient-PIV nor diff-maps/front-velocity fully satisfied the user. The user is now exploring this in parallel with another AI tool (ChatGPT/"Astra") — next session, check in on what that produced (`output/spatial_change_refined_20260916_025431/` and `output/spatial_change_preview_20260916_023908/` are its outputs) before trying anything further from scratch.
- [ ] **Decide rose-map's fate.** Paused mid-exploration in Session 58 ("forget about it temporarily" this session) — still an open question whether/how it fits into the final spatial-change approach, especially given its known bipolar-elongation blind spot (flagged Session 58).

## Last Session Recap
※ recap: Explored PIV (and found/fixed a real quiver sign bug) plus two texture-free alternatives (diff maps, front-normal velocity) for characterizing spike..spike+4 spatial changes on `2025_06_11-0002` — none fully satisfied the user, who is now exploring the same problem in parallel with another AI tool. Also delivered the PI-requested detected-vs-failure Vm/B% reliability comparison across all 6 recordings. No production code touched; all work in new `prototype_*.py` scratch scripts.

---

# Log of the project progress 2026-09-15 Tue (Session 58)
Last working file: (none — scratch-only session, no pipeline files touched)
Last working line: n/a

## List of modified files
- None. This entire session was exploratory prototyping in the scratchpad (`rose_map_prototype.py`, `rose_map_batch.py`) — no `classes/`/`functions/`/`ach_domain_analysis.py` files were edited.

## Summary of current progress
- User asked how to better visualize/quantify whether a hotspot's bright pixels show directional spreading, as a replacement/addition to `RegionAnalyzer`'s current inner/outer-ring peak-latency metric (`region_analyzer.py:313-384`, `compute_ring_traces`) — flagged as weak: only 2 radial bins, assumes radial symmetry, not physically grounded, and multi-cluster mode measures something different (asynchrony) than single-cluster mode (spreading).
- Iterated through several proposed approaches with the user before landing on one:
  - Centroid-trajectory tracking — user correctly pointed out this confounds real movement with the mask simply shrinking/growing asymmetrically (a shrink toward one side looks identical to movement).
  - Fixed-origin "new pixels only" front-tracking — addressed the shrink confound but the user proposed something better.
  - **User's own idea** (adopted): fix the origin at the spike-frame bright-mask centroid, then for spike..spike+N compute an 8-direction (compass) normalized radial histogram ("rose map") of the bright mask at each frame, plus an anisotropy score. This separates "how much" (total bright count) from "which way" (share per direction) by construction, so shrink/grow alone doesn't fake a directional signal.
- Built a scratch prototype (`rose_map_prototype.py`, later generalized to `rose_map_batch.py`) computing, per frame from spike to spike+4, on the same spike-centered MED stack `RegionAnalyzer` already uses (confirmed explicitly at the user's check "this should be done on MED?"):
  - 8-sector share histogram anchored at the fixed spike-frame centroid.
  - Anisotropy `R` = resultant-vector length of the 8 shares (0 = isotropic, 1 = fully one-sided).
  - 3-row figure: frame panels (bright mask + origin marker) / rose maps / per-direction share vs. frame line plot.
- **Caught and fixed two real bugs during prototyping, both before trusting any result:**
  - Angle sign: `atan2(dr, dc)` used image-row-down as positive, but polar plots draw "up" as positive — the whole rose map was vertically flipped relative to the actual image. Fixed with `atan2(-dr, dc)`.
  - Sector/label misalignment: `np.histogram` bins from `linspace(-180, 180, 9)` don't center on the 8 compass directions (first bin centered at -157.5°, not -22.5°/E) — the printed "S"/"SW" labels were actually reporting the N/NW bins' data. Fixed by defining bin edges offset -22.5° from each compass-direction center, with an explicit angle-wrapping step. Caught by inspecting the rendered rose-map wedge positions against the printed shares and noticing they didn't match.
- Ran the corrected prototype on 4 recordings (`2025_06_11-0002` 60X, `2025_06_11-0003` 10X, `2025_11_13-0017` 40X, `2025_12_15-0012` 10X) — results are genuinely recording-dependent, not a uniform pattern:
  - `0002`: clean, monotonic, real signal — `R` climbs 0.01→0.66 as the blob shrinks and retreats toward N/NW from its spike-frame centroid.
  - `0003`: noisy, non-monotonic (`R` bounces 0.10→0.24→0.02→0.22→0.40) — mostly small edge-fragment noise, not a clean trend.
  - `0017`: dominated by the static-structure issue from this session's earlier z-score removal — 60-80% of the frame is "bright," `R` stays low (0.01-0.14) throughout; not much real hotspot signal to extract direction from.
  - `0012`: revealed a genuine **blind spot in the `R` metric** — the mask is elongated N-S from the start and stays elongated (a "bowtie": N and S shares both grow together, 24%→37% and 23%→31%, while other directions shrink), but `R` stays near 0 (0.00→0.09) because N and S are diametrically opposite and cancel out in the resultant-vector sum. `R` only detects one-sided (unipolar) bias, not symmetric two-sided (bipolar) elongation.
- All output is scratch-only, saved for reference: `output/test2/rose_map_prototype_0002.png`, `_0003.png`, `_0017.png`, `_0012.png`, plus the two scripts (`rose_map_prototype.py`, `rose_map_batch.py`) in this session's scratchpad directory:
  `C:\Users\Kang\AppData\Local\Temp\claude\D--Programs-PG-005\5894e6bd-ab69-4f03-9cf9-5bbc33b3c812\scratchpad\`
  **Caveat**: this is an OS temp directory tied to this specific session, not part of the repo — it is not guaranteed to persist. If it's gone next session, `rose_map_batch.py` (the more complete, generalized version) can be regenerated from the PNGs' methodology described above (fixed spike-frame centroid origin, 8-sector compass-aligned histogram via `SECTOR_EDGES = np.deg2rad(np.arange(-22.5, 360, 45))`, resultant-vector `R`, `atan2(-dr, dc)` for correct on-screen orientation) rather than re-derived from scratch.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Explored and prototyped a directional-spreading visualization/metric at the user's request — landed on the user's own fixed-origin rose-map idea, validated (with two bugs caught and fixed) against 4 real recordings

## What should we do next? (TODOs)
- Not yet wired into the real pipeline — this was explicitly kept as scratch-only prototyping. Before any `region_analyzer.py`/`plot_results.py` change, still need to decide:
  - Add a second "elongation" statistic (e.g. fit shares as a 2nd-moment ellipse, or compute R per opposite-direction pair) to catch the `0012`-style bipolar pattern that plain `R` misses.
  - Possibly test more sectors (e.g. 16) and/or more recordings to see how robust the `0002`-style unipolar pattern really is.
  - The `0017` result is a reminder that the density_thresh/window_px re-tuning TODO from Session 57 (for raw-intensity static structure) is still open and may affect how trustworthy any of these direction numbers are.
- User's original "modify the latency analysis" task is still open-ended beyond this rose-map exploration — no decision yet on whether rose-map replaces or supplements the existing `get_peak_latency_ms()`/ring-trace metric.

## Last Session Recap
※ recap: User wanted a better way to check whether a hotspot's bright pixels show directional spreading, replacing/supplementing the current 2-ring peak-latency metric. After iterating through a few proposals (centroid tracking had a shrink/grow confound the user caught themselves), the user proposed a fixed-origin, 8-direction rose-map + anisotropy score, which we prototyped in the scratchpad against 4 real recordings (no pipeline files touched). Caught and fixed two real bugs along the way (a vertical-flip angle-sign error, and a sector-label/bin misalignment) before trusting any result. Final finding: the method genuinely works and shows a clean monotonic directional signal on `2025_06_11-0002`, but testing on 3 more recordings surfaced real recording-dependent variation, including a genuine blind spot in the anisotropy score itself — it misses symmetric "bowtie" (bipolar) elongation like `2025_12_15-0012` shows, since opposite-direction shares cancel out in the resultant-vector sum. Nothing wired into the real pipeline yet; next step (not yet started) is likely adding a second elongation-aware statistic before deciding how/whether to integrate this into `region_analyzer.py`.

---

# Log of the project progress 2026-09-15 Tue (Session 57)
Last working file: `functions/database_ops.py`
Last working line: `compute_region_stats()`, metric-column Float64 cast

## List of modified files
- `functions/zscore_img_segs.py` → renamed `functions/load_img_segs.py` — removed the per-segment z-score numba kernel (`_cpu_zscore_segment`) entirely; the function now just reads raw (detrended, unnormalized) frames off the proc TIFF. Per the user's explicit request to stop re-baselining reliability/median/categorization against each segment's own z-score.
- `ach_domain_analysis.py`, `classes/spike_reliability.py` — updated to call `load_img_segs`/use `lst_segments` throughout; reliability check and the final median/categorization now both run on the same raw values.
- `functions/spike_alignment.py` — `zscore_range` → `intensity_range` in `spike_centered_median`/`spike_centered_avg` (math was already unit-agnostic, pure rename).
- `classes/results_exporter.py` — DB columns `zscore_min`/`zscore_max` → `intensity_min`/`intensity_max`, added non-destructively via `_ensure_columns` (old DBs keep their orphaned `zscore_min`/`zscore_max` columns, harmless).
- `functions/plot_results.py`, `classes/region_analyzer.py` — updated plot labels/titles/docstrings that would otherwise be actively wrong now that values aren't z-scored (`"Mean z-score"` → `"Mean intensity"`, `"Ring z-score traces"` → `"Ring intensity traces"`, etc.).
- `export_zscore_segments.py`, `export_als_segments.py` — updated to call `load_img_segs` instead of the removed `zscore_img_segs`.
- `functions/fit_bg_hist.py` — removed `cache=True` from `_cpu_histogram_counts`/`_cpu_masked_std`: both call `numba.get_num_threads()`/`get_thread_id()` (dynamic globals), so caching was always silently ineffective and only produced a `NumbaWarning` every run. Confirmed via `ruff` + real GPU/CPU pipeline run that removing it changes nothing functionally.
- `functions/database_ops.py` — **real bug fix**: `compute_region_stats()` crashed with `InvalidOperationError: quantile operation not supported for dtype null` whenever a metric column (e.g. `lasting_time_ms`) was entirely NULL in the DB — `pl.read_database` infers such a column as polars dtype `Null`, and `.quantile()` (unlike `.mean()`/`.std()`) refuses to run on that dtype. Fixed by casting all four `metric_cols` to `Float64` right after reading, before any aggregation. Verified with a synthetic all-null `lasting_time_ms` row: previously crashed, now returns clean `None`s for every stat on that metric.

## Summary of current progress
- Investigated (no code change) what the reliability montage's background mask actually is: `SpikeReliabilityChecker.check()` categorizes the spike/spike+1 frame fresh from the raw segment each time (same method as CAT.tif generation), not a pre-existing CAT file.
- Investigated (no code change) the density-gating mechanism in detail: `uniform_filter(bright_mask, window_px)` computes local density *from the categorized mask itself*, not an independently-derived signal — `hotspot_mask = bright_mask & (density >= density_thresh)`.
- Confirmed `SpatialCategorizer.compute_baseline_threshold()` is generic (`mean + BASELINE_SIGMA_MULT * std` over whatever baseline frames are passed) — works identically on raw or z-scored input, which made the z-score removal safe.
- At the user's request, removed z-scoring from the entire spike-aligned pipeline (reliability check + median + categorization + region analysis) — see modified files above. Verified end-to-end on `2025_06_11-0003`: 18/82 (22.0%) reliability, cluster detected, decay tau fit succeeded (86ms, R²=0.97), new `intensity_min`/`intensity_max` DB columns populated correctly.
- Flagged an important side-effect the user didn't anticipate: the proc TIFF (`*_BIEXP_GAUSS.tif`) is already z-scored at an *earlier, separate* preprocessing stage (`img_proc.py`'s `img_zscore_convert`, stack-wide background-noise normalization) — untouched by this session's change. "Raw segments" now means background-normalized but no longer per-segment/per-pixel re-baselined, not fully raw camera counts.
- Diagnosed (no code change) a follow-up visual confusion from the same change: with per-segment z-scoring removed, static bright structure (fixed hot pixels, autofluorescence) that used to be erased by the old *per-pixel* baseline correction now clears the single global `mean + Nσ` threshold everywhere, producing scattered "bright" specks in the montage background that the density gate correctly excludes from any cluster. Ran a live diagnostic on `2025_06_11-0003` seg37: 15,330 bright pixels across 101 raw blobs, only 615 survive the local-density gate and merge into 1 accepted cluster — every other blob (including an 1179px one) has zero pixels locally dense enough to pass, which is why visually similar-sized blobs can have different accepted/rejected status (acceptance depends on neighborhood bright-pixel density, not the blob's own size).
- Confirmed (via code read, no change) that the spike-centered median already only uses reliability-detected segments (`ach_domain_analysis.py:380-382`, falls back to all segments only when 0 detected, with `final_significant` forcing that case to "no detection" regardless).
- One-off scratch script (not committed, not part of the pipeline): exported per-segment spike/spike+1 categorized-mask TIFFs for `2025_06_11-0002` to `output/test2/cat_check/` for manual inspection, per user request.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Removed z-scoring from the spike-aligned pipeline (reliability, median, categorization, region analysis) — renamed `zscore_img_segs`→`load_img_segs`, `zscore_range`→`intensity_range`, DB columns `zscore_min/max`→`intensity_min/max`, updated all affected plot labels/docstrings
- ✅ Verified the no-zscore pipeline end-to-end on real data (`2025_06_11-0003`)
- ✅ Removed the ineffective `cache=True` from `fit_bg_hist.py`'s two CPU-jitted functions
- ✅ Fixed the real `compute_region_stats()` null-quantile crash (Float64 cast on metric columns), verified with a synthetic all-null test case
- ✅ Confirmed spike-centered median already uses only reliability-detected segments (no change needed)
- ✅ Explained/diagnosed (no code changes) the reliability montage's background-mask semantics and the density-gate's blob-acceptance behavior, at the user's request

## What should we do next? (TODOs)
- Re-tune `density_thresh`/`window_px` per objective for raw intensities — these were tuned back when segments were z-scored (much less static-structure speckle in the bright mask); the user acknowledged "the density threshold gave me a hard time" but said reliability is acceptable for now. Revisit only if it comes up again.
- User's stated next task: modify the **latency analysis** (not yet scoped/discussed this session).

## Last Session Recap
※ recap: At the user's request, removed z-scoring entirely from the spike-aligned pipeline (reliability check, median, categorization, region analysis) — the proc TIFF already carries an earlier, separate stack-wide z-score normalization from `img_proc.py`, so this only removed the *per-segment, per-pixel* re-baselining step. Verified end-to-end on real data. Spent significant time explaining/diagnosing (no code changes) exactly what the reliability montage displays and why the density-gated hotspot detector accepts/rejects visually-similar-looking bright blobs (acceptance depends on local neighborhood density, not blob size) — this surfaced as user confusion after the z-score removal made static bright structure newly visible in the montage background. Confirmed the spike-centered median already correctly uses only reliability-detected segments. Also fixed two smaller, previously-flagged items at the user's request: removed an ineffective `cache=True` from `fit_bg_hist.py`, and fixed a real `compute_region_stats()` crash on all-null metric columns (polars `Null`-dtype quantile). Next up: latency analysis (not yet scoped).

---

# Log of the project progress 2026-09-15 Tue (Session 56)
Last working file: `classes/spike_reliability.py`
Last working line: end of file (`SpikeReliabilityChecker.export_montage`)

## List of modified files
- `classes/spatial_categorization.py` — extracted the hardcoded `2` in `compute_baseline_threshold()` to a named constant `BASELINE_SIGMA_MULT` (user later changed its value from `2.0` to `3.0` directly in the IDE, outside this session's own edits).
- `classes/abf_clip.py` — two real fixes:
  - Replaced the old 4-CSV spike export (`_export_spike_csv`) with a single PNG (`_export_spike_plot`) showing the full Vm trace with picked/skipped/collapsed spikes color-marked. Added `Peak_Time`/`Peak_Value` columns to `df_picked_spikes`/`df_skipped_spikes` so the plot can place markers.
  - **Bug fix**: `set_interval_frames` was picked via the *mode* of `all_min_available`, silently dropping any spike whose own margin fell short of that value (52% of spikes dropped on a real test recording). Replaced with the 20th-percentile of `all_min_available` (new `KEEP_FRACTION_QUANTILE = 0.2` constant), which guarantees ≥80% of spikes clear the bar by construction — verified drop rate fell from 52% to 18% on `2025_06_11-0003`. Also extracted the old hardcoded `20` cap to `MAX_SET_INTERVAL_FRAMES`.
- `ach_domain_analysis.py` — multiple structural cleanups, all confirmed with the user before applying:
  - Renamed `ref_df`/`cell_df`/`save_thread` → `df_checked_tiff`/`df_cell_group`/`figure_export_thread`; added blank-line separation to the previously dense setup block (~L335-369 pre-session numbering).
  - Renamed `animal_index_map` → `animal_idx_lut`, and its builder `ResultsExporter.build_animal_idx_lut()` (was `build_animal_index_map`).
  - Added clarifying comments around the `AbfClip` construction + no-segments guard block, and around the background-figure-export-thread throttle at the top of the main loop.
  - **Structural fix**: the reliability montage export used to run *after* the median/categorize/region-analysis block, even though it only depends on `seg_results` (available right after the reliability check) — moved the cheap metadata lookups (`export_data`, `animal_idx`, `slice_val`, `at`, `frame_duration_ms`) earlier and the montage export to run immediately after the reliability check, before the heavier median/region-analysis steps.
  - Moved `compute_segment_reliability()` out entirely — first to `functions/spike_reliability.py` as a renamed free function `compute_reliability()`, then (per the user's explicit request) restructured again into a class, `SpikeReliabilityChecker`, in `classes/spike_reliability.py` — owns `check()` (the detection loop) and `export_montage()` (the montage PNG export that used to be a separate `_save_reliability_montage()` helper). The class imports `SpatialCategorizer`/`RegionAnalyzer`'s helper functions rather than instantiating either class, preserving the original performance intent (avoiding a full `RegionAnalyzer` per raw segment).
- `functions/plot_results.py` — new `plot_spike_detection_summary()` (full Vm trace + picked/skipped/collapsed spike markers), used by `AbfClip._export_spike_plot()`.
- `functions/__init__.py`, `classes/__init__.py` — lazy-import registry updates to match all of the above (registered `plot_spike_detection_summary` and `SpikeReliabilityChecker`; removed the short-lived `functions/spike_reliability.py` registration after that module was superseded by the class).
- `export_zscore_segments.py` (new, from earlier in the session) — exports GAUSS-normalized per-spike z-scored segments of `2025_06_11-0003` and `2025_12_15-0012` to `output/` subfolders for manual inspection.
- All changes were committed by the user along the way (two intermediate backup commits: `4d10000`, `2303dca`); working tree is clean as of this wrap-up.

## Summary of current progress
- Confirmed all 6 TODOs from Session 55's tracker are done: spike+1 reliability check, density-gated detection merged into `RegionAnalyzer`, tested against the production median pipeline, reliability merged into the main workflow, detected-only-segments median, and results re-plotted (reliability montage + B% removal) — user explicitly confirmed this at the top of today's session.
- Ran an "assembly-line" debugging pass at the user's request: one issue at a time, each investigated and confirmed before any edit, each fix verified against a real recording (`2025_06_11-0003` and/or `2025_12_15-0012`) before moving on. Every fix in this session followed that pattern — no code was changed without an explicit go-ahead (one exception self-corrected: an early rename was applied before confirmation, caught by the user, and reverted before being reapplied properly).
- Found and fixed a real, previously-unnoticed bug: the spike-segment windowing (`set_interval_frames`) used the *mode* of available margins, which silently discarded roughly half of all detected spikes on a real recording. This was not a hypothetical — verified with before/after counts on real data.
- Iteratively restructured the reliability-checking code's location and shape based on direct user feedback: function → moved to `functions/` → renamed → finally reshaped into a class in `classes/` that owns both the detection logic and its own figure export, at the user's explicit design request (mirroring how `RegionAnalyzer`/`SpatialCategorizer` are structured).
- User expressed real frustration partway through ("the codes so messy", "I am really messed") — this is a reaction to the accumulated state of research code being audited closely for the first time, not to anything done in today's session; today's changes were all incremental, targeted, and verified. See [[feedback_pace_and_tone]].

## Completed TODOs/Tasks (before new wrap-up)
- ✅ All 6 TODOs from Session 55 confirmed done by the user
- ✅ Extracted `BASELINE_SIGMA_MULT` magic number in `SpatialCategorizer`
- ✅ Renamed `ref_df`/`cell_df`/`save_thread` and `animal_index_map` to intuitive names, with formatting cleanup
- ✅ Replaced `AbfClip`'s 4-CSV spike export with a single annotated PNG plot
- ✅ Fixed the `set_interval_frames` mode-based spike-dropping bug (52% → 18% drop rate, verified on real data)
- ✅ Extracted `MAX_SET_INTERVAL_FRAMES` magic number
- ✅ Fixed the reliability-montage export ordering (now runs right after the reliability check, not after the median/region-analysis block)
- ✅ Restructured per-segment reliability checking into `SpikeReliabilityChecker` (`classes/spike_reliability.py`), verified end-to-end after each restructuring step

## What should we do next? (TODOs)
- (none — the `CATEGORY_BRIGHT`/`CATEGORY_BACKGROUND` duplication was fixed in a same-day follow-up: `classes/region_analyzer.py` now imports `CATEGORY_BRIGHT` from `classes/spatial_categorization.py` instead of redefining it; `CATEGORY_BACKGROUND` dead code removed. Verified all three import paths — `region_analyzer`, `spatial_categorization`, and `classes`'s lazy loader — resolve to the identical object.)
- (Informational, not a bug) `DENSITY_THRESH` is currently `0.1` in `classes/region_analyzer.py`, versus `0.15` recorded as the confirmed value in an earlier session's plan doc. User clarified this is their own in-progress tuning, to be optimized per objective (10X/40X/60X) later — no action needed unless the user raises it again.

## Last Session Recap
※ recap: Ran a user-directed "assembly-line" debugging pass over `ach_domain_analysis.py`/`AbfClip`/the reliability-checking code, fixing issues one at a time with confirmation before each edit and real-data verification after. Found and fixed a genuine bug (mode-based spike windowing was dropping ~52% of spikes; replaced with a 20th-percentile cutoff, verified 52%→18%). Replaced `AbfClip`'s messy 4-CSV export with one annotated PNG. Renamed several unclear variables/functions. Restructured the reliability-checking code through several iterations into `SpikeReliabilityChecker`, a class in `classes/spike_reliability.py`, per the user's explicit design direction. Later the same day, closed the last open item: deduplicated `CATEGORY_BRIGHT`/`CATEGORY_BACKGROUND` (region_analyzer.py now imports from spatial_categorization.py). No open TODOs remain from this session.

---

# Log of the project progress 2026-09-14 Mon (Session 55)
Last working file: `prototype_density_hotspots.py`
Last working line: 17 (`WINDOW_PX = 200`)

## List of modified files
- `ach_domain_analysis.py` (M)
- `classes/region_analyzer.py` (M)
- `classes/results_exporter.py` (M)
- `classes/spatial_categorization.py` (M)
- `functions/plot_results.py` (M)
- `prototype_density_hotspots.py` (new, untracked) (<- Break here, line 17)
- `prototype_export_zscore_before_after.py` (deleted, staged)
- `prototype_print_thresholds.py` (deleted, staged)
- `prototype_reliability_sigma2.py` (deleted, staged)
- `prototype_segment_contours.py` (deleted, staged)

No commits made this session — all changes above remain unstaged/staged as listed, your call when to commit.

## Summary of current progress (based on modified files, existing plans)
- Confirmed the prior session's `spatial_categorization.py` refactor (single `baseline_frames_2sigma` threshold, binary 0/1 `CATEGORY_BRIGHT`, `CATEGORY_DIM` removed entirely) still holds — regenerated real `_CAT.tif` files from the new code and diffed them against production originals; user confirmed "looks ok."
- Cleaned up `output/` and 6 prototype scripts left over from that verification phase (kept pre-existing `gaussian`/`reliability_contours`/`reliability_prototype` folders).
- Diagnosed two real gaps in the existing detection pipeline: `compute_area_pct`/`pick_critical_frame` in `RegionAnalyzer` count bright pixels blindly with no spatial-density awareness, and `morphological` grouping in `SpatialCategorizer` has no minimum-blob-size filter — both let noise skew per-segment reliability.
- Ran `SpatialCategorizer` + `RegionAnalyzer` directly on each individual raw segment (not just the spike-centered median) for the two test recordings (`2025_06_11-0003`: 48 segments, `2025_12_15-0012`: 20 segments) to measure per-segment reliability; first pass gave 60.4%/30.0%, but flagged that `RegionAnalyzer`'s 10σ significance bar (tuned for the low-noise median) looks unstable on individual noisier raw segments.
- Diagnosed a per-segment threshold-instability problem: `baseline_frames_2sigma` estimates `mean + 2*std` from only ~10 pre-spike frames per segment; on some segments this produces catastrophic over-thresholding (up to a third of an entire 1024×1024 frame spuriously "bright").
- Designed and built `prototype_density_hotspots.py`: a local-density-gated hotspot detector — `uniform_filter` density map (fixed `WINDOW_PX`) → density-threshold gate → cluster via `RegionAnalyzer`'s existing `_run_cluster_seeker` (reused directly, not reinvented) → whole-blob expansion so a hotspot's own ragged edge pixels aren't cut off by the density gate.
- Iterated through several visualization/counting bugs live with the user: the raw smoothed density field being drawn instead of the real pixel shape, tiny noise-level "qualifying" islands (2–44px) being counted as separate clusters, contour+badge annotations being unreadable on near-uniform noise-failure frames — resolved by reusing `_run_cluster_seeker`'s eps-bridging + 5%-of-bright-pixels relative filter and switching to translucent color-fill shading.
- Landed on `WINDOW_PX=200`, `DENSITY_THRESH=0.15` as the latest (not yet finalized) settings: `2025_06_11-0003` = 20/48 (41.7%) segments detected, `2025_12_15-0012` = 12/20 (60.0%).
- Flagged but not yet investigated: `seg28`'s `threshold_used` computed to exactly `2.0000`, raising a question of whether `baseline_frames_2sigma` is effectively pinned near 2.0 for most segments by construction of the per-pixel z-scoring itself, rather than genuinely varying per segment.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Confirmed `spatial_categorization.py`'s threshold refactor via real `_CAT.tif` regeneration and diff against production originals
- ✅ Cleaned up `output/` and stale prototype scripts from the refactor-verification phase
- ✅ Diagnosed the spatial-blindness (`compute_area_pct`) and noise-passthrough (`morphological`, no size filter) gaps in the existing detection pipeline
- ✅ Built and iteratively fixed a working density-gated hotspot-detection prototype, reusing `RegionAnalyzer._run_cluster_seeker` rather than reinventing clustering
- ✅ Landed on current `WINDOW_PX=200` / `DENSITY_THRESH=0.15` settings with measured reliability numbers for both test recordings
- ✅ (Carried over from Session 53, now resolved) 3-way pixel categorization simplified: `CATEGORY_DIM` removed entirely, `CATEGORY_BRIGHT` renumbered 0/1

## What should we do next? (TODOs)
- [ ] Check if including the spike+1 frame (not just the spike frame) increases reliability
- [ ] Merge the density-hotspot detection code into `region_analyzer.py` (production)
- [ ] Test the modified `region_analyzer.py` against the original production pipeline that analyzes the median segment (not raw segments) — confirm nothing breaks
- [ ] Merge the reliability analysis into the main workflow/pipeline
- [ ] Check whether using only detected segments (vs. all segments) to compute the final median improves results
- [ ] Rethink how results get presented/plotted (categorized frames, clusters, reliability) given `CATEGORY_DIM` is gone and color-fill cluster shading is the new visualization style

## Last Session Recap
※ recap: Diagnosed spatial-blindness/noise gaps in the existing hotspot detection pipeline and built `prototype_density_hotspots.py`, a density-gated detector reusing `RegionAnalyzer`'s cluster seeker; landed on WINDOW_PX=200/DENSITY_THRESH=0.15 (41.7%/60.0% reliability on the two test recordings), with 6 concrete next steps pending (spike+1 test, production merge, median-pipeline regression test, reliability-in-workflow, detected-only-median check, replot).

---

# Log of the project progress 2026-09-08 Tue (Session 54)
Last working file: (discussion only — no files modified this session besides this log)
Last working line: n/a

## List of modified files
- `docs/continue_from_here.md` — this log entry only; no production code touched.

## Summary of current progress
- Continued the spike-reliability discussion from Session 53 with the user's original biological framing, stated explicitly for the first time: a patched cholinergic interneuron's action potential (spike) should trigger ACh release, which should show up as a hotspot at a consistent spatial position in the spike frame; segments are clipped per-spike from the ABF-derived spike times, and the median across segments was meant to reveal that consistent-position "trend."
- Worked through why "median shows the trend" and "per-trial reliability isn't 100%" aren't necessarily contradictory: a per-pixel median only needs a **majority** (>50%) of trials to show elevation, not all of them — so median-based detection is compatible with imperfect release probability, as long as reliability is above ~50%.
- The open puzzle from Session 53 (14.6% per-trial detection vs. median "DETECTED" for `2025_06_11-0003`) was reframed: the low 14.6% number is from a strict per-trial significance test calibrated for single-trial noise (10σ), not a direct measurement of "what fraction of trials show any elevation at the hotspot pixel." It's plausible far more than 24/48 trials show a real, if statistically-weak, elevation there — meaning the median may be working correctly while the per-trial threshold undercounts.
- Proposed (not yet run, user wants to think first) a direct empirical check: pull the raw z-score value at the detected hotspot pixel across all 48 individual segments at the spike frame, and look at what fraction show any positive elevation (not pass/fail against a threshold) — would distinguish "median is legitimately tracking a majority trend, per-trial threshold is just too strict" from "something else is going on spatially."
- User raised a further, deeper open question during TODO confirmation: **whether a given cholinergic interneuron's hotspot positions are even spatially consistent trial-to-trial** — i.e. before fixing the threshold, it may be worth first asking whether "the same pixel lights up across trials" is actually true, or whether release sites vary in position from trial to trial (which would undermine the whole per-pixel-median approach regardless of threshold tuning).
- User explicitly asked me to stay calm/unhurried about this — these are open, exploratory statistical questions to sit with, not bugs to rush to fix (see [[feedback_pace_and_tone]]).

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Restated and grounded the biological rationale behind spike-segment clipping and median-based hotspot detection
- ✅ Reasoned through the majority-vs-100%-reliability requirement for per-pixel median to shift
- ✅ Reframed the Session 53 per-trial-vs-median mismatch as likely a threshold-calibration issue rather than a median-correctness issue
- ✅ Proposed (but did not yet run) a raw-distribution check to test this reframing empirically

## What should we do next? (TODOs)
- [ ] **Reconsider the per-trial significance threshold.** The flat `10σ` bar (`AREA_PCT_SIGMA_MULT` in `classes/region_analyzer.py`) is calibrated for single-trial noise and likely undercounts true per-trial reliability relative to what the median-based detection can pick up. No fix designed yet — needs more thought before touching code.
- [ ] **Open question to return to: are a given cholinergic interneuron's hotspot positions spatially consistent across trials?** Raised by the user as a prior question to the threshold issue — worth checking (e.g. per-trial peak/cluster centroid positions, do they cluster spatially or drift?) before assuming the per-pixel-median approach is the right tool at all.
- [ ] (Optional, offered not yet run) Empirical check: raw z-score distribution at the detected hotspot pixel across all 48 segments of `2025_06_11-0003`, to test whether >50% of trials show any elevation there.
- [ ] (Carried over, still open) Commit the uncommitted z-score pipeline rename changes from `functions/fit_bg_hist.py` (Session 52 and earlier).
- [ ] (Carried over, still open) Run the full proc-list/GUI-driven pipeline (`ctrl_img_proc.py`) with all cumulative z-score pipeline changes before trusting it in production.
- [ ] (Carried over, still open) Check whether `data/pick_list.json`'s change from an earlier GUI smoke-test is wanted or should be reverted.

## Last Session Recap
※ recap: Discussed the biological rationale behind spike-segment clipping and median-based hotspot detection, and reframed the per-trial-vs-median reliability mismatch as likely a too-strict per-trial threshold rather than a flaw in the median approach. User raised a deeper open question — whether hotspot positions are even spatially consistent trial-to-trial — to think about before any fix. No code changed; purely conceptual discussion, paused for lunch.

---

# Log of the project progress 2026-09-07 Mon (Session 53)
Last working file: `classes/region_analyzer.py` (reading only — investigating detection logic, no edits)
Last working line: n/a (whole-file / cross-file investigation)

## List of modified files
- (none — `git status` clean; this was a pure investigation/diagnostic session, no production code touched, per standing "never touch the codes without confirmation" rule)
- One scratch prototype script (`median_vs_trials_check.py`) and diagnostic PNGs were written to the session scratchpad / `output/reliability_prototype/` (gitignored, outside version control) — not part of the reviewed codebase.

## Summary of current progress
- Continued the parked **spike reliability analysis** work from Session 51/52: built a prototype (previous session segment) that runs the existing hotspot-detection pipeline (`SpatialCategorizer` + `RegionAnalyzer`) on each individual per-spike segment (`lst_zscore`) instead of only the median-merged segment, computing `reliability% = 100 × detected_segments / total_segments` on 2 real recordings (`2025_06_11-0003`: 48 segments, `2025_12_15-0012`: 20 segments).
- **User flagged an inconsistency**: for `2025_06_11-0003`, per-trial reliability came out to only 7/48 = 14.6% detected, yet the production median-based pipeline reports the same recording as "hotspot detected" (per `data/ana_20260902_001.txt`'s prior run). Investigated and confirmed this numerically by running the same detection on the median segment directly (`significant=True, n_clusters=1, B%=1.56%` → DETECTED), reproducing the mismatch live.
- **Root-caused the mismatch**: the significance bar (`baseline_mean + 10×baseline_std`) is computed from each segment's *own* baseline noise. A single trial's baseline noise is much larger than the median-of-48 segment's baseline noise (median-of-N noise scales down by roughly 1/√N ≈ 1/7 for N=48), so the same fixed 10σ multiplier is effectively ~7× stricter for individual trials than for the median. This means per-trial reliability% and the median "detected" flag are not measuring comparable things — a real but small/inconsistent signal can clear the median's low-noise bar while failing almost every individual trial's high-noise bar.
- Verified the z-score normalization order: confirmed (`functions/zscore_img_segs.py`) that each segment is independently z-scored per-pixel using *only that segment's own pre-spike baseline frames* **before** `spike_centered_median()` ever merges segments — median operates on 48 already-normalized segments, not raw data.
- Verified the median derivation itself is correct: confirmed (`classes/abf_clip.py:260-267`) that every segment is guaranteed identical length and spike-frame-aligned at the same index by construction (single shared `set_interval_frames` value), so the per-pixel-per-frame median (`functions/spike_alignment.py::_cpu_median_axis0`) is comparing true same-time-offset-from-spike data across trials — no alignment bug found.
- At the user's prompting, confirmed the 3-way pixel categorization (background/dim/bright) is only meaningfully 2-way for detection: `compute_area_pct()` and `_run_cluster_seeker()` (`classes/region_analyzer.py`) both check only `CATEGORY_BRIGHT`; the `dim` category is used nowhere in the B%/clustering/significance math — it only affects the categorized-frame plot's "gray" coloring in `functions/plot_results.py:350`. Collapsing dim into background would not change any detection number.
- These findings surfaced mid-investigation; user has not yet decided how (or whether) to address the 10σ/noise-scaling mismatch, redefine the reliability metric, or simplify the dim/bright categorization — explicitly deferred, no action items confirmed yet (see TODOs below).

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Re-derived and explained the full existing hotspot-detection chain (categorize → B% → significance → clustering) on request
- ✅ Explained the `base977` naming (97.7th-percentile one-tailed cutoff at baseline mean + 2σ) and clarified the one-tailed vs. two-tailed (97.7% vs 95%) distinction
- ✅ Verified the median-segment derivation (`spike_centered_median`) is mathematically correct and that segment alignment is guaranteed by construction
- ✅ Verified z-score normalization happens per-segment, before the median merge, using only local pre-spike baselines
- ✅ Diagnosed and numerically confirmed the per-trial-vs-median detection mismatch (14.6% reliability vs. median "DETECTED"), traced to the fixed 10σ bar being applied to noise levels that differ ~7× between single-trial and median-of-48 baselines
- ✅ Confirmed the `dim` pixel category is unused in all detection math (B%, clustering) — cosmetic only

## What should we do next? (TODOs)
- [ ] **Spike reliability analysis — still not finalized.** The per-segment prototype exists only in the scratchpad; nothing has been formalized into `ach_domain_analysis.py`/`RegionAnalyzer`/the GUI. Before any implementation: decide how to make per-trial and median-based detection results comparable (e.g. a noise-scaled or fixed-magnitude threshold instead of a flat `10σ` multiplier applied identically at both noise scales), then redefine what "reliability%" should actually report.
- [ ] Decide whether to act on the dim/bright finding — collapsing background/dim into one category would simplify `SpatialCategorizer`/`RegionAnalyzer` with no change to detection numbers, but this is a cleanup opportunity, not a bug fix; not yet requested.
- [ ] (Carried over, still open) Commit the uncommitted z-score pipeline rename changes from `functions/fit_bg_hist.py` (Session 52 and earlier) — still sitting uncommitted per earlier summaries; verify this is still the case next session.
- [ ] (Carried over, still open) Run the full proc-list/GUI-driven pipeline (`ctrl_img_proc.py`) with all cumulative z-score pipeline changes before trusting it in production.
- [ ] (Carried over, still open) Check whether `data/pick_list.json`'s change from an earlier GUI smoke-test is wanted or should be reverted.

## Last Session Recap
※ recap: Investigated the spike-reliability prototype's odd result (14.6% per-trial detection vs. median "DETECTED" for `2025_06_11-0003`) and root-caused it: the fixed 10σ significance bar is ~7× stricter for noisy single trials than for the much-less-noisy median-of-48 segment, so the two numbers aren't directly comparable. Also confirmed the `dim` pixel category is unused in all detection math (cosmetic only). No code changed — pure investigation; reliability analysis and any threshold fix remain undecided/unimplemented.

---

# Log of the project progress 2026-09-05 Sat (Session 52)
Last working file: `functions/zscore_img_segs.py` (viewing, not editing — user was re-reading this while wrapping up)
Last working line: n/a (whole-file review)

## List of modified files
- `img_proc.py` — `process_biexp()` pipeline changed from bi-exp detrend + Gaussian blur (saving ΔF/F0 ratio values) to bi-exp detrend + stack-wide z-score normalization + Gaussian blur. Output filename unchanged (`*_BIEXP_GAUSS.tif`), per explicit user request to keep the original naming. `SIGMA` constant changed `6.0` → `4.0` (see rationale below).
- `functions/detrend.py` — `biexp_detrend()` (both `_cpu_biexp` and `_gpu_biexp`) now returns the residual `y - trend` instead of the ratio `(y - trend) / trend`. Docstrings updated to match. The `/trend` division was the root cause of the original speckle bug (division blows up wherever the fitted trend is near zero, e.g. background/out-of-tissue pixels), so it was removed rather than patched.
- `functions/background_zscore.py` (new) — `fit_background_sigma(residual)`: pools the whole stack's residual values into one histogram, finds the peak, fits a Gaussian to the peak + left side only (right side is contaminated by real signal transients), returns `(mean, sigma)`. `zscore_normalize(residual, mean, sigma)`: applies that single global mean/sigma to the whole stack.
- `functions/__init__.py` — registered `fit_background_sigma`/`zscore_normalize` in the lazy-import system (`TYPE_CHECKING` block, `__all__`, `_LAZY_IMPORTS`).
- `data/pick_list.json` — changed as a side effect of a GUI launch smoke-test (`main.py` started then killed after 6s to verify nothing broke) — not an intentional content change; flagged here in case it's unwanted and needs reverting.

## Summary of current progress
- **Root-caused and fixed the speckle problem** the user reported after Gaussian blur on `*_BIEXP_GAUSS.tif`: the old pipeline computed ΔF/F0 as `(y-trend)/trend` *before* blurring — near-zero `trend` values (background/dim regions) turned small noise into huge outliers, which blur then smeared into visible speckle blobs instead of averaging away.
- **New pipeline**: bi-exp detrend → residual (`y-trend`, no division) → stack-wide z-score (global background-noise mean/sigma from a histogram peak fit) → Gaussian blur → save as `*_BIEXP_GAUSS.tif` (same name, new meaning: values are now z-scores, not ΔF/F0 fractions).
- **Confirmed float16 storage is still safe** for z-score magnitudes (verified empirically: max z-scores stayed under 93, well within float16's representable range; float16 precision is relative/exponent-based, not tied to small values specifically — this was a real misconception the user had going in, corrected using `docs/knowledgebase/float_precision_and_dtypes.md`).
- **Chose `SIGMA=4.0` over the old `6.0`** based on real-data visual comparison (not just theory): on `2025_06_11-0003` frame 242, a real hotspot resolved into two distinguishable lobes at sigma=4 but merged into one blob at sigma=6 — direct evidence that sigma=6 (~8µm at 10X, close to the existing `EPS_UM=10µm` inter-varicosity spacing used elsewhere) was destroying real spatial structure, not just noise.
- **Found and fixed a real bug during implementation**: the new module was first named `zscore_normalize.py`, colliding with its own exported `zscore_normalize` function name. Python's import system auto-attaches a submodule to its parent package under the submodule's filename, which silently shadowed the lazy-loaded function and caused `TypeError: 'module' object is not callable`. Fixed by renaming the module to `background_zscore.py`.
- **Verified ALS correction and spike-aligned analysis need no changes** — checked (not assumed) that ALS (`functions/als.py`) is exactly scale- and shift-equivariant (its per-frame weight depends only on the sign of `data - baseline`, never magnitude), and that `zscore_img_segs()` re-normalizes per-pixel-per-segment using each pixel's own local pre-spike baseline, which exactly cancels any uniform global affine transform applied upstream. Proved algebraically (not just hand-waved) that Gaussian blur (weight-normalized convolution) and ALS both commute exactly with a globally-uniform affine transform, so the new z-score step is mathematically inert for final spike-aligned analysis numbers — it only affects the standalone appearance/usability of the exported GAUSS/ALS TIFFs.
- Verified end-to-end: ruff clean on all touched files, `process_biexp()` run directly on 2 real files (`2025_06_11-0003.tif`, `2025_12_15-0012.tif`) via a since-deleted prototype script — numbers matched the earlier prototype exactly, and a full GUI launch smoke-test came back clean.
- Also exported ALS-corrected versions of both test files (lam=70, p=0.05, n_iter=100) to `output/` per user request, confirming ALS peak values barely shift relative to pre-ALS GAUSS values (consistent with the equivariance proof above) — these were prototype outputs, not part of the reviewed code changes.
- 4 one-off `prototype_*.py` scripts used during this session's exploration (histogram-fit visualization, sigma comparison, real-pipeline check, ALS check) were all deleted at the end per user request — nothing left in repo root.
- User pushed back on the resulting design ("z-score over z-score" — calculating a global z-score in `img_proc.py` when `zscore_img_segs()` already re-z-scores locally downstream) — confirmed their instinct was correct: the global step is provably redundant for final analysis numbers, kept only for the exported intermediate file's standalone visual quality. This is flagged as an open TODO below (user has a new idea: maybe scaling instead of full re-z-scoring is what's actually needed).

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Diagnosed the post-blur speckle bug in `*_BIEXP_GAUSS.tif` (root cause: `/trend` division exploding near-zero-trend background pixels)
- ✅ Prototyped and validated a stack-wide z-score normalization approach (bi-exp residual → global background histogram/Gaussian fit → z-score) on 2 real recordings
- ✅ Confirmed float16 export is safe for z-score magnitudes (corrected a precision misconception)
- ✅ Empirically chose `SIGMA=4.0` over `6.0` using real hotspot structure (two-lobe merge test on `2025_06_11-0003`)
- ✅ Landed the z-score pipeline in `img_proc.py` + new `functions/background_zscore.py`, keeping `*_BIEXP_GAUSS.tif` naming
- ✅ Found and fixed a lazy-import module/function name collision bug (`zscore_normalize.py` → `background_zscore.py`)
- ✅ Verified ALS correction (`als_correct.py`/`functions/als.py`) needs no changes — proved scale/shift equivariance
- ✅ Verified spike-aligned analysis (`zscore_img_segs()`, `SpatialCategorizer`) needs no changes — proved the new global z-score step is downstream-canceled
- ✅ Ran real pipeline end-to-end on 2 test files + full GUI smoke test — both clean
- ✅ Exported ALS-corrected prototype outputs (lam=70, p=0.05, n_iter=100) for user review
- ✅ Cleaned up all 4 prototype scripts

## What should we do next? (TODOs)
- [ ] **Spike reliability analysis (parked)** — not started, carried from a prior session. Target: last popup (`view_align_spike.py`/`ctrl_align_spike.py`/`ach_domain_analysis.py`), reusing per-spike segments from `AbfClip` before median-merging. Open questions: reliability scope (trial-to-trial vs. cross-recording), auto-run vs. toggle, report destination.
- [ ] **Run the full proc-list-driven pipeline** (via the GUI / `ctrl_img_proc.py`, not just direct-call prototype scripts) with the new z-score logic before trusting it in production — only the 2 direct-call test files have been verified so far, not the real proc-list → GUI → `img_proc.run()` path.
- [ ] **Reconsider the purpose of `zscore_img_segs()`** — user's own note: since the new global z-score step in `img_proc.py` is downstream-canceled by `zscore_img_segs()`'s local per-pixel-per-segment re-normalization anyway, maybe `zscore_img_segs()` doesn't need to do a full re-z-score at all — perhaps just a scaling step would suffice. Needs a fresh design discussion (not yet scoped) on what `zscore_img_segs()` should actually be responsible for now that upstream units have changed from ΔF/F0 to z-score.
- [ ] Check whether `data/pick_list.json`'s change (from this session's GUI smoke test) is wanted or should be reverted — flagged, not yet decided.

## Last Session Recap
※ recap: Fixed the post-blur speckle bug in `*_BIEXP_GAUSS.tif` by replacing the ΔF/F0 ratio detrend with a stack-wide z-score normalization (bi-exp residual → global background histogram/Gaussian fit → z-score → blur at sigma=4, chosen via real hotspot evidence over the old sigma=6). Verified ALS and spike-aligned analysis need no changes (proved the new step is downstream-canceled). Fixed a lazy-import name-collision bug along the way. Open: full GUI-pipeline verification still pending, spike reliability analysis still parked, and a new idea to simplify `zscore_img_segs()` now that it's redundant with the upstream z-score.

---

# Log of the project progress 2026-09-05 Sat (Session 51)
Last working file: (discussion only — no files modified this session)
Last working line: n/a

## List of modified files
- (none — `git status` clean; this was a planning/discussion session, no code touched)

## Summary of current progress
- Discussed where to add spike reliability analysis: agreed it belongs in the last popup (`views/view_align_spike.py` / `controllers/ctrl_align_spike.py` / `ach_domain_analysis.py`), since `AbfClip` already produces per-spike segments (`clip.lst_img_frame_ranges`) there before they get merged into `median_segment` — reusing that data avoids re-clipping the ABF in a separate popup.
- Scoping questions raised but not yet answered: (1) reliability of what — trial-to-trial consistency within one recording vs. consistency across recordings/animals; (2) how it should run — folded into the existing `run()` loop automatically vs. an optional toggle in the popwin; (3) where results should land — appended to the existing per-entry xlsx report vs. a separate report.
- User then redirected focus to a current issue in `img_proc.py` — not yet described, to be picked up next.

## Completed TODOs/Tasks (before new wrap-up)
- (none — discussion only)

## What should we do next? (TODOs)
- [ ] **Spike reliability analysis (parked)** — not started. Target: last popup (`view_align_spike.py`/`ctrl_align_spike.py`/`ach_domain_analysis.py`). Needs the 3 open scoping questions above answered before a plan can be drafted.
- [ ] **Current `img_proc.py` issue** — user flagged wanting to focus on this next; symptom/details not yet shared.

## Last Session Recap
※ recap: Scoped (discussion only, no code) where spike reliability analysis should live — the last popup, reusing per-spike segments already produced by `AbfClip` before median-merging. Three open questions remain (reliability scope, auto-run vs toggle, report destination). Session then pivoted to a not-yet-described `img_proc.py` issue.

---

# Log of the project progress 2026-07-07 Mon (Session 50)
Last working file: `classes/region_analyzer.py`
Last working line: end of file

## List of modified files
- `classes/region_analyzer.py` — renamed all cryptic variables in `_run_cluster_seeker` to self-explaining names (`dilated`→`within_eps_of_bright`, `cc_map`→`component_map`, `n_components`→`n_raw_components`, `total_bright`→`total_bright_px`, `kept`→`accepted_components`, `comp_label`→`component_id`, `comp_mask`→`bright_px_in_component`, `pixel_count`→`bright_px_count`, `new_label`→`cluster_idx`, `comp_coords`→`bright_px_coords`, lambda `item`→`component`); added inline comments explaining the why of each step; fixed 8 stale DBSCAN references in docstrings/comments (module docstring, `RegionAnalyzer` class docstring, `_run_cluster_seeker` return label, `compute_ring_traces`/`compute_cluster_trace` docstrings, `pick_critical_frame` docstring, `MIN_CLUSTER_FRACTION` comment); removed a double blank line
- `docs/dbscan_notes.md` — full rewrite: removed all DBSCAN-specific content, replaced with the distance-transform + connected-components algorithm; corrected `B+D%`→`B%` (dim pixels excluded); updated `AREA_PCT_SIGMA_MULT` to `10.0` (was stale `5.0`); updated `EPS_UM` to `30.0`; removed `min_samples`/`MIN_DENSITY_FRAC`/`eps_and_min_samples` section; removed saturation guard section (added explanation it was removed and why); replaced all `_detect_clusters` references with `_run_cluster_seeker`
- `docs/distance_transform_clustering.md` — updated all variable names to match the new `_run_cluster_seeker` names; added explanation of why `~` flip is needed; expanded the `binary_dilation` cost section with the per-pixel AND operation explanation

## Summary of current progress
- Variable naming in `_run_cluster_seeker` is now self-documenting — every variable name describes what it holds, and inline comments explain the non-obvious algorithmic choices
- Both clustering docs are now accurate and consistent with the actual code
- All DBSCAN references purged from `region_analyzer.py` docstrings and comments
- ruff clean throughout

## Completed TODOs
- ✅ Renamed all cryptic variables in `_run_cluster_seeker` to self-explaining names
- ✅ Added inline why-comments to `_run_cluster_seeker`
- ✅ Rewrote `docs/dbscan_notes.md` for the distance-transform algorithm
- ✅ Updated `docs/distance_transform_clustering.md` with new variable names and expanded cost explanation
- ✅ Cleaned 8 stale DBSCAN docstring/comment references in `region_analyzer.py`

## What should we do next? (TODOs)
- (none)

## Last Session Recap
※ recap: Documentation and naming cleanup session — renamed all cryptic variables in `_run_cluster_seeker` to self-explaining names with inline why-comments; fully rewrote `docs/dbscan_notes.md` for the new distance-transform approach (removed saturation guard, DBSCAN params, B+D% → B%); updated `docs/distance_transform_clustering.md` with new variable names; purged 8 stale DBSCAN references from `region_analyzer.py` docstrings. No logic changes.

---

# Log of the project progress 2026-07-07 Mon (Session 49)
Last working file: `docs/knowledgebase/rolling_background_thread_pattern.md`
Last working line: 46

## List of modified files
- `ach_domain_analysis.py` — removed ISO timestamps from `[SKIPPED]` lines; gated `peak_latency_ms` + `lasting_time_ms` on `n_clusters > 0`; split `_save_entry_figures` to use `"spatial"` and `"latency"` subfolders; updated console log; split `build_stats_report()` into two tables (Spatial / Temporal)
- `classes/region_analyzer.py` — 4 bug fixes: threshold floor (`AREA_PCT_MIN_ELEVATION = 0.5`), `skimage_label` connected components for saturated path, `DBSCAN_MIN_SAMPLES = 100` fixed constant, `_compute_max_area()` + `get_results()` gated on `len(clusters) > 0`
- `classes/results_exporter.py` — docstring updated: `region_sta/` → `spatial/` + `latency/`
- `functions/database_ops.py` — `compute_region_stats()` expanded with `median`, `iqr_q1`, `iqr_q3`, `cv_pct`, `geomean`, `geostd_factor`; new `_geomean_geostd()` helper; `import math` added
- `functions/plot_results.py` — module docstring updated: `region_sta/` → `spatial/` / `latency/`
- `docs/knowledgebase/rolling_background_thread_pattern.md` — code example updated to `"spatial"` / `"latency"`
- `run_ach.slm` — removed stale `cp -r results/ /bucket/...` line

## Summary of current progress
- **6 bug fixes**: threshold floor for near-zero baseline, connected-component labeling for saturated frames, `DBSCAN_MIN_SAMPLES = 100` constant across all objectives, `_compute_max_area()` / latency / lasting time all properly null-gated when no clusters detected
- **Stats report split into two tables**: Spatial (area + radius — shows Mean, Std, CV%, Median, IQR, GeoMean, GeoStd*) and Temporal (latency + lasting time — Mean, Std only); `GeoStd*` is a dimensionless multiplier (typical range = GeoMean ÷ GeoStd to GeoMean × GeoStd)
- **Output folder split**: `*_SPATIAL.png` → `results/spatial/`, `*_LATENCY.png` → `results/latency/` (was both in `region_sta/`); all `region_sta` references cleaned up across code and docs

## Completed TODOs
- ✅ Removed timestamps from `[SKIPPED]` log lines
- ✅ Threshold floor for near-zero baseline std (`AREA_PCT_MIN_ELEVATION = 0.5`)
- ✅ Saturated path: `skimage_label` connected components
- ✅ `DBSCAN_MIN_SAMPLES = 100` fixed constant across all objectives
- ✅ `_compute_max_area()` / latency / lasting time gated on `n_clusters > 0`
- ✅ Stats report expanded with Median/IQR/CV%/GeoMean/GeoStd for area metrics
- ✅ Stats report split into Spatial and Temporal tables
- ✅ Output folders split: `spatial/` and `latency/` (was `region_sta/`)

## What should we do next? (TODOs)
- (none)

## Last Session Recap
※ recap: Fixed 6 bugs in `region_analyzer.py`/`ach_domain_analysis.py` (threshold floor, connected-component saturated path, DBSCAN min_samples, null-gating of area/latency); expanded stats report with Median/IQR/CV%/GeoMean split into Spatial vs Temporal tables; split output from `region_sta/` into `spatial/` and `latency/`. Pipeline not yet re-run to verify the new layout.

---

# Log of the project progress 2026-07-06 Sun (Session 48)
Last working file: `ach_domain_analysis.py`
Last working line: ~391 (`console.log(f"[green]Exported {dir_names}/, region_sta/  ...")` — dropped the stale `full_traces/` mention)

## List of modified files
- `classes/region_analyzer.py` — implemented Lasting Time, but the final design diverged twice from the plan doc's original spec before landing:
  - First tried: extinction-frame detection via a baseline+5σ threshold on raw B+D% (`pick_extinction_frame`, per the original plan) — abandoned after checking real `*_SPATIAL.png` files together with the user and finding real recordings (e.g. `2025_04_03-0034`) show a non-monotonic decay with a plateau/shoulder well above baseline before finally settling, which a "2 consecutive frames below threshold" rule can misfire on.
  - Second tried: switch the extinction signal from raw B+D% to Bright-only % (`compute_bright_pct`, excluding Dim pixels) to dodge the Dim-halo shoulder — user then reconsidered mid-implementation ("if we use fitting, we don't need specific to bright area?") once fitting was on the table, since a reported R² already makes a bad fit visible; this function was added then removed again.
  - **Final design**: fit a single-exponential decay `A*exp(-T/tau)` to raw B+D% (`area_pct`) from its post-critical-frame peak (`decay_peak_frame_idx`) to segment end (`fit_decay_tau()`, using `scipy.optimize.curve_fit`); `lasting_time_ms = tau` (the fit's time constant itself, in ms via `get_lasting_time_ms(frame_duration_ms)`); R² reported alongside so a bad/bi-phasic fit is visible rather than silently misleading. Guards: skips the fit (all `None`) if fewer than `MIN_DECAY_FIT_FRAMES=3` post-peak frames exist, or the post-peak trace barely varies (`MIN_DECAY_FIT_RANGE`), or `curve_fit` doesn't converge.
  - `get_results()` gained `decay_peak_frame_idx`, `decay_peak_offset`, `decay_fit_r2`.
- `classes/results_exporter.py` — new DB columns `decay_peak_offset INTEGER`, `decay_fit_r2 REAL`, `lasting_time_ms REAL`; `lasting_time_ms` threaded through `export_all()`/`_upsert_record()` exactly like the existing `peak_latency_ms` (computed by the caller with `frame_duration_ms`, passed in as its own arg — not baked into `region_data` since `get_results()` doesn't take `frame_duration_ms`).
- `functions/database_ops.py` — `compute_region_stats()`'s `metric_cols` gained `"lasting_time_ms"` (user's explicit ask, separate from the earlier plan) — verified it shows up correctly in the aggregated stats table (mean 122.35 ms, std 60.83, n=2 cells on the test ana list).
- `ach_domain_analysis.py` — computes `lasting_time_ms = region_analyzer.get_lasting_time_ms(frame_duration_ms)`, logs it (or a "fit failed" note) to console, passes it to `export_all()`. Also, per a separate small request mid-session: moved `plot_full_trace`'s export from `full_traces/` to `region_sta/` and renamed its filename suffix `TRACE`→`LATENCY`; fixed a stale `full_traces/` mention in the "Exported ..." console log line.
- `functions/plot_results.py` — new `_draw_decay_fit()` helper overlays the fitted exponential decay curve (dashed) on the row-0 B+D% trace (`ax_bd` in `plot_spatiotemporal_summary`), labeled `τ=... ms (R²=...)`, or an italic "fit failed" note when no fit is available. Docstring's `plot_full_trace` folder reference updated to `region_sta/, *_LATENCY.png`.
- `.claude/plans/xy-span-lasting-time.md` — only lightly touched (marked as in-progress at some point during the session); **still describes the old extinction-frame+threshold design, not the final decay-fit approach** — flagged as a TODO below.
- Deleted 4 stray debug/profiling scripts from repo root (`_fix_latency_db_tmp.py`, `_geom_test.py`, `_geom_test.png`, `_profile_io_tmp.py`, `_profile_tmp.py`) — pre-existing clutter, unrelated to this session's actual work but swept up in the same commit.
- `results/region_sta/`, `results/full_traces/` — regenerated/removed data, not tracked by git (`results/` is gitignored); `full_traces/` folder was deleted (old `*_TRACE.png` files superseded by the new `region_sta/*_LATENCY.png`), pipeline rerun twice to regenerate clean output under the new layout.
- Everything above is already committed as `4d1f8b6` ("complete lasting time estimation") — working tree is clean.

## Summary of current progress
- Lasting Time is fully implemented and verified end-to-end, but via a materially different design than what the plan doc describes — went through two real pivots (baseline-threshold extinction → Bright%-only extinction → exponential-decay-fit) driven by checking actual `*_SPATIAL.png` files together with the user at each step rather than assuming the original plan would hold up.
- Visually verified the decay fit on real data: clean single-spike recording (`2025_04_03-0032`) fits tightly (τ=202ms, R²=0.98); a recording with a genuine post-peak plateau/shoulder (`2025_04_03-0034`) still fits reasonably (τ=213ms, R²=0.91) — confirming the fit-based approach is robust to the exact failure mode that broke the threshold-based design.
- Full 8-recording pipeline reran clean twice this session (once after the lasting-time implementation, once after the `full_traces`→`region_sta`/`LATENCY` rename) — ruff clean throughout, `results.db` schema confirmed, `compute_region_stats()` confirmed reporting `lasting_time_ms` correctly.
- Separately explored (discussion only, no code): quantifying Vm spike-peak-time distribution across trials within a recording — some recordings (e.g. `2025_04_03-0032`, 3 trials) show tightly aligned peaks safe to average; others (e.g. `2025_06_11-0002`, 46 trials) show peaks scattered across the whole ~50ms frame window (real burst firing, no single dominant time). Discussed why mean±std is misleading for bimodal/scattered cases and sketched a mode+concentration% alternative, but user explicitly deferred this pending a discussion with their boss — **not started, no code, don't assume a direction next session**.
- User flagged wanting to work on GUI tuning next, plus three new feature ideas (small ROI, loading the red channel TIFF, a patched-soma coordinate locator) — none scoped or discussed yet.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Lasting Time fully implemented: `fit_decay_tau()`, `get_lasting_time_ms()`, `decay_peak_frame_idx`/`decay_peak_offset`/`decay_fit_r2` in `get_results()`
- ✅ DB columns `decay_peak_offset`/`decay_fit_r2`/`lasting_time_ms` added and populated
- ✅ `lasting_time_ms` added to `compute_region_stats()`'s aggregated metrics
- ✅ Console logging for lasting time (tau + R², or fit-failed note)
- ✅ Decay-fit curve overlay on the B+D% trace row in `plot_spatiotemporal_summary`
- ✅ Verified end-to-end on real data: clean-decay case and shoulder/plateau case, both visually confirmed
- ✅ `full_traces/` → `region_sta/` export folder consolidation, `TRACE`→`LATENCY` filename suffix rename, old folder cleaned up
- ✅ `append_stats.py` checked — confirmed no changes needed (generic wrapper around already-updated `compute_region_stats()`)

## What should we do next? (TODOs)
- [ ] **Update `.claude/plans/xy-span-lasting-time.md`** — the Lasting Time section still describes the abandoned extinction-frame+threshold design (`pick_extinction_frame`, `extinction_threshold_pct`/`extinction_frame_idx`/`extinction_frame_offset`), not the final exponential-decay-fit approach that's actually implemented and committed (`fit_decay_tau`, `decay_peak_frame_idx`/`decay_peak_offset`/`decay_fit_r2`, `lasting_time_ms = tau`). Should be rewritten to match reality before it misleads a future session.
- [ ] **GUI tuning** — user wants to work on this next; no specifics given yet, needs scoping discussion first.
- [ ] **Add small ROI support** — new feature idea, not yet scoped.
- [ ] **Load red channel TIFF** — new feature idea, not yet scoped.
- [ ] **Patched-soma coordinate locator** — new feature idea, not yet scoped.
- [ ] **Vm peak-time distribution (on hold)** — idea to quantify per-trial spike-time spread (mode + concentration % instead of mean±std, since some recordings are genuinely bimodal/scattered); explicitly deferred by the user pending a discussion with their boss — do not resume without checking in first.

## Last Session Recap
※ recap: Implemented Lasting Time via an exponential-decay fit on B+D% (τ + R², after two design pivots away from the plan doc's threshold-based extinction-frame approach, driven by real-data checks with the user), verified end-to-end on both a clean-decay and a shoulder/plateau recording, and renamed `full_traces/`→`region_sta/`+`LATENCY` suffix. Discussed (no code) a mode+concentration% alternative for Vm peak-time spread, deferred pending the user's boss. Next: fix the now-stale plan doc, then GUI tuning + 3 new unscoped feature ideas (small ROI, red channel TIFF, patched-soma locator).

---

# Log of the project progress 2026-07-06 Sun (Session 47)
Last working file: `functions/plot_results.py`
Last working line: `_draw_span_bbox()` — `linewidth=2.0`

## List of modified files
- `classes/region_analyzer.py` — `_compute_max_area()` now also computes and stores `max_area_x_min_px` / `max_area_y_min_px` (top-left corner of the DBSCAN-kept accepted-mask bbox, in pixels); returned as two extra values and unpacked in `__init__`. Used by callers drawing a Rectangle overlay.
- `functions/plot_results.py` — new `_draw_span_bbox()` helper draws a yellow dashed `Rectangle` patch (`linewidth=2.0, edgecolor="#f1c40f", facecolor="none", linestyle="--", alpha=0.85, zorder=4`) on the max-area frame panel; called from `plot_spatiotemporal_summary()` in the 9-panel loop after `_draw_cluster_shading()`.

## Summary of current progress
- Completed the X/Y span verification end-to-end (clean `results.db` run, DB inspection via `_read_experiments()`, visual PNG check) — all 8 recordings populated with correct span values, no excluded recordings.
- Added the bbox rectangle overlay (Option A) for X/Y span visualization: a yellow dashed `Rectangle` drawn over the DBSCAN-kept accepted-mask bounding box on the max-area frame panel only. Confirmed on both a 10X recording (tight cluster, small bbox) and a 60X saturated recording (bbox fills most of frame, as expected).
- Both `region_analyzer.py` and `plot_results.py` pass ruff clean after changes.
- Pipeline verified exit 0 twice this session (pre-bbox for DB inspection, post-bbox for visual confirmation).

## Completed TODOs
- ✅ X/Y span verified end-to-end: pipeline, DB columns, console log, panel title text
- ✅ Bbox rectangle overlay added to SPATIAL PNG on max-area frame panel
- ✅ `max_area_x_min_px` / `max_area_y_min_px` stored in `RegionAnalyzer` for caller use
- ✅ Bbox linewidth tuned to 2.0 (visually confirmed)
- ✅ Panel title `fontsize=12` change from Session 46 implicitly verified by two pipeline runs this session

## What should we do next? (TODOs)
- [ ] **Implement lasting time** — per `# X/Y Span And Lasting Time Plan`: `get_lasting_time_ms(frame_duration_ms)` method in `RegionAnalyzer`; `extinction_threshold_pct` / `extinction_frame_idx` / `extinction_frame_offset` added to `get_results()`; DB columns `extinction_frame_offset` and `lasting_time_ms` in `ResultsExporter`; B+D% trace annotations in `plot_spatiotemporal_summary()` (horizontal threshold line, extinction-frame marker, optional shading from critical frame to extinction frame).

## Last Session Recap
※ recap: Verified X/Y span end-to-end (DB + visual PNG check). Added a yellow dashed bounding-box rectangle overlay (`_draw_span_bbox()`) on the max-area frame panel in SPATIAL PNGs, backed by new `max_area_x_min_px`/`max_area_y_min_px` attributes in `RegionAnalyzer._compute_max_area()`. Tuned linewidth to 2.0. All ruff clean, pipeline exit 0. Next: implement lasting time.

---

# Log of the project progress 2026-07-05 Sun (Session 46)
Last working file: `functions/plot_results.py`
Last working line: ~365 (`ax.set_title(fontsize=12)` in `_plot_frame_panel`)

## List of modified files
- `functions/plot_results.py` — multiple plot layout and labeling changes:
  - Fixed panel title alignment: switched from flat `fig.add_gridspec(3, 9)` to a nested layout — `gs_outer = fig.add_gridspec(3, 1)` + `gs_panels = gs_outer[1].subgridspec(1, 9)`. All 9 panel axes now share identical height, so `ax.set_title()` anchors consistently across all panels.
  - Removed B+D area stat from panel titles (was one of 3–4 lines per panel, causing height variation and requiring `max_title_lines` compensation). Panels now show only the frame label + hotspot area (critical frame only) + span (max-area frame only). This allowed going back to plain `ax.set_title()` with single color.
  - Added µm² to row 0 legend labels: format `xxxxx µm² (xx%)` for spike-1/spike/spike+1/critical entries.
  - Changed figure `suptitle` from "Spatiotemporal Analysis" → "Spatial Analysis".
  - Changed row 0 `ax.set_title` to "Bright+Dim area coverage per frame | star = critical frame (spike or spike+1)"; y-label to "B+D area (%)".
  - Removed the "RegionAnalyzer — OBJ=..." settings text block (was placed between row 0 and row 1); also removed the now-dead `settings_text` variable and unused imports (`CATEGORY_BRIGHT`, `CATEGORY_DIM`, `SATURATION_AREA_PCT`).
  - Changed panel title `fontsize` 10 → 12 (matches the other row titles); **not yet verified by a pipeline run**.
- `classes/abf_clip.py` — replaced `→` (U+2192) with `->` in two console.log strings to avoid `UnicodeEncodeError` under `uv run` (cp1252 console encoding on Windows).
- `ach_domain_analysis.py` — removed `✓` from one console.log f-string for the same reason.

## Summary of current progress
- Resolved the long-standing panel title misalignment in `*_SPATIAL.png` (different numbers of stat lines per panel causing frame labels to sit at different heights) by switching to a nested subgridspec. The fix is structural — no more manual y-coordinate calculation, no `max_title_lines` pre-loop.
- Simplified the figure: RegionAnalyzer settings banner removed, B+D per-panel clutter removed; that info is now fully represented in the row 0 legend (which gained µm² values this session).
- Fixed Unicode crash that prevented `uv run` from completing on Windows (cp1252 encoding, `→` and `✓` chars).
- Verified end-to-end (ruff clean, pipeline exit 0, PNGs visually confirmed) for all changes except the final `fontsize=12` bump on panel titles.

## Completed TODOs
- ✅ Fixed panel title alignment (nested subgridspec)
- ✅ Removed B+D from panel titles; back to `ax.set_title()` single color
- ✅ Added µm² to row 0 legend in `xxxxx µm² (xx%)` format
- ✅ Renamed figure title to "Spatial Analysis"
- ✅ Updated row 0 axis titles ("Bright+Dim area coverage per frame", "B+D area (%)")
- ✅ Removed RegionAnalyzer settings text + dead imports
- ✅ Fixed Unicode crash (`→`, `✓`) in `abf_clip.py` and `ach_domain_analysis.py`
- ✅ Updated `docs/dbscan_notes.md` (stale "demo figures" ref + `critical_frame_area_um2` note)

## What should we do next? (TODOs)
- [ ] **Complete X/Y Span + Lasting Time plan** — plan doc: `# X/Y Span And Lasting Time Plan`. X/Y span is partly in (`RegionAnalyzer`, `ResultsExporter` DB columns, console log, SPATIAL title display); needs full pipeline verification on a clean `results.db`. Lasting time is not yet implemented: `get_lasting_time_ms()` method, `extinction_threshold_pct`/`extinction_frame_idx`/`extinction_frame_offset` in `get_results()`, DB columns `extinction_frame_offset`/`lasting_time_ms`, and B+D% trace annotations (threshold line, extinction marker, shading). Unit tests for both span (empty mask) and extinction logic (normal decay, 1-frame dip ignored, no extinction, spike+1 critical frame) per the plan's test section.

## Last Session Recap
※ recap: Fixed SPATIAL.png panel title alignment using nested subgridspec (structural fix — all panels identical height → `ax.set_title()` consistent). Simplified the figure: removed B+D from panel stat lines (moved µm² info to row 0 legend instead), removed the RegionAnalyzer settings banner, renamed the figure and axis titles. Fixed a Unicode crash breaking `uv run` on Windows. Updated `docs/dbscan_notes.md`: fixed "demo figures" ref to "export figures" and added a note on `critical_frame_area_um2` and its Session 45 bug fix.

---

# Log of the project progress 2026-07-04 Sat (Session 45)
Last working file: `classes/region_analyzer.py`
Last working line: 414 (end of the new `_detect_clusters()` helper, added last)

## List of modified files
- `classes/region_analyzer.py` — three real changes this session:
  - Fixed `max_area_um2`: was raw non-background pixel count (`area_pct`) on the max-area frame, no clustering/filtering at all — user had flagged this as "probably also wrong" at the end of Session 44. Now runs the same DBSCAN cluster-detection path used for the critical frame on the max-area frame too, and sums only the kept-cluster pixels (excludes DBSCAN noise/label=-1 pixels). Confirmed correct on multi-cluster frames: sums area across *all* kept clusters, not just the largest (user's explicit call).
  - Fixed `critical_frame_area_um2`: same raw/unfiltered-area bug, found while spot-checking real DB rows after the `max_area_um2` fix — on `2025_06_11-0003` the two stats described the same physical frame but disagreed (62,379 µm² raw vs 53,090 µm² cluster-filtered). Now computed from `self.label_frame` (kept clusters only), matching `max_area_um2`'s definition. `critical_frame_area_pct` deliberately stays raw (it drives the saturation/significance threshold logic).
  - Readability refactor (user's explicit request mid-session): fixed a stale module docstring (still described the old pre-DBSCAN "largest connected component" approach); deduplicated the saturation-guard branch — previously implemented twice with slightly different code (once in `__init__`, once in a helper) — into one shared `_detect_clusters()` function; split `__init__` (~85 lines doing 4 different things) into `__init__` + `_build_clusters()` + `_compute_max_area()`.
- `classes/results_exporter.py`, `functions/database_ops.py`, `functions/plot_results.py`, `spike_analysis.py` — untouched today, carried over uncommitted from Session 44 (see that session's entry below for what's in them)
- `data/ana_list_20260622_000.txt` — touched only by repeated verification pipeline reruns (stats block regenerated each time), not a real content change
- `docs/continue_from_here.md` — this wrap-up

## Summary of current progress
- Fixed two real, verified bugs in `RegionAnalyzer` found via actual DB inspection (not assumption): `max_area_um2` and `critical_frame_area_um2` both counted raw non-background pixels instead of the DBSCAN-filtered "real region" area — confirmed by comparing both stats on frames where they describe the same physical frame and seeing them disagree before the fix, then match exactly after
- Did a scoped, behavior-preserving readability refactor of `region_analyzer.py` per the user's explicit request: fixed a stale docstring, removed a real duplication risk (the saturation guard existed in two slightly-different copies), and split an overloaded `__init__` into named steps
- Verified against the real 8-recording ana list after every change this session — ruff clean, pipeline exit 0 each time, and confirmed the refactor was behavior-preserving by diffing DB values before/after (identical)
- Nothing committed yet — all 7 modified files remain uncommitted on the `gui` branch (4 of them carried over from Session 44, untouched today)

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fixed `max_area_um2` — cluster-filtered area (noise excluded) instead of raw non-background count
- ✅ Fixed `critical_frame_area_um2` — same fix, found via real-data consistency check against the first fix
- ✅ Confirmed multi-cluster area semantics: sum across all kept clusters, not just the largest
- ✅ Deduplicated the saturation-guard logic into a single `_detect_clusters()` helper
- ✅ Fixed stale module docstring (described the old pre-DBSCAN approach)
- ✅ Split `RegionAnalyzer.__init__` into `__init__` + `_build_clusters()` + `_compute_max_area()`

## What should we do next? (TODOs)
- [ ] **Update `docs/dbscan_notes.md`** — confirmed still stale across many sessions; still describes the old RMS-based R and pre-DBSCAN approach, doesn't mention any of the DBSCAN/cluster-filtered-area work from this or recent sessions

## Last Session Recap
※ recap: Fixed two real bugs where `max_area_um2` and `critical_frame_area_um2` counted raw non-background pixels instead of DBSCAN-filtered region area (found by comparing the two stats on real data); then did a behavior-preserving readability refactor of `region_analyzer.py` (stale docstring, deduped saturation-guard logic, split `__init__`). Verified end-to-end against real data throughout. Nothing committed yet. Next: update stale `docs/dbscan_notes.md`.

---

# Log of the project progress 2026-07-04 Sat (Session 44)
Last working file: `functions/plot_results.py`
Last working line: 346 (`def _overlay_clusters(...)` — right after removing the now-reverted `_draw_max_area_span`)

## List of modified files
- `classes/region_analyzer.py` — fixed `critical_frame_area_pct` being a percentage when it should have been µm² (added `critical_frame_area_um2`, kept `critical_frame_area_pct` too per user's explicit request); dropped the write-only `region_sta_results` JSON blob concept from `get_results()`'s shape; fixed a real R-overshoot bug via new `_resolve_R()` (enclosing-circle radius capped at the centroid's distance to the nearest frame edge — previously a farthest-pixel R could draw a circle past the frame boundary for off-center clusters); renamed per-cluster `R_px`/`R_um` → `R_lat_px`/`R_lat_um` (critical/latency frame only); added an independent `max_area_frame_idx`/`max_area_offset`/`max_area_um2` (whichever of spike/spike+1 has the larger raw B+D%, decoupled from the threshold-based critical-frame pick) — a first attempt added a clustering-based `R_px`/`R_um` for this frame, then an x/y bounding-box-span version, **both were ultimately removed this session** per user's final call; net state now: `max_area_frame_idx`/`max_area_offset`/`max_area_um2` only, no R/span for the max-area frame
- `classes/results_exporter.py` — removed the `region_sta_results` TEXT column (and the now-dead `NumpyEncoder`/`json` import) since nothing in the codebase ever read it back and every field in it already had its own column; added then removed `max_area_x_span_um`/`max_area_y_span_um` columns; final schema has `critical_frame_area_um2`, `max_area_offset`, `max_area_um2`, `R_lat_px`, `R_lat_um` (no plain `R_px`/`R_um` anymore)
- `functions/database_ops.py` — `compute_region_stats()` dropped `n_clusters` from the aggregated stats table (per user request — raw column stays in the DB, just not in the mean/std report); metric now reports `max_area_um2` instead of the old `critical_frame_area_pct`
- `functions/plot_results.py` — all centroid markers (DBSCAN cluster centroids in both `region_sta` and `full_traces` plots) changed from `+` to `×`, drawn with explicit `zorder` so they stay visible over translucent overlays; added then fully removed the max-area x/y-span visualization (bounding-box cross + label) after visual review showed the region wasn't useful as designed
- `spike_analysis.py` — console logging updated to match the renamed `R_lat_um` field and the new (then-trimmed-back-down) max-area log line; final version just logs `Max-area frame {tag}: area={um2} µm²`
- `data/ana_list_20260622_000.txt` — touched by repeated verification pipeline runs (stats block regenerated each time)

## Summary of current progress
- Fixed three real, verified bugs found via actual data/visual inspection (not assumption): (1) `critical_frame_area_pct` was percentage when µm² was needed, (2) R could draw a circle that overshoots the frame edge for off-center clusters — fixed via `_resolve_R()`'s edge-distance cap, (3) the write-only `region_sta_results` JSON blob was pure duplication of already-flat columns, confirmed by inspecting real DB rows
- Extensive back-and-forth on what a second, independent "max-area frame" stat should report: tried a second DBSCAN-clustered R, then an x/y bounding-box span with a visual cross+centroid marker — the bounding-box version had its own real bug (cross drawn as `centroid ± span/2` instead of the actual bbox coordinates, since the mean centroid isn't necessarily at the box's geometric center) which was fixed, but the whole x/y-span idea was still discarded by the user at the very end as "didn't fit what I want to see"
- User separately flagged `max_area_um2` itself as "probably also wrong" — not investigated further this session, deferred
- Full 8-recording pipeline verified clean (ruff + end-to-end rerun, `results.db` schema inspected) after every single change this session — multiple full reruns, all exit 0
- Nothing committed yet — all 6 modified files are uncommitted on the `gui` branch

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fixed `critical_frame_area_pct` → added real `critical_frame_area_um2`
- ✅ Removed the redundant/write-only `region_sta_results` JSON blob + dead `NumpyEncoder`
- ✅ Fixed R-overshoot-past-frame-edge bug via `_resolve_R()`
- ✅ Removed `n_clusters` from `compute_region_stats()`'s aggregated table
- ✅ Added independent max-area frame selection (`max_area_frame_idx`/`max_area_offset`/`max_area_um2`)
- ✅ Renamed critical-frame per-cluster R to `R_lat_px`/`R_lat_um` for clarity against the new max-area stat
- ✅ Changed all centroid markers from `+` to `×` across both export plots
- ⚠️ Added, debugged, and then fully removed a max-area x/y-span stat + visualization (net result: reverted, but the investigation/bugfix work happened)

## What should we do next? (TODOs)
- [ ] **Redesign the max-area stat** — user flagged `max_area_um2` as "probably also wrong"; needs a fresh design discussion on what this should actually measure before touching code again
- [ ] **Redesign the R / spatial-extent visualization** — the x,y bounding-box-span approach didn't fit what the user wants to see; needs a new approach from scratch, discussed before any code
- [ ] **Commit today's work** — all 6 modified files (`classes/region_analyzer.py`, `classes/results_exporter.py`, `functions/database_ops.py`, `functions/plot_results.py`, `spike_analysis.py`, `data/ana_list_20260622_000.txt`) are uncommitted on `gui`
- [ ] **Update `docs/dbscan_notes.md`** — confirmed still stale (still describes RMS-based R and pre-DBSCAN approach), carried over from multiple earlier sessions

## Last Session Recap
※ recap: Fixed three real bugs (wrong-unit area stat, R overshooting the frame edge, a write-only duplicate DB blob), then spent most of the session iterating on a new independent "max-area frame" stat — tried clustering-based R, then x/y bounding-box span with visualization, ultimately reverting the span/R part entirely per user's final call since it "didn't fit." `max_area_um2` itself was flagged as possibly still wrong. Also restyled all centroid markers from `+` to `×`. Nothing committed. Next: redesign both the max-area stat and its visualization from scratch in a fresh discussion, then commit.

---

# Log of the project progress 2026-07-03 Fri (Session 43)
Last working file: `classes/region_analyzer.py`
Last working line: 368 (`R = float(dists.max())` in `compute_ring_traces` — verified during a ring-trace double-check with the user)

## List of modified files
- `classes/region_analyzer.py` — completed steps 9-10 of the DBSCAN integration plan: `get_results()`/`get_summary()` now return the `clusters`/`n_clusters`/`has_region` shape instead of referencing the deleted `self.bright_largest`/`self.dim_largest` (previously broken); renamed `analysis_frame_idx`→`critical_frame_idx` and `pick_analysis_frame`→`pick_critical_frame` throughout; added `critical_frame_offset`/`critical_frame_area_pct` to `get_results()` and `saturated` to `get_summary()`; added a new `SATURATION_AREA_PCT = 15.0` guard — when the critical frame's B+D% is at/above this, DBSCAN is skipped (it was blowing up to 73GB+ memory on a real 82.5%-dense 60X recording) and every non-background pixel is instead treated as one big cluster (ring-split like any other single-cluster case, not reported as zero clusters)
- `functions/plot_results.py` — full rewrite. `plot_spatiotemporal_summary` (→ `region_sta/`) now mirrors the validated demo's "main" figure: a B+D% signal trace across the full segment (with spike-1/spike/spike+1 markers and a star at the critical frame) + 6 panels (spike-1..spike+4) with cluster shading drawn only on the critical frame's own panel, plus a DBSCAN-settings text row. `plot_full_trace` (→ `full_traces/`) now mirrors the demo's "rings" figure: 9 panels (spike-4..spike+4) with the fixed ring/circle overlay repeated identically on every panel + the full-segment z-score trace with that 9-panel window shaded/annotated. Also fixed a real rendering bug found via visual inspection: matplotlib was auto-expanding each panel's view to fit circles that extend past the frame edge, shrinking the visible image — panels now pin `xlim`/`ylim` to the image bounds and disable autoscale so circles clip at the edge instead
- `classes/results_exporter.py` — new `experiments` table schema replacing bright/dim columns (`has_bright_region`, `x_span_um`, `bright_area_um2`, `dim_area_um2`, etc.) with cluster-based ones (`n_clusters`, `has_region`, `saturated`, `critical_frame_offset`, `critical_frame_area_pct`, `R_px`, `R_um`; `centroid_y`/`centroid_x` now mean the largest cluster's centroid); `region_analysis` column renamed to `region_sta_results`; removed `optimize_region_data()` (no longer needed — `RegionAnalyzer.get_results()` is already serialization-safe); export filenames/folder structure deliberately left untouched per user's explicit instruction
- `functions/database_ops.py` (renamed from `query_databases.py`) — `compute_region_stats()`/`get_excluded_recordings()`/`get_cell_recording_status()` redesigned around the new schema: the `has_region` column is used directly instead of the old `bright_area_um2` null/zero-check helper (`_bright_excluded_expr`, deleted); `compute_region_stats()` now reports `n_clusters`/`R_um`/`critical_frame_area_pct`/`peak_latency_ms` instead of `bright_area_um2`/`dim_area_um2`/`total_area_um2`
- `functions/__init__.py` — lazy-import paths updated for the `database_ops` rename
- `spike_analysis.py` — fixed `RegionAnalyzer` construction (new `(cat_stack, med_stack, spike_frame_idx, obj=)` signature, dropped the removed `min_area_um2` param), fixed `get_peak_latency_ms()` call (now 1-arg), updated console logging to report cluster/saturated results, updated `plot_spatiotemporal_summary`/`plot_full_trace` call sites for their new signatures
- `archive/_demo_dbscan_tmp.py` (moved from repo root) — kept as the validated reference for the plot redesign per user's choice; fixed its one remaining `analysis_frame_idx`→`critical_frame_idx` reference so it still runs
- `data/ana_list_20260622_000.txt` — touched by verification pipeline runs (new stats block appended by `write_stats_report`)

## Summary of current progress
- Finished the DBSCAN-into-RegionAnalyzer integration that spanned sessions 39-42 — all 12 originally-planned steps are now done, plus the 3 downstream consumers (`spike_analysis.py`, `results_exporter.py`, `plot_results.py`) fully updated to match
- Found and fixed a real incident during verification: running the real pipeline against a live ana list crashed with the machine's memory ballooning to 73GB+ (thrashing, not just slow) — root-caused to `_run_cluster_seeker`'s DBSCAN call choking on an 82.5%-dense 60X frame (865K non-background pixels, ~5,250 estimated avg neighbors/point). Fixed via a `SATURATION_AREA_PCT=15%` guard, after an initial wrong implementation that treated the saturated case as *zero* clusters — corrected mid-session per user feedback ("saturated means we skip DBSCAN but still want the one cluster's rings, not nothing")
- Rewrote `plot_results.py` twice: first pass merged both of the validated demo's figures into one hybrid (wrong per user review), second pass restored the actual two-figure split faithfully from `archive/_demo_dbscan_tmp.py`. Also fixed a circle-clipping/panel-shrinking bug found via visual inspection of the exported PNGs
- Verified everything end-to-end multiple times against the real 8-entry ana list (`data/ana_list_20260622_000.txt`) — final run clean, exit 0, no memory issues, visually confirmed both saturated and real-cluster cases render correctly
- User double-checked the ring-trace math at the end of the session: confirmed current code uses max-distance R (not RMS — that was an older, already-replaced approach) with an R/√2 equal-area inner/outer split, and that inner/outer rings pool bright+dim pixels together (no B/D separation in the trace itself, only in the descriptive panel titles) — but flagged two open concerns for next session (see TODOs)
- Nothing committed yet — all changes are uncommitted on the `gui` branch, by the user's explicit choice (deferred to next session)

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Finished steps 9-10 of the DBSCAN integration plan (`get_results()`/`get_summary()`)
- ✅ Updated downstream consumers: `spike_analysis.py`, `classes/results_exporter.py`, `functions/plot_results.py`
- ✅ Renamed `analysis_frame_idx`→`critical_frame_idx`/`pick_analysis_frame`→`pick_critical_frame` throughout
- ✅ Diagnosed and fixed a real 73GB memory-blowup crash (DBSCAN on an oversaturated frame) via a new `SATURATION_AREA_PCT` guard
- ✅ Corrected the saturation guard to treat the saturated frame as 1 whole-frame cluster, not 0
- ✅ Renamed `query_databases.py`→`database_ops.py`, redesigned `compute_region_stats`/`get_excluded_recordings`/`get_cell_recording_status` for the new schema
- ✅ New `results.db` schema (`region_sta_results`, `n_clusters`, `saturated`, `critical_frame_offset`/`critical_frame_area_pct`, `R_px`/`R_um`)
- ✅ Rewrote `plot_results.py` to faithfully match the two validated demo figures (`region_sta` vs `full_traces`)
- ✅ Fixed circle-clipping/panel-shrinking bug in the row-1 panels
- ✅ Archived `_demo_dbscan_tmp.py`
- ✅ Verified full pipeline end-to-end multiple times against real data

## What should we do next? (TODOs)
- [ ] **Commit today's work** — nothing committed yet; `classes/region_analyzer.py`, `classes/results_exporter.py`, `functions/plot_results.py`, `functions/database_ops.py` (rename), `functions/__init__.py`, `spike_analysis.py`, `archive/_demo_dbscan_tmp.py` (move) are all uncommitted on `gui`
- [ ] **Update `docs/dbscan_notes.md`** — still stale, describes the pre-DBSCAN approach; carried over from multiple earlier sessions
- [ ] **Implement detection gate (`has_event` boolean)** — B+D%-vs-baseline event detection gate, carried from multiple earlier sessions, still not implemented
- [ ] **Bring back the plot of spiking traces** — user flagged this in an earlier wrap-up, still not scoped/discussed
- [ ] **Consider numba.jit acceleration** in `region_analyzer.py` — new idea raised this session, not yet investigated for which functions would actually benefit
- [ ] **Reconsider the R radius calculation for stats** — currently R comes from whichever frame was picked as the critical frame (spike or spike+1); user wants to reconsider whether R should instead be the *largest* R found across frame 0 vs frame+1, not just the critical frame's own R
- [ ] **Fix `critical_frame_area_pct` in stats** — user flagged this as "a wrong stat" during wrap-up; exact issue not yet elaborated, needs investigation next session

## Last Session Recap
※ recap: Finished the DBSCAN-into-RegionAnalyzer integration (steps 9-10 + all 3 downstream consumers), found and fixed a real 73GB memory-blowup crash via a new saturation guard (corrected mid-session to treat saturated frames as 1 cluster, not 0), and rewrote `plot_results.py` to match the validated demo's two-figure structure exactly (plus a circle-clipping fix). Verified end-to-end against real data multiple times — all clean. Nothing committed yet (deferred by choice). Next: commit, then revisit stale docs, the detection gate, numba acceleration, and two open questions about the R-radius calc and a possibly-wrong `critical_frame_area_pct` stat.

---

# Log of the project progress 2026-07-03 Fri (Session 42)
Last working file: `classes/region_analyzer.py`
Last working line: 27 (`AREA_PCT_SIGMA_MULT = 5.0` — tuned down from 10.0)

## List of modified files
- `classes/region_analyzer.py` — completed most of Session 41's 12-step DBSCAN integration plan (steps 3-8, 11-12; steps 9-10 remain):
  - Step 3: `eps_and_min_samples(obj)` — µm→px DBSCAN param conversion, reuses `PIXEL_SCALE`
  - Step 4: `_run_cluster_seeker(frame, eps_px, min_samples)` — DBSCAN + size-filter clustering (renamed from `_run_dbscan_filtered`)
  - Step 5: `compute_ring_traces(...)` — inner/outer ring split (R/√2), vectorized trace calc; dropped `touches_boundary` (no consumer)
  - Step 6: `__init__` rewrite — now `RegionAnalyzer(cat_stack, med_stack, spike_frame_idx, obj="10X")`, analyzes immediately on construction, storing `self.clusters`; dropped the now-superseded `min_area_um2` param
  - Steps 7-8: deleted `_find_largest_category`, `_find_all_dim`, and `area_in_combined_region` (superseded by DBSCAN clustering, no external callers)
  - Steps 11-12 (reworked twice): `get_temporal_traces()`/`get_peak_latency_ms()` no longer take a `segment` param (already computed at construction). Final design, after user clarified the original 0/1/>1-cluster idea was two genuinely different computations, not one unified formula: **1 kept cluster** → inner/outer ring-split latency (spread within one release site); **>1 kept clusters** → new `compute_cluster_trace()` (whole-cluster trace, no ring split) + latency = latest-peaking cluster's time minus earliest-peaking cluster's time (asynchrony across release sites)
  - `R` now stored as both `R_px` and `R_um` per cluster (µm conversion via the previously-unused `_px_to_um` helper)
  - `AREA_PCT_SIGMA_MULT` tuned from `10.0` → `5.0` after real-data testing showed the fixed-at-10 threshold silently defaulted to the spike frame on `2025_06_11-0005` even though spike+1 was clearly the dominant signal (2.45% vs 1.23%); validated the new value against all 5 test recordings — only that one recording's frame-pick changes, the other 4 are unaffected
  - Renamed for clarity: `compute_bd_pct`→`compute_area_pct`, `bd_pct`→`area_pct` (param/var), `BD_PCT_SIGMA_MULT`→`AREA_PCT_SIGMA_MULT`, `stack`→`cat_stack` (ctor param), `_nanargmax_relative`→`_peak_offset_from_spike` — all to avoid "bd"/"db" (database) confusion and improve readability
  - File reorganized: `class RegionAnalyzer` now precedes all module-level helper functions (previously the other way around)
- `_demo_dbscan_tmp.py` — repurposed from a standalone validation demo (its own duplicate DBSCAN/ring-trace logic) into a live test harness that imports and exercises the real `RegionAnalyzer` class against 5 real CAT/MED recordings; plotting rewritten to show every cluster's ring/circle overlay and z-score trace (not just one), with the trace x-axis aligned to spike-relative offsets and latency shown directly on the trace subplot title

## Summary of current progress
- Completed 10 of the original 12 steps from Session 41's DBSCAN integration plan (steps 9-10 — `get_results()`/`get_summary()` — remain; both are currently broken, still referencing the deleted `self.bright_largest`/`self.dim_largest`)
- Validated the full pipeline end-to-end against 5 real recordings (both 10X and 60X objectives, 0/1/multi-cluster cases) using the repurposed `_demo_dbscan_tmp.py`
- Found and fixed two real bugs via real-data testing rather than assumption: (1) the ring/latency PNG visualization was silently only showing one cluster's ring and trace even when multiple clusters existed, misrepresenting the max-across-clusters logic — rewrote to overlay every cluster; (2) `pick_analysis_frame`'s significance threshold (10σ) was strict enough that neither spike nor spike+1 cleared it for one real recording, so the code fell through to an unconditional "default to spike frame" that ignored spike+1 being nearly 2× higher — fixed by tuning the threshold to 5σ (a first attempted code-level fallback fix was reverted per user request in favor of this simpler threshold tune)
- Major mid-session design correction: the user's original 0/1/>1-cluster latency rule was initially (mis)implemented as one unified "compute ring-split latency per cluster, take the max" loop; user clarified this collapsed two different concepts into one — single-cluster ring latency (spatial spread) and multi-cluster asynchrony latency (temporal gap between separate release sites) needed genuinely different computations, not just a shared branch. Rebuilt accordingly, including a new `compute_cluster_trace()` function and a visualization redesign (single R circle per cluster instead of ring pairs when >1 cluster)

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Step 3: `eps_and_min_samples(obj)`
- ✅ Step 4: `_run_cluster_seeker` (DBSCAN + size filter)
- ✅ Step 5: `compute_ring_traces` (inner/outer ring split)
- ✅ Step 6: `__init__` rewrite (`cat_stack`, `med_stack`, `spike_frame_idx`, `obj`)
- ✅ Steps 7-8: removed `_find_largest_category`/`_find_all_dim`/`area_in_combined_region`
- ✅ Steps 11-12: `get_temporal_traces()`/`get_peak_latency_ms()` reworked for 1-cluster ring latency vs >1-cluster cross-cluster asynchrony latency
- ✅ New `compute_cluster_trace()` for the >1-cluster whole-region trace
- ✅ `R` now available in both px (`R_px`) and µm (`R_um`)
- ✅ Tuned `AREA_PCT_SIGMA_MULT` 10.0→5.0, validated against all 5 real recordings
- ✅ Naming cleanup: `compute_area_pct`, `area_pct`, `AREA_PCT_SIGMA_MULT`, `cat_stack`, `_peak_offset_from_spike`, `_run_cluster_seeker`
- ✅ File reorganized: class before module-level helpers
- ✅ `_demo_dbscan_tmp.py` rewritten as a real `RegionAnalyzer` test harness, validated visually and numerically against 5 real recordings

## What should we do next? (TODOs)
- [ ] **Finish steps 9-10**: `get_results()`/`get_summary()` — still broken right now, referencing the deleted `self.bright_largest`/`self.dim_largest`; needs `{"clusters": [...]}` and `n_clusters`/`has_region` shapes instead
- [ ] **Update downstream consumers**: `spike_analysis.py` (still constructs `RegionAnalyzer` with the old single-frame + `min_area_um2=900` signature), `classes/results_exporter.py`, `functions/plot_results.py` — all still expect the old `bright_largest`/`dim_largest` result shape
- [ ] **Update `docs/dbscan_notes.md`** — still stale, describes the old RMS-based R and pre-DBSCAN approach, doesn't mention this session's `compute_cluster_trace`/asynchrony-latency design
- [ ] **Implement detection gate (`has_event` boolean)** — B+D%-vs-baseline event detection gate, carried from earlier sessions, still not implemented
- [ ] **Consider bringing back the plot of spiking traces** — user flagged this during wrap-up, not yet scoped/discussed
- [ ] `_demo_dbscan_tmp.py` — kept for now per user's choice (still useful for testing steps 9-10 and downstream-consumer changes); revisit deleting it once `RegionAnalyzer` is fully wired into the real pipeline

## Last Session Recap
※ recap: Completed steps 3-8 and 11-12 of the DBSCAN-into-RegionAnalyzer integration (steps 9-10 remain), validated end-to-end against 5 real recordings, and fixed two real bugs found via that testing (ring/latency plot only showing one cluster; frame-picking silently defaulting to spike over a clearly-dominant spike+1 — fixed via `AREA_PCT_SIGMA_MULT` 10→5). Mid-session redesigned the multi-cluster latency logic after clarifying it needed genuinely different math for 1 vs >1 clusters, not a shared formula. Next: finish steps 9-10, then update the three downstream consumers still expecting the old bright/dim result shape.

---

# Log of the project progress 2026-07-02 Thu (Session 41)
Last working file: `classes/region_analyzer.py`
Last working line: 71 (`return spike_frame_idx` — final fallback in `pick_analysis_frame`)

## List of modified files
- `classes/region_analyzer.py` — added 2 of 12 planned functions toward integrating the validated DBSCAN pipeline (from `_demo_dbscan_tmp.py`) into `RegionAnalyzer`, replacing the largest-CC approach:
  - `compute_bd_pct(stack)` — vectorized bright+dim (non-background) pixel % per frame, reuses this file's own `CATEGORY_BACKGROUND` constant (native rewrite, not a demo import)
  - `pick_analysis_frame(bd_pct, spike_frame_idx)` — **redesigned from the demo**, not a straight port: instead of "whichever of spike/spike+1 has higher B+D% wins" (demo's logic, noise-prone), now compares each candidate frame against a baseline-derived significance threshold (`baseline_mean + BD_PCT_SIGMA_MULT * baseline_std`, computed over every frame before the spike frame); defaults to the spike frame unless spike+1 clears the threshold (delayed-signal case). New tunable constant `BD_PCT_SIGMA_MULT = 10.0`.

## Summary of current progress
- Full integration plan for moving DBSCAN clustering into `RegionAnalyzer` was worked out and corrected multiple times this session before any code was written — see "How to work with the user" below, the iteration process itself is important context for continuing this task
- Both new functions verified with `ruff check classes/region_analyzer.py` — clean
- Working tree otherwise unchanged; only `classes/region_analyzer.py` has uncommitted changes

## The 12-step integration plan (dependency-ordered)
1. ✅ `compute_bd_pct(stack)` — B+D% per frame
2. ✅ `pick_analysis_frame(bd_pct, spike_frame_idx)` — pick spike or spike+1 (redesigned, see above)
3. ⬜ `eps_and_min_samples(obj)` — µm→px conversion for DBSCAN (only needed once a frame is picked)
4. ⬜ `_run_dbscan_filtered(frame, eps_px, min_samples)` — DBSCAN + size-filter clustering
5. ⬜ `compute_ring_traces(label_frame, centroid, med_stack, cluster_k)` — inner/outer ring z-score traces per cluster
6. ⬜ `__init__` rewrite — **constructor signature must change**: currently takes a single already-picked 2D `spike_frame`, but steps 1-2 need the whole categorized **stack** (multiple frames) to compute B+D% and pick a frame internally. This is the biggest structural change in the plan.
7. ⬜ Remove `_find_largest_category` (superseded by DBSCAN clustering)
8. ⬜ Remove `_find_all_dim` (dim no longer unioned separately — DBSCAN clusters bright+dim together as one non-background point pattern)
9. ⬜ `get_results()` → returns `{"clusters": [...]}` instead of `{"bright_largest":, "dim_largest":}`
10. ⬜ `get_summary()` → `has_bright_region`/`has_dim_region` becomes `n_clusters`/`has_region`
11. ⬜ `get_temporal_traces()` → bright/dim category trace becomes inner/outer ring trace per cluster
12. ⬜ `get_peak_latency_ms()` → reworked for ring-peak (inner vs outer) instead of bright/dim peak

## How to work with the user on this task (read this first in the new session)
- **Go step by step, one function at a time.** After showing code for a step, stop and wait for an explicit "go"/"yes" — do not call Edit/Write until you get it, even if the user approved the overall list earlier. Approving the list is not approval to implement item 1.
- **Don't port `_demo_dbscan_tmp.py` code verbatim.** The user explicitly rejected importing/copying from the demo file — every function must be a native rewrite inside `region_analyzer.py`, reusing this file's own constants/conventions (e.g. `CATEGORY_BACKGROUND`, `PIXEL_SCALE`) even when the underlying logic matches the demo.
- **Trace real data dependencies before ordering steps.** Earlier in this session the step order was wrong (DBSCAN/eps listed before frame-selection) — the user caught that DBSCAN can't run until `pick_analysis_frame` has already chosen a frame. Steps 3-5 depend on steps 1-2, and step 6 (constructor) depends on all of steps 1-5 existing first.
- **Discuss design before coding, especially thresholds/constants.** `pick_analysis_frame` went through a real design discussion (relative-% threshold → baseline+kσ threshold) before any code was shown — the user prefers this over jumping straight to an implementation.
- The demo file `_demo_dbscan_tmp.py` is open in the user's IDE and is the reference implementation for *behavior*, not code to copy.

## What should we do next? (TODOs)
- [ ] Continue the 12-step `RegionAnalyzer` integration — resume at step 3 (`eps_and_min_samples`)
- [ ] Update `docs/dbscan_notes.md` — still stale (describes old RMS-based R, doesn't mention the max-distance fix, boundary-crop flag, or this session's baseline+σ frame-picking logic)
- [ ] Implement the full detection gate (`has_event` boolean) — `pick_analysis_frame` only chooses *which* frame to analyze; the original TODO wanted a B+D%-vs-baseline gate that also decides *whether* an event occurred at all — not yet implemented
- [ ] Once steps 6-12 land: update downstream consumers that read the old `bright_largest`/`dim_largest` shape — `spike_analysis.py`, `functions/plot_results.py`, `classes/results_exporter.py`

## Last Session Recap
※ recap: Started integrating the validated DBSCAN pipeline into `RegionAnalyzer`, working step-by-step with explicit approval per step. Completed 2 of 12 planned functions (`compute_bd_pct`, `pick_analysis_frame` — the latter redesigned from the demo to use a baseline+10σ significance threshold instead of "whichever frame is higher"). Next: resume at step 3 (`eps_and_min_samples`), continuing the same one-step-at-a-time, discuss-before-code workflow.

---

# Log of the project progress 2026-07-01 Wed (Session 40)
Last working file: `_demo_dbscan_tmp.py`
Last working line: 286 (`ax_bd.set_xlabel("Frame offset from spike (0 = spike)", fontsize=9)`)

## List of modified files
- `_demo_dbscan_tmp.py` — multiple changes this session:
  - Fixed `compute_ring_traces()`: R changed from RMS distance to `dists.max()` (true enclosing-circle radius, per explicit user choice over 90th-percentile); outer ring bound at R is now automatic (no pixel can exceed the max by definition)
  - Added `touches_boundary` detection — flags clusters whose mask touches the frame edge (R is truncated/underestimated in that case); shown as `⚠ boundary-cropped` in the rings PNG
  - Split the single combined output PNG into two files per recording: `{stem}_ideas_demo.png` (main) and `{stem}_rings_demo.png` (new)
  - Main PNG: row 2 (frame panels) expanded from 3 panels (spike-1/spike/spike+1) to 6 panels (spike-1 → spike+4); DBSCAN cluster shading stays on just the analysis frame (`ai`); DBSCAN settings moved out of the figure suptitle into a bold row-level title above the panel row
  - New rings PNG: row 1 = 9 panels (spike-4 → spike+4), same fixed ring/circle overlay drawn on every panel (mask doesn't change frame-to-frame, by design); row 2 = full-segment inner/outer ring z-score trace with a shaded band annotating which x-range the 9 panels above cover
  - Renamed the confusing `[ai]` panel tag to `[DBSCAN frame]`
  - Converted px-only labels to µm in both PNGs (ring R/split/inner/outer boundaries, EPS setting text)
  - Idea 1 B+D% plot x-axis changed to be relative to the spike frame (`0` = spike), instead of absolute frame index

## Summary of current progress
- Clarified with user that R was previously RMS-based (radius of gyration), not the literal enclosing-circle radius they expected — switched to `dists.max()` per their explicit choice; confirmed the R/√2 inner/outer split and outer-bound-at-R now happen automatically as a consequence
- Confirmed the eps px↔µm conversion (`eps_and_min_samples()`) is correct and consistent per objective (10X→7px, 60X→45px for EPS_UM=10µm) — noted but left open: `int()` truncates rather than rounds
- Confirmed the ring/circle masks are computed once from the single analysis frame and reused unchanged across all frames when building the trace (same fixed-mask pattern as `RegionAnalyzer.get_temporal_traces()`) — flagged as a known limitation for actual spreading/moving signals
- Major PNG layout redesign per user's 3-point spec: wider frame-panel row in the main PNG, and a new dedicated rings PNG with a 9-panel context row + full-trace row with window annotation — both verified visually against real data after implementation
- Real-data check surfaced a concern: `2025_10_13-0029` (60X) showed R=76µm — nearly the full FOV — suggesting outlier pixels may still inflate max-based R; not chosen as a follow-up TODO this session (user declined when offered)
- New idea raised for later: the B+D% trace (already computed) could be used to estimate how long an ACh release event lasts, not just whether one occurred

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fixed R computation in ring analysis (RMS → max distance / enclosing-circle radius)
- ✅ Bounded the outer ring at R (automatic consequence of the R fix)
- ✅ Added boundary-touch detection/flagging for truncated clusters
- ✅ Confirmed eps px↔µm conversion is correct
- ✅ Restructured export PNGs per user spec (main: 6-panel row + DBSCAN settings row-title; new rings PNG: 9-panel row + annotated full trace)
- ✅ Converted px-only display labels to µm
- ✅ Made Idea 1 B+D% plot x-axis relative to spike frame (0 = spike)
- ✅ Fixed confusing `[ai]` panel label → `[DBSCAN frame]`

## What should we do next? (TODOs)
- [ ] **Integrate DBSCAN pipeline into `RegionAnalyzer`** — move the validated `lookup_obj`/`eps_and_min_samples`/`run_dbscan_filtered`/`compute_ring_traces` logic from `_demo_dbscan_tmp.py` into `classes/region_analyzer.py`, replacing the largest-CC approach. User confirmed this is the last remaining piece before the demo work is "finished."
- [ ] **Update `docs/dbscan_notes.md`** — currently stale: still describes R as RMS-based and "buggy," doesn't mention today's max-distance fix, boundary-crop flag, two-PNG split, or µm conversion
- [ ] **Detection criterion (B+D% vs baseline)** — implement the B+D% spike-frame vs baseline comparison as the ACh event detection gate in `RegionAnalyzer`; new idea to fold in: also estimate release *duration* from how long B+D% stays elevated above baseline, not just whether an event occurred
- [ ] **Add a "clipped segment" row to the main PNG** (`_demo_dbscan_tmp.py`, `save_main_figure`) — new row between the full B+D% trace (row 0) and the frame-panel row: same B+D% line, but x-axis (and y-axis, auto-scaled) restricted to just the spike-1→spike+4 window shown in the panels below, with the same spike-1/spike/spike+1 dashed markers + DBSCAN-frame star. Deferred — spec confirmed via clarifying questions, not yet implemented.

## Last Session Recap
※ recap: Fixed R in ring analysis (RMS → max/enclosing-circle distance, per user's explicit choice) and confirmed the outer-ring bound is now automatic; restructured the demo's export PNGs into two files (main: 6-panel spike-1→+4 row with DBSCAN settings as a row-title; new rings PNG: 9-panel spike-4→+4 row with fixed ring overlay + annotated full trace); converted px-only labels to µm; made the B+D% plot's x-axis relative to spike (0=spike). Confirmed eps px→µm conversion is correct. Next: integrate the validated DBSCAN pipeline into `RegionAnalyzer` (last piece), update stale `docs/dbscan_notes.md`, implement the B+D%-vs-baseline detection criterion (plus estimating release duration), and add a zoomed "clipped segment" row to the main PNG (spec confirmed, not yet coded).

---

# Log of the project progress 2026-06-30 Mon (Session 39)
Last working file: `_demo_dbscan_tmp.py`
Last working line: ~175 (`main()`)

## List of modified files
- `_demo_dbscan_tmp.py` — multiple changes this session:
  - Fixed CAT colormap: dim → gray (`#888888`), bright → white (`#ffffff`), bg → black (`#000000`); previous cyan/yellow was misleading
  - Restored cluster fill overlay (`CLUSTER_RGBA`) in `_draw_cluster_shading` — was accidentally removed in a bad intermediate edit
  - Replaced hardcoded `EPS=20px` / `MIN_SAMPLES=50` with physically-grounded constants: `EPS_UM=10.0µm` (inter-varicosity gap, tunable) + `MIN_DENSITY_FRAC=0.1`; added `PIXEL_SCALE` dict and `eps_and_min_samples(obj)` converter
  - Added `lookup_obj(tif_path)` — queries `rec_data.db` for the objective of each recording (`REC_{date}` table, `Filename = {date}-{serial}.tif`); OBJ now shown in PNG title alongside `EPS_UM` and computed `eps_px`
  - Added `import sqlite3` and `REC_DB` path constant
  - `run_dbscan_filtered()` now takes `eps_px` / `min_samples` as arguments instead of reading module-level globals
  - `save_figure()` now takes `obj`, `eps_px`, `min_samples` as arguments; suptitle shows `OBJ=60X  EPS=10µm=45px`
  - Summary table updated to show OBJ and eps_px per recording

## Summary of current progress
- Clarified that EPS_UM=10µm is the right framing (inter-varicosity *gap* distance, not axon span or varicosity size) — varicosity diameter (0.6µm, Umbriaco 1994) is sub-resolution and irrelevant as a spatial anchor; Gaussian sigma=6px is the blob scale in pixel space but EPS is a physical gap so must be converted per OBJ; axon span (~300µm) is the cluster size upper bound, not EPS
- Corrected the sigma-based EPS reasoning: the kernel half-width (3σ=18px) is the right pixel-space scale, but since EPS represents a physical gap, it must still be defined in µm and converted per OBJ (18px = 4µm at 60X, 24µm at 10X — inconsistent)
- First per-OBJ run: EPS_UM=10µm → 45px at 60X, 7px at 10X; results look plausible but not yet visually validated by user
- Identified bug in `compute_ring_traces`: R = RMS distance of cluster pixels from centroid — inflated by scattered outlier pixels far from the dense core; makes rings much larger than the visual cluster (e.g. 0029: visual cluster ~50px radius but R=109px); a percentile-based R (e.g. 90th percentile) would be more robust

## Completed TODOs (from Session 38)
- ✅ DBSCAN parameters made OBJ-consistent via physical-unit EPS_UM + per-recording px/µm conversion (was hardcoded EPS=20px regardless of OBJ)

## What should we do next? (TODOs)
- [ ] **Validate / tune EPS_UM=10µm** — review the per-OBJ output PNGs (just generated, not yet inspected by user); decide if 10µm gap is correct or needs adjusting
- [ ] **Fix R computation in ring analysis** — switch from RMS to 90th-percentile distance so R reflects the actual cluster extent without outlier inflation; also consider capping outer ring at R (currently all cluster pixels beyond R/√2 go to outer regardless of distance)
- [ ] **Integrate DBSCAN into RegionAnalyzer** — replace the current largest-CC approach with the validated DBSCAN pipeline from `_demo_dbscan_tmp.py`; `lookup_obj` / `eps_and_min_samples` logic will move into `RegionAnalyzer.__init__` (already has `self.obj` and `self.pixel_per_um`)
- [ ] **Detection criterion (B+D% vs baseline)** — implement the B+D% spike-frame vs baseline comparison as the ACh event detection gate in `RegionAnalyzer`

## Last Session Recap
※ recap: Design + demo-tuning session — corrected CAT colormap (gray/white), restored cluster fill overlay, and made DBSCAN parameters physically consistent via `EPS_UM=10µm` with per-OBJ pixel conversion (OBJ now looked up from `rec_data.db` and shown in PNG title). Key conceptual clarification: EPS is the inter-varicosity *gap* (physical distance → needs µm→px conversion), not varicosity size (sub-resolution) or axon span (cluster size upper bound). Also identified that R in the ring analysis is inflated by outlier pixels (RMS → should be percentile-based). Next: validate EPS_UM=10µm visually, fix R, then integrate DBSCAN into RegionAnalyzer.

---

# Log of the project progress 2026-06-28 Sat (Session 38)
Last working file: `functions/plot_results.py`
Last working line: ~210 (end of `_plot_frame_panel`)

## List of modified files
- `functions/plot_results.py` — `_plot_frame_panel` redesigned: removed x/y span crosshair and centroid dot overlay (`show_centroid=False, show_span=False`); now shows per-frame B/D/B+D area in µm² + % (e.g. `B: 1234 µm² (2.3%)`) for all 9 panels — µm² allows cross-OBJ comparison (% alone is not comparable across 10X/40X/60X FOVs); legend simplified to bright contour only; `_plot_trace_panel` no longer calls `region_analyzer.get_peak_latency_ms()` — latency is now computed inline from already-computed `bright_peak_rel`/`dim_peak_rel`, removing the redundant 3rd call to `get_temporal_traces`

## Summary of current progress
- Simplified row-1 panel titles: removed x/y spans and centroids ("for now"), showing µm² + % of each frame directly counted from `cat_frame` pixels — not restricted to spike-frame mask, so area changes naturally reflect spreading/shrinking across frames
- Identified and fixed a triple-`get_temporal_traces` redundancy: spike_analysis.py called it once (for DB latency), `_plot_trace_panel` called it twice more (once for trace plot, once inside `get_peak_latency_ms` for the title); fixed by inlining latency from already-computed peak indices — neither option meaningfully speeds up the pipeline (temporal traces are negligible vs TIFF I/O), but removes unnecessary duplication
- Extensive design discussion on flow/spreading detection in `RegionAnalyzer` — identified that the current largest-CC approach fails for two real patterns seen in data:
  - **Scenario A** (one release site, scattered small islands): no single CC survives `min_area_um2=900`, event missed entirely despite visible increased pixel density in spike frame
  - **Scenario B** (multiple distinct varicosities releasing simultaneously): single centroid falls in empty space between sites, meaningless
- Rejected: Gaussian smoothing on MED to find release center — MED is already Gaussian-filtered upstream; double-smoothing degrades spatial resolution
- Viable for Scenario A only: intensity-weighted centroid of all bright pixels using MED z-scores as weights — finds center of mass of scattered islands without CC size filter; fails for Scenario B
- User confirmed Scenario B is real (showed TIFF_0029 ABF_0026: white dots scattered across whole frame, none detected despite visible spike-frame elevation)
- **DBSCAN identified as the right approach** for both scenarios: treats bright pixels as a 2D point pattern, finds dense clusters automatically (no assumed number of clusters), discards noise pixels (`label=-1`), gives per-cluster centroid + area; key parameters `eps` (neighbourhood radius in pixels) and `min_samples` (minimum cluster size) need tuning against real data
- Detection criterion (user's Point 1): compare B+D% at spike frame vs baseline frames — the only metric that works for both Scenario A and B without needing cluster structure first

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Simplified row-1 panel titles (removed spans/centroids, added µm² + %)
- ✅ Fixed redundant 3rd `get_temporal_traces` call in `_plot_trace_panel`

## What should we do next? (TODOs)
- [ ] **RegionAnalyzer redesign (broad)** — overall redesign to handle delayed signals (+1 frame tolerance), multi-cluster (Scenario B), and flow/spreading; current largest-CC approach is too brittle for real data patterns; carried from Session 37
- [ ] **DBSCAN cluster detection** — implement DBSCAN-based high-density area detection in `RegionAnalyzer` to replace/augment largest-CC approach; `eps` and `min_samples` need tuning against real data; handles both island clusters (Scenario A) and multiple distinct varicosities (Scenario B)
- [ ] **Detection criterion** — implement B+D% at spike frame vs baseline comparison as the ACh event detection criterion; currently the only approach that robustly detects both Scenario A and B without requiring cluster structure
- [ ] **Per-cluster spreading analysis** — once DBSCAN gives clusters, track B+D area growth per cluster across post-spike frames to characterize spreading from each release site

## Last Session Recap
※ recap: Simplified row-1 panel titles to show B/D/B+D in µm² + % per frame; fixed a redundant triple-call to `get_temporal_traces`. Main session was design discussion for flow/spreading detection — identified two real failure modes of the current largest-CC approach (island clusters + multiple varicosities), confirmed by real data; rejected Gaussian on MED (already filtered upstream); landed on DBSCAN as the right approach for finding high-density clusters in the bright pixel point pattern. Next: implement DBSCAN in RegionAnalyzer + detection criterion (B+D% vs baseline).

---

# Log of the project progress 2026-06-27 Fri (Session 37)
Last working file: `classes/region_analyzer.py`
Last working line: 243 (end of file — read only, no edits this session)

## List of modified files
- (none — design discussion session only)

## Summary of current progress
- Discussed how spike-centered alignment in `spike_centered_median`/`spike_centered_avg` already uses the spike's biological significance (spike timing as the anchor for the ACh release event)
- Reviewed the current `RegionAnalyzer` design: anchors bright/dim detection to the spike frame only via `_find_largest_category` (bright) and `_find_all_dim` (dim)
- Identified key bottleneck: ACh signal can be delayed 1–2 frames from the spike frame, so spike-frame-only anchor may miss or underestimate the real core zone
- Explored pixel-vote approach (count how many post-spike frames label each pixel as bright/dim) — captures 1-2 frame delay but user raised concern it is unreliable with only 3 frames
- Key insight: bright and dim should NOT be treated as fully independent regions — they form one spatially connected ACh event; isolated dim pixels are only meaningful if spatially connected to a bright+dim blob
- Edge case identified: isolated dim pixel in frame N that connects to main blob in frame N+1 → could be the leading edge of spreading signal, not noise — single-frame spatial analysis can't resolve this
- Direction emerging: use union of bright+dim blobs across post-spike frames to define the core zone
- User also has a new idea for temporal traces — not yet shared, to be discussed in new session

## Completed TODOs/Tasks (before new wrap-up)
- (none — design discussion only, no code changes)

## What should we do next? (TODOs)
- [ ] **RegionAnalyzer redesign** — define core zone without anchoring to spike frame only; signal can be delayed 1–2 frames (biologically plausible); delay >2 frames unlikely to be spike-driven per user's hypothesis
- [ ] **Handle edge case** — isolated dim pixel in frame N that connects to main blob in frame N+1 is likely a leading edge, not noise; need a temporal approach that captures this
- [ ] **New idea for temporal traces** — user has a new design in mind, to be shared and implemented in new session

## Last Session Recap
※ recap: Design-only session (no code changes) — identified that current `RegionAnalyzer` anchors bright/dim detection to spike frame only, missing 1–2 frame delayed ACh signals; key insight that bright+dim should be treated as one connected spatial event, not independently; edge case (isolated dim connecting to main blob in next frame) flagged; new temporal trace idea and core zone redesign to continue in next session.

---

# Log of the project progress 2026-06-23 Tue (Session 36)
Last working file: `functions/query_databases.py`
Last working line: 222 (`metric_cols = ["bright_area_um2", "dim_area_um2", "total_area_um2", "peak_latency_ms"]` in `compute_region_stats`)

## List of modified files
- `functions/zscore_img_segs.py` — each segment cast to `float32` right after reading from the TIFF, before any mean/std/normalize math; numpy has no native float16 arithmetic and was emulating it in software, ~6x slower than float32 (benchmarked before committing to the fix)
- `functions/spike_centered_processes.py` — `spike_centered_median`'s `.astype(np.float32)` on the stacked array now passes `copy=False`; segments are already float32 after the fix above, so this avoids a second full-array copy on top of `np.stack`'s own copy
- `spike_analysis.py` — added a startup diagnostic log (`CPUs available to this job: N (numba NUMBA_NUM_THREADS=N)`) via `os.sched_getaffinity`/`numba.config`, after diagnosing a "no multicore speedup on deigo" report back to a missing `--cpus-per-task` in the user's `srun` call; renamed `_build_stats_report` → public `build_stats_report`; added `write_stats_report(ana_list_path, results_db_path)` + `_strip_existing_report()`/`_STATS_BLOCK_MARKER` so re-running the stats step overwrites its own previous block instead of stacking duplicates (verified by running twice on a test file — only one block remained)
- `classes/region_analyzer.py` — fixed a real bug in `get_temporal_traces()`: undetected bright/dim regions defaulted to an all-zero trace instead of all-NaN, so `_nanargmax_relative`'s only "not found" check (`np.all(np.isnan(...))`) never triggered — `get_peak_latency_ms()` silently returned a fake latency instead of `None` whenever either region wasn't actually detected. Now defaults to `np.full(n_frames, np.nan)`
- `functions/query_databases.py` — `compute_region_stats()` gained a `total_area_um2` metric (`bright_area_um2 + dim_area_um2` per cell), null wherever dim wasn't detected (same "don't fabricate a measurement" principle as the latency fix)
- `append_stats.py` — **new**, promoted from the throwaway `_append_stats_tmp.py`: derives `results.db`'s path from the ana list's own footer (no separate `--results_db` arg needed), reuses `spike_analysis.build_stats_report`/`write_stats_report` instead of duplicating logic
- `_fix_latency_db_tmp.py` — **new** (throwaway, still present, committed): nulls out `peak_latency_ms` for rows where `has_bright_region=0 OR has_dim_region=0`. Run against the local `results/results.db` (140/173 rows fixed) — **not yet run against the real `Z:/Kang/Cluster/results/results.db`**, by user's own account
- `_append_stats_tmp.py` — deleted (superseded by `append_stats.py`)
- `data/ana_list_20260618_000.txt`, `data/ana_list_20260622_000.txt` — updated by running the real pipeline / `append_stats.py` against real data during testing

## Summary of current progress
- Root-caused and fixed a real performance complaint ("z score normalization still fucking slow") down to numpy's lack of native float16 arithmetic — benchmarked the fix (~6x) before committing, then found and fixed a second, smaller redundant-copy issue in `spike_centered_median` once segments were already float32
- Diagnosed a "no multicore acceleration on deigo" report down to a missing `--cpus-per-task` in the user's `srun` call (SLURM cgroups cap visible CPUs independent of the node's physical core count) — checked `sinfo`/`scontrol`/`sacctmgr` to confirm no hard per-job CPU/mem caps existed, then added a startup diagnostic log so this is visible immediately on every future run without a manual check
- Walked through `rsync`/Windows-copy-tool questions (verbose flag meaning, trailing-slash semantics, robocopy `/MT`) while the user moved `results/` to bucket storage — no code changes, just CLI guidance
- Found and fixed a real correctness bug in `RegionAnalyzer.get_temporal_traces()`: undetected regions produced fake non-`None` peak-latency values instead of correctly reporting "not measured" — fixed the code (NaN instead of zero defaults) and repaired the local `results.db` (140/173 bogus rows nulled); the real cluster/bucket database still needs the same repair
- Promoted the throwaway `_append_stats_tmp.py` into a proper `append_stats.py` per user request, fixing a duplicate-report-block bug the user caught along the way (`write_stats_report()` now overwrites instead of appending blindly) — shared between `run()` and the new script so neither duplicates
- Added a `total_area_um2` (bright+dim) stat per user request, consistent with the same "null when not detected" semantics as the latency fix
- When asked to confirm carry-over TODOs, user explicitly declined to flag the real-DB latency fix or throwaway-script cleanup as open items this session — only the GUI Results Browser tab was confirmed as still wanted
- User committed all of today's work (`01e86fa "debug and speed up again"`, `0a500ce "some inprovement and fixing dim latencies"`) — working tree is fully clean, nothing left uncommitted

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fixed float16 slowdown in `zscore_img_segs` (cast to float32 immediately after reading)
- ✅ Fixed redundant copy in `spike_centered_median` (`astype(..., copy=False)`)
- ✅ Diagnosed and fixed "no CPU parallelism on deigo" (missing `--cpus-per-task`; added a startup diagnostic log)
- ✅ Fixed `RegionAnalyzer.get_temporal_traces()` zeros-vs-NaN bug causing fake peak-latency values
- ✅ Repaired local `results/results.db` (140/173 bogus `peak_latency_ms` rows nulled)
- ✅ Promoted `_append_stats_tmp.py` → formal `append_stats.py`, fixed a duplicate-report-block bug along the way
- ✅ Added `total_area_um2` metric to `compute_region_stats()`

## What should we do next? (TODOs)
- [ ] **Start Step 3**: a GUI "Results Browser" tab — browse `results.db` + view `region_sta/`/`full_traces/` PNGs; `compute_region_stats()`/`get_cell_recording_status()` are already built and proven, ready to wire into a new `views/view_results_browser.py` + `controllers/ctrl_results_browser.py`. (Note: the `.claude/plans/kind-mapping-codd.md` plan doc referenced in earlier sessions' logs no longer exists on disk — `.claude/plans/` is currently empty — so this needs a fresh plan before starting, not a resume of an old one.)

## Last Session Recap
※ recap: Fixed a real float16 perf bug in `zscore_img_segs` (~6x speedup) and a redundant-copy issue in `spike_centered_median`; diagnosed missing `--cpus-per-task` on deigo SLURM jobs; fixed a `RegionAnalyzer` bug producing fake peak-latency values for undetected regions (local `results.db` repaired, real cluster DB still pending); promoted `append_stats.py` to a formal script (fixing a duplicate-block bug along the way) and added a `total_area_um2` stat. Next: GUI Results Browser tab (Step 3) — needs a fresh plan, the old one is gone.

---

# Log of the project progress 2026-06-22 Mon (Session 35)
Last working file: `classes/abf_clip.py`
Last working line: 111 (`ws_collapsed = wb.create_sheet("Collapsed_Peaks")` in `_export_spike_xlsx()`)

## List of modified files
- `classes/abf_clip.py` — when `set_interval_frames < 1` (every spike's segment would collapse to the spike frame alone, no baseline), now skips those spikes instead of letting them reach `SpatialCategorizer` and crash on `np.concatenate([])`; added a pre-processing dedup step in `get_available_spiking_frames()` that collapses multiple electrophysiological spikes landing in the same image frame (burst firing faster than the 1/fs_imgs camera shutter) down to the first (onset) peak per frame, before any gap/margin math runs — keeps the true spike record in `df_peaks`/the `Peaks` xlsx sheet untouched, only affects which frame is used for windowing; collapsed duplicates are recorded in a new `df_collapsed_peaks` and exported to a new `Collapsed_Peaks` sheet in the per-recording xlsx
- `spike_analysis.py` — when an entry has no valid segments, the skip is now also appended to `ana_list_*.txt` (not just console), so it's visible in the persisted run record
- `functions/spike_centered_processes.py` — `_cpu_median_axis0`'s internal buffers and the `stacked` array cast changed from `float64` to `float32`; source GAUSS/ALS tiffs are saved as `float16` so `float64` was 4x more memory than the data ever carried for no precision gain (confirmed `float16` itself can't be fed to numba — raises `NotImplementedError`, tested directly). Fixes an OOM `Killed` on the cluster for a 120-segment recording
- `docs/diagram_set_interval_frames_collapse.md` — new ASCII-diagram writeup explaining why a few colliding/tightly-clustered spikes can collapse the analysis window for an entire recording (one shared `set_interval_frames` value derived from the *mode* of all gaps)

## Summary of current progress
- Root-caused and fixed a real cluster crash: `SpatialCategorizer._calculate_global_thresholds()` crashed on `np.concatenate([])` when a recording's spikes were clustered tightly enough that the auto-derived `set_interval_frames` came out to `0` (1-frame segments, no baseline). Traced the full mechanism back to `abf_clip.py`'s mode-based window derivation before fixing it
- Root-caused and fixed a separate `Killed` (OOM) crash in `spike_centered_median()` — verified via direct numba compile tests (not assumed) that `float16` isn't usable in the kernel and that the existing `float64` upcast was an unnecessary 4x memory blow-up over the actual `float16` source data; switched to `float32`
- Per user's request, implemented burst-spike deduplication in `AbfClip` (first peak per frame wins, recomputes margins from the deduplicated list) instead of relying on `find_peaks(distance=...)`, preserving the true electrophysiological spike record while fixing the windowing collision — verified the dedup logic against synthetic frame-index data before committing to the approach
- Audited the full `spike_analysis.py` call graph for other unnecessary `float64` usage — found none in the live pipeline (one unused sibling function, `spike_centered_avg()`, still has the same pattern but isn't called from this pipeline)
- User committed all of today's work (`3dd6b06 fix collision of peaks situation (gap 0)`, plus the earlier `2133c7c basic statistics completely` for the `ana_list_*.txt` logging change) — working tree is fully clean, nothing left uncommitted

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fixed `SpatialCategorizer` crash on zero-baseline segments (skip in `AbfClip` instead of crashing downstream)
- ✅ Fixed OOM `Killed` crash in `spike_centered_median()` (`float64` → `float32`)
- ✅ Implemented first-peak-per-frame deduplication for burst-firing spike collisions, with `Collapsed_Peaks` xlsx logging
- ✅ Audited pipeline for other `float64` memory issues (none found in the live call graph)

## What should we do next? (TODOs)
- [ ] **Start Step 3**: the GUI "Results Browser" tab from `.claude/plans/kind-mapping-codd.md` — browse `results.db` + view `region_sta/`/`full_traces/` PNGs; `compute_region_stats()`/`get_cell_recording_status()` are already built and proven, ready to wire into a new `views/view_results_browser.py` + `controllers/ctrl_results_browser.py`

## Last Session Recap
※ recap: Fixed two real cluster crashes — a `SpatialCategorizer` crash on zero-baseline (too-tightly-clustered) spikes, and an OOM `Killed` in `spike_centered_median()` (unnecessary `float64` upcast over `float16` source data). Also added burst-spike deduplication in `AbfClip` (first-peak-per-frame, with new `Collapsed_Peaks` xlsx logging) instead of changing peak-detection `distance`, to preserve the true spike record. All committed (`3dd6b06`, `2133c7c`). Next: build the GUI Results Browser tab (Step 3, plan already written).

---

# Log of the project progress 2026-06-21 Sun (Session 34)
Last working file: `spike_analysis.py`
Last working line: 321 (`f"[green]✓ Exported {dir_names}/, region_sta/, full_traces/ ...` in `run()`)

## List of modified files
- `classes/abf_clip.py` — fixed a real crash hit on the cluster (`2024_12_19_0012`): `get_available_spiking_frames()`'s `inter_spike_frames` could go negative when two detected spikes floor-divide to the same/adjacent frame index, which could make `set_interval_frames` itself negative and invert `left_bound`/`right_bound` into an empty frame range — `zscore_img_segs` then collapsed `np.mean`/`np.std` on a 1-D empty array to a scalar and crashed on `baseline_std[baseline_std == 0] = 1`. Fixed with one `np.clip(inter_spike_frames, 0, None)` (root cause, not a band-aid — verified numerically against the exact collision scenario before/after); also converted the remaining `console.print` calls in this file (and `functions/query_databases.py`, `functions/spike_centered_processes.py`) to `console.log` for consistency with the rest of the pipeline
- `functions/plot_results.py` — `_plot_trace_panel`'s `window` param is now `int | None` (`None` = don't crop the x-axis); new `plot_full_trace()` — standalone single-panel export figure of the same bright/dim/total trace data, spanning the *entire* segment instead of the ±4-frame window used in `plot_spatiotemporal_summary()`'s row-2 panel
- `functions/__init__.py` — registered `plot_full_trace` in the lazy-import table
- `spike_analysis.py` — wired `plot_full_trace()` into `run()`, exported to a new `results/full_traces/` folder (filename stem uses `file_type="TRACE"`, same naming convention as `region_sta`'s `"SPATIAL"`)

## Summary of current progress
- Diagnosed and fixed a real production crash from the user's own cluster run — traced the exact mechanism (spike-frame collision → negative inter-spike margin → negative `set_interval_frames` → inverted bounds → empty array → scalar mean/std → crash) before writing the fix, then verified the clamp neutralizes the exact collision scenario with an isolated numeric check (not guessed/assumed fixed)
- Swept the remaining `console.print` calls in pipeline-adjacent files (excluding other tabs' controllers, which are out of scope) to `console.log`, per direct user request
- Added `plot_full_trace()` for the bright/dim/total temporal trace across the whole segment (previously only viewable cropped to ±4 frames inside the combined summary figure) — generated a real PNG from local test data (`2025_12_15-0013`) to confirm it renders correctly (flat baseline, sharp spike-aligned peak, decay) before considering it done; demo script/PNG deleted after
- User committed all of today's work (`60068e3 basic statistics completely`) — working tree is fully clean, nothing left uncommitted

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Root-caused and fixed the cluster crash on `2024_12_19_0012` (negative `set_interval_frames` from colliding spike frame indices)
- ✅ Converted remaining `console.print` → `console.log` in `abf_clip.py`/`query_databases.py`/`spike_centered_processes.py`
- ✅ Added `plot_full_trace()` + wired it into `run()`, exporting to `results/full_traces/`

## What should we do next? (TODOs)
- [ ] **Start Step 3**: the GUI "Results Browser" tab from `.claude/plans/kind-mapping-codd.md` — browse `results.db` + view `region_sta/`/`full_traces/` PNGs; `compute_region_stats()`/`get_cell_recording_status()` are already built and proven, ready to wire into a new `views/view_results_browser.py` + `controllers/ctrl_results_browser.py`

## Last Session Recap
※ recap: Fixed a real cluster crash in `abf_clip.py` (negative inter-spike margin could invert a frame range into an empty array), swept remaining `console.print`→`console.log` in pipeline files, and added `plot_full_trace()` (full-segment trace PNG, exported to `results/full_traces/`). All committed (`60068e3`). Next: build the GUI Results Browser tab (Step 3, plan already written).

---

# Log of the project progress 2026-06-21 Sun (Session 33)
Last working file: `spike_analysis.py`
Last working line: 160 (`"Per-Neuron Recording List (filename, detected) [detected/total]:\n"` in `_build_stats_report`)

## List of modified files
- `classes/region_analyzer.py` — added back `min_area_um2` constructor parameter (previously removed in an earlier session as an "undiscoverable magic number" — now reintroduced as an explicit, documented argument): filters candidates in `_find_largest_category()` before picking the largest, and skips undersized components in `_find_related_dim()` before merging into the combined dim region; added module-level `_nanargmax_relative()` helper and `get_peak_latency_ms(segment, spike_frame_idx, frame_duration_ms)` (extracted from the figure's inline calc, single source of truth for both the DB value and the figure title)
- `functions/plot_results.py` — `_plot_trace_panel` now calls `region_analyzer.get_peak_latency_ms()` instead of computing latency inline; removed the now-duplicate local `_nanargmax_relative` (imports the one in `region_analyzer.py`)
- `classes/results_exporter.py` — wired the previously-unused `plot_spatiotemporal_summary()` figure into real exports (renamed its output folder `spatials`→`region_sta`); promoted `_build_export_stem`→public `ResultsExporter.build_export_stem()` staticmethod and `_derive_site_code`→public `ResultsExporter.derive_site_code()` staticmethod (both needed cross-module reuse from `spike_analysis.py`); added `bright_area_um2`/`dim_area_um2`/`dim_x_span_um`/`dim_y_span_um`/`peak_latency_ms`/`med_filename` columns to `results.db`'s `experiments` table
- `classes/abf_clip.py` — per-entry spike-detection xlsx now saved under a new `results/xlsx/` subfolder instead of loose at `results/` root
- `functions/query_databases.py` — three new DB-query functions: `compute_region_stats()` (mean±std of bright/dim area + peak latency, averaged per unique cell, cells with no detected bright region excluded), `get_excluded_recordings()` (which images/cells were excluded), `get_cell_recording_status()` (per-recording detection status, used to build the per-neuron file list); all three share one `_bright_excluded_expr()` predicate so they can't disagree on what counts as "excluded"
- `functions/__init__.py` — registered the 3 new query functions in the lazy-import table
- `spike_analysis.py` — `RegionAnalyzer(...)` now called with `min_area_um2=400`; new `_format_neuron_line()` + `_build_stats_report()` build a "Region Analysis Statistics" text block (neurons-detected summary, mean/std/n_detected table, per-neuron `(filename, detected)` breakdown with line-wrapping for multi-recording neurons) appended to the bottom of the ana_list file at the end of every `run()`; the spike-frame overlay figure (`plot_spatiotemporal_summary`) is now actually exported per entry; cell-summary xlsx moved under `results/xlsx/`
- `data/ana_list_20260605_000.txt` — user ran the real pipeline on this file mid-session (their own action, not mine) — confirms the appended report renders correctly in production; that particular run showed 0/2 neurons detected (not investigated further this session, by request)
- Committed mid-session by the user as `82c8305` (everything up through the `min_area_um2`/`region_sta` rename/`xlsx/` folder work); the DB-query-functions + `med_filename` + report-formatting work above is **not yet committed**

## Summary of current progress
- Resumed from Session 32 by answering a design question (does `RegionAnalyzer` use dilation/erosion for noise removal? — no, that's upstream in `SpatialCategorizer._apply_morphological()`; `RegionAnalyzer` only picks the largest connected component), which led directly into reintroducing `min_area_um2` as a real, documented filter parameter
- Discovered and closed Step 1/2 gaps from the postponed `.claude/plans/kind-mapping-codd.md` plan: the spike-frame overlay figure was computed but never exported by the real pipeline, and `results.db` had no flat columns for dim-region size, bright/dim area, or peak latency
- Built and proved `compute_region_stats()` against a **real** pipeline run (not synthetic) — used local test data (`2025_12_15-0013/0014/0024/0025` + paired ABFs) to run the actual pipeline end-to-end, producing 2 real cells (`neoChAT-676` × `3R`/`4R`, `CELL_1`) with real bright/dim/latency numbers; all demo scripts and the temporary results dir were deleted after verification (project convention: throwaway diagnostic scripts get deleted once their job is done)
- Iterated the appended-report format through 2 rounds of user feedback: first replaced a flat "Excluded images/cells: N" summary with a per-neuron `(filename, detected)` breakdown (user wanted to know *which* recordings failed, not just a count) — required adding a persisted `med_filename` column since the filename couldn't be reconstructed from `results.db` alone; then fixed spacing (blank line after `dir_results:`) and added line-wrapping + a header for multi-recording neurons, matching a screenshot the user provided of their real output
- User independently ran the real pipeline on `ana_list_20260605_000.txt` partway through the session and committed the in-progress work (`82c8305`) — confirms the report append mechanism works correctly against production data, though that specific run detected 0/2 neurons (flagged but explicitly not chosen as a follow-up TODO this session)

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Re-added `min_area_um2` to `RegionAnalyzer` (constructor + both finder methods), wired to `400` in `spike_analysis.py`
- ✅ Renamed export folder `spatials`→`region_sta`; added `results/xlsx/` subfolder for both xlsx outputs
- ✅ Wired the spike-frame overlay figure into the real export pipeline (was computed but never saved)
- ✅ Added `bright_area_um2`/`dim_area_um2`/`dim_x_span_um`/`dim_y_span_um`/`peak_latency_ms`/`med_filename` columns to `results.db`
- ✅ Built `compute_region_stats()`, `get_excluded_recordings()`, `get_cell_recording_status()` — verified against a real pipeline run, not just synthetic data
- ✅ Appended a "Region Analysis Statistics" report (summary line, metrics table, per-neuron detection breakdown) to the bottom of `ana_list_*.txt` after each run, with 2 rounds of formatting fixes per direct user feedback

## What should we do next? (TODOs)
- [ ] **Start Step 3**: the GUI "Results Browser" tab from `.claude/plans/kind-mapping-codd.md` — browse `results.db` + view `region_sta/` PNGs; `compute_region_stats()`/`get_cell_recording_status()` are already built and proven, ready to wire into a new `views/view_results_browser.py` + `controllers/ctrl_results_browser.py`

## Last Session Recap
※ recap: Re-added `min_area_um2` filtering to `RegionAnalyzer`, wired the previously-unused overlay figure into real exports (renamed `spatials`→`region_sta`), and added 6 new `results.db` columns (areas, peak latency, `med_filename`). Built and proved (against a real pipeline run) 3 new stats functions, then wired a per-neuron detection report into the bottom of `ana_list_*.txt`, iterated twice on its format per user feedback. Next: build the GUI Results Browser tab (Step 3, plan already written).

---

# Log of the project progress 2026-06-21 Sun (Session 32)
Last working file: `classes/results_exporter.py`
Last working line: 330 (`zscore_range: tuple[float, float],` param in `_upsert_record`)

## List of modified files
- `classes/results_exporter.py` — major resync + redesign: discovered `export_all()`/`_upsert_record()` were stale against the Session 29 `RegionAnalyzer` redesign (still assumed a per-frame list of `bright_largest` dicts and `n_frames`/`total_dim_regions`/`total_bright_regions` — would have crashed if called). Fully reworked: new compact filename code via `_build_export_stem()`/`_derive_site_code()` (`{exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_{TYPE}`), new `build_animal_index_map()` staticmethod (batch-local 1-based animal index), flat `results/{category}/` folder layout (dropped the per-date subfolder — `exp_date` is already in every filename), `SLICE` column fixed `INTEGER`→`TEXT`, `n_frames`/`total_dim_regions`/`total_bright_regions` replaced with `has_bright_region`/`has_dim_region`, dropped the now-redundant `data_dir` column, added a real `ANIMAL_ID` column (separate from the batch-local `A{n}` filename index) and `zscore_min`/`zscore_max` columns (the median stack's z-score range, previously only printed to console), `export_figure()` fixed to call `figure.savefig()` (was calling the wrong `.save()` API) and now creates its target folder on demand instead of upfront
- `spike_analysis.py` — extracted the `__main__` pipeline body into a callable `run(ana_list_path, detrend_mode, use_als, db_path, exp_db_path, emitter=None)`, mirroring `img_proc.py`'s established pattern so the GUI can call it; wired `ResultsExporter.export_all()` into the per-row loop (replacing the old inline `_MED.tif`/`_CAT.tif` saves); all `console.print` → `console.log`; added per-step/per-entry/total elapsed-time reporting (mirrors `img_proc.py`'s timing style)
- `controllers/ctrl_align_spike.py` — wired `btn_run_analysis.clicked` → `BackgroundWorker` → `spike_analysis.run()`, routing progress into `le_current_total`/`le_status` (mirrors `ctrl_img_proc.py`'s `start_processing()`); `export_ana_list()` now stores `self._ana_list_path` so the run button knows what to run
- `classes/region_analyzer.py` — removed dead `find_largest_regions()` method (verified zero call sites anywhere in the repo) and its docstring mention
- `classes/plot_results.py` → **moved to `functions/plot_results.py`** (file had only pure `Figure`-returning functions left after class cleanup, matching the project's classes-vs-functions convention) — also deleted 4 unused/broken GUI viewer classes (`PlotPeaks`/`PlotSegs`/`PlotSpatialDist`/`PlotRegion`; two were already broken against the current `RegionAnalyzer` API, none had any live caller) and their dead support code (`CustomToolbar`/`WindowToolbar`/`center_on_screen`)
- `classes/mpl_canvas.py` — **new**: `MplCanvas` moved out of `plot_results.py` into its own file (still actively used by `views/view_als_correct.py`)
- `classes/__init__.py`, `functions/__init__.py` — updated lazy-import tables for the moves/deletions above
- `CLAUDE.md`, `AGENTS.md` — updated the "New plots" rule to point to `functions/plot_results.py` (was `classes/plot_results.py`)
- Deleted stale `results/results.db` (42-row old-schema test DB, dated 2026-02-17, predated all tracked sessions)

## Summary of current progress
- Picked up Session 30's #1 priority backlog item (compact export filename + folder structure for `ResultsExporter`), but discovered along the way that `ResultsExporter` was completely out of sync with the current `RegionAnalyzer` API — fixed both in the same pass, verified end-to-end against real local test data (`2025_12_15-0013_BIEXP_GAUSS_*.tif`)
- Wired `btn_run_analysis` (a TODO carried since Session 25) by extracting `spike_analysis.py`'s pipeline into a callable `run(emitter=None)`, exactly mirroring `img_proc.py`/`ctrl_img_proc.py`'s existing `BackgroundWorker` pattern
- Cleaned up `classes/plot_results.py` end-to-end: deleted 4 dead/broken GUI classes, split `MplCanvas` into its own file, then moved the remaining pure-function module to `functions/` — fully resolves the long-standing "forces PySide6 import for headless export" memory note
- User test-ran the pipeline on the **deigo** OIST HPC cluster — walked through `srun`/partition troubleshooting (`AssocMaxWallDurationPerJobLimit` on `short`, switched to `compute`, succeeded), then real-world testing surfaced 2 concrete design issues: unnecessary per-date folder nesting, and unused `regions`/`spatials` folders — both fixed (flattened layout; those 2 folders now created on-demand only, by `export_figure()`)
- Final round of fixes per direct user review of the remaining old backlog (each verified against current code before acting, not assumed): removed dead `find_largest_regions()`, added a real `ANIMAL_ID` column to `ResultsExporter`'s `experiments` table, and decided `ref_df` persistence is unnecessary now that `results.db` is the actual persistence layer (dropped from backlog)
- Last fix of the session: the median stack's z-score range (computed by `spike_centered_median()`) was only ever printed to console — added `zscore_min`/`zscore_max` columns so it's queryable from `results.db` like everything else

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Implement the compact export filename code (`A{n}S{slice}C{site}`) + flat folder structure in `ResultsExporter` (Session 30 priority #1)
- ✅ Wire `btn_run_analysis` in `ctrl_align_spike.py` to call the pipeline (carried since Session 25)
- ✅ Refactor `classes/plot_results.py` (split GUI from headless) — went further than planned, moved to `functions/`
- ✅ Decide fate of `RegionAnalyzer.find_largest_regions()` — removed (confirmed dead code)
- ✅ Add `ANIMAL_ID` column to `ResultsExporter`'s `experiments` SQLite table
- ✅ Decided: `ref_df` saving is unnecessary (`results.db` already serves that role) — dropped from backlog
- ✅ Save the median stack's z-score range (`zscore_min`/`zscore_max`) to `results.db`

## What should we do next? (TODOs)
- [ ] Design a way to quickly find/browse exported results (by animal/slice/site/date) — carried from Session 30, still untouched
- [ ] ROI + Otsu + `regionprops`-centroid soma detection on `EMI=RED` images, then contour-overlay export + centroid-to-soma distance calc — confirmed this session as the next priority after the browse/find feature

## Last Session Recap
※ recap: Resynced `ResultsExporter` to the current `RegionAnalyzer` API and implemented the compact filename/flat-folder export design; wired `btn_run_analysis` via a new `spike_analysis.run(emitter=None)`; cleaned up `plot_results.py` (deleted dead GUI classes, moved pure functions to `functions/`); fixed 2 issues found testing on the deigo cluster (folder nesting, unused subfolders); added `ANIMAL_ID` + `zscore_min`/`zscore_max` to the DB and removed dead code. Next: results browse/find feature, then soma detection.

---


Last working file: `spike_analysis.py`
Last working line: 120 (`write_cell_summary_xlsx(cell_df, cell_summary_path)` / confirmation print, in `__main__`)

## List of modified files
- `functions/query_databases.py` — **new**: `count_unique_cells(ref_df) -> pl.DataFrame` — groups `ref_df` by `(ANIMAL_ID, SLICE, AT)` (the cell-identity triple, e.g. `neoChAT-677, 2R, CELL_1`), aggregates each group's `Filename` values into a sorted `Filenames` list + `n_images` count, sorted for stable output
- `functions/xlsx_writer.py` — **new**: `write_cell_summary_xlsx(cell_df, output_path)` — writes a `count_unique_cells()` result to a single-sheet xlsx via `openpyxl` directly (not `polars.write_excel`, since that needs `xlsxwriter`, which isn't installed and wasn't added as a new dependency); joins each row's `Filenames` list into a comma-separated string since xlsx cells can't hold a real list
- `functions/__init__.py` — registered `count_unique_cells` (from `.query_databases`) and `write_cell_summary_xlsx` (from `.xlsx_writer`) in the lazy-import table + `__all__`
- `spike_analysis.py` — `__main__` now computes `cell_df = count_unique_cells(ref_df)` right after `ref_df`, prints `Found {N} entries ... -> {M} unique cells`, then immediately saves it via `write_cell_summary_xlsx` to `results_dir / f"{args.ana_list.stem}_cells.xlsx"` with a confirmation print — both before the per-row `AbfClip` processing loop
- Created and deleted a throwaway `_diag_at_check.py` mid-session to check `AT` column severity (SITE_* vs CELL_*) before building the real function — job done, removed

## Summary of current progress
- **Motivating question**: today's pick (`ana_list_20260618_000.txt`, 214 entries) doesn't mean 214 distinct cells — needed to know the real cell count and which images belong to which cell, plus first check how bad the known `AT = SITE_*` (vs `CELL_*`) data-quality issue actually is
- **Severity check (diagnostic, not code)**: for this specific 214-row pick, **0** rows have `AT = SITE_*` — all are clean `CELL_1`/`CELL_2`. Across the *whole* `rec_data.db`, though, `SITE_*` is 62.9% (1160/1844) of all `AT` values — every `REC_*` table before 2024-10-03 used `SITE_*` exclusively (the `CELL_*` convention didn't exist yet), and many dates after that still mix `CELL_*`/`SITE_*` within the same day. Also spotted one naming inconsistency: `REC_2025_04_03` has a stray `CELL1` (no underscore) instead of `CELL_1`
- User explicitly decided: no special-casing/guard needed for `SITE_*` going forward, since recordings are now made through the program and that ambiguity won't recur
- Built `count_unique_cells()` using `(ANIMAL_ID, SLICE, AT)` as the cell key — verified against the real pick: **214 images → 43 unique cells**, 3–10 images each
- Added `write_cell_summary_xlsx()` and wired it into `spike_analysis.py` right after the cell count is computed
- **Tested end-to-end**: temporarily pointed the ana list's `dir_results` footer at a local folder (since `Z:\Kang\Cluster\results` isn't mounted on this dev machine) to confirm the real pipeline run reaches and executes the xlsx-save step — confirmed `Found 214 entries in ana_list_20260618_000.txt -> 43 unique cells` + `Saved cell summary -> ana_list_20260618_000_cells.xlsx`, with the xlsx's 44 rows (header + 43 cells) verified by reading it back; reverted the footer edit afterward so the data file is unchanged from the user's perspective (`git diff` confirms no net change to `data/ana_list_20260618_000.txt`). The pipeline's next step still fails reaching `Z:\Kang\Cluster\proc_tiffs\...` — pre-existing, unrelated to this work, and only reproducible on a machine with the `Z:` drive mapped

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Determine the exact number of distinct cells in a picked ana_list, and which image files belong to each (`count_unique_cells`)
- ✅ Checked severity of the `AT = SITE_*` vs `CELL_*` issue before designing the grouping key — confirmed not present in today's pick, decided out of scope going forward per user
- ✅ Save the cell summary as xlsx to `results_dir` (`write_cell_summary_xlsx`, wired into `spike_analysis.py`)

## What should we do next? (TODOs)
- (none new from today — both candidate follow-ups (wiring `cell_df`/xlsx export into `ResultsExporter` proper, and normalizing the stray `CELL1` value in `REC_2025_04_03`) were explicitly marked out-of-scope-for-now by the user; Session 30's backlog above remains open and untouched this session)

## Last Session Recap
※ recap: Added `count_unique_cells()` (groups picked entries by `ANIMAL_ID`+`SLICE`+`AT` into distinct cells — verified 214 images → 43 cells for `ana_list_20260618_000.txt`) and `write_cell_summary_xlsx()`, wired both into `spike_analysis.py`'s `__main__`; confirmed via a temporary local `dir_results` override that the real pipeline reaches and executes the xlsx-save step. Also checked the known `AT=SITE_*` data-quality issue's severity (0% in today's pick, 63% historically pre-Oct-2024) — user decided no special-casing needed going forward.

---

# Log of the project progress 2026-06-21 Sun (Session 30)
Last working file: `classes/plot_results.py`
Last working line: 266 (`_add_scale_bar(..., font_size=6)` in `_plot_frame_panel`)

## List of modified files
- `classes/region_analyzer.py` — `get_temporal_traces(segment)`: missing-region trace default changed from `NaN`-filled to `0`-filled (per explicit user decision: "0"); added new public method `area_in_combined_region(frame, category) -> float` — counts pixels of a given category (BRIGHT/DIM) inside the union of `bright_largest`/`dim_largest` masks for *any* frame, returns µm² (used by row-1 panels to show a dynamic per-frame area instead of a static repeated number)
- `classes/plot_results.py` — `_plot_frame_panel` reworked per user feedback over several rounds:
  1. Row-1 panels now overlay the **spike frame's fixed** `bright_largest`/`dim_largest` contours on every panel (previously each panel ran independent per-frame detection via `find_largest_regions`)
  2. A pixel-coloring overlay (bright/dim pixels within the combined area, tinted magenta/cyan) was added then **reverted** — user found it visually bad
  3. x/y-span crosshair for bright is now shown **only on the spike frame panel** (`show_span=is_spike_frame`); dim never shows centroid/span, only its contour
  4. Bright/dim area numbers in non-spike-frame titles are now **dynamic per-frame counts** via the new `area_in_combined_region()`, not a static repeated spike-frame value; spike frame panel alone keeps the full `[x, y, area]` text
  5. `_overlay_region()` gained `show_centroid`/`show_span` flags, and now handles `dim_largest["contour"]` being a *list* of arrays (merged dim can be disjoint) vs. bright's single array
  6. Cosmetic: row 2 (trace panel) height increased (`height_ratios=[3, 1.2]` → `[3, 2.2]`, figure height 7 → 8.5), grid added to the trace panel, scale-bar text shrunk (`font_size=6`) so `"200 µm"` no longer overflows the row-1 panel border
- Deleted `_diag_spatiotemporal_summary.py`, `_diag_temporal_traces.py`, `_diag_trace_values.py` — temporary scripts used to visually validate every round of the above; their job is done

## Summary of current progress
- `plot_spatiotemporal_summary()` (the static export figure for spike-aligned region analysis) is now visually finalized after ~8 rounds of iteration this session: fixed spike-frame contours on all panels, no pixel-coloring overlay, span shown only on the spike frame, dynamic area counts elsewhere, taller/gridded trace panel, readable scale bar
- `get_temporal_traces()` finalized at the simple fixed-mask design (no category-filtering, no distant-region, no bbox-union — all of those were tried and reverted in earlier parts of this session/prior sessions) with `0` as the missing-region default
- Confirmed via direct discussion: exporting these figures needs the **function** `plot_spatiotemporal_summary()` (a plain function building a `Figure` directly, no `pyplot`, no `show()` — fully headless) but **not** any class; however importing `classes/plot_results.py` at all still pulls in PySide6 because GUI window classes live in the same file — flagged for future refactor (not done this session)
- User redirected next priority: implement the already-decided export filename/folder design, and additionally design a way to quickly find/browse specific results

## Completed TODOs (from Session 29, continued)
- ✅ Finished item 3 visual finalization: `get_temporal_traces()` + `plot_spatiotemporal_summary()` row-1/row-2 design locked in (see above) — the original Session 29 TODO "finish item 3: peak-to-peak latency + recovery-duration metric" is **partially superseded**: peak-latency annotation is implemented (`Peak Latency: {ms} ms` in the trace panel title), but the recovery-duration metric was never revisited this session — still open if still wanted
- ✅ Cleaned up `_diag_spatiotemporal_summary.py`, `_diag_temporal_traces.py`, `_diag_trace_values.py`

## What should we do next? (TODOs)
- [ ] **Priority**: Implement the compact export filename code (`{exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_{TYPE}.ext`) and flat `results/{exp_date}/{category}/` folder structure in `ResultsExporter` (design already decided in Session 29's "export/output redesign" note above)
- [ ] **Priority**: Design a way to quickly find/browse specific exported results (e.g. by animal/slice/site/date) — new idea, no design yet, likely depends on the filename/folder structure above plus the `ANIMAL_ID`/`CELL_KEY` SQLite columns
- [ ] Wire `btn_run_analysis` in `ctrl_align_spike.py` to call `spike_analysis.py`'s pipeline, with GUI line-edits showing live progress/steps (mirrors the existing `img_proc.py`/ALS emitter pattern) — still unconnected
- [ ] Decide fate of `RegionAnalyzer.find_largest_regions()` — no longer used by the actual export figure (row 1 switched to fixed spike-frame regions this session); currently dead code outside of (now-deleted) diagnostics
- [ ] Wire `plot_spatiotemporal_summary()` into the real `ResultsExporter` pipeline — it only existed in the (now-deleted) diagnostic script so far
- [ ] Refactor `classes/plot_results.py`: split plain `Figure`-returning export functions away from the `QMainWindow` GUI classes (`PlotPeaks`/`PlotSegs`/`PlotSpatialDist`/`PlotRegion`), since the GUI classes currently force a PySide6 import even for headless PNG export ([[project_plot_results_refactor]] memory)
- [ ] Add `ANIMAL_ID` column + derived `CELL_KEY` to `ResultsExporter`'s `experiments` SQLite table (carried over from Session 29, still not done)
- [ ] Build ROI + Otsu + `regionprops`-centroid soma detection on `EMI=RED` images, then contour-overlay export + centroid-to-soma distance calc (carried over from Session 29, item 5)
- [ ] Implement actual `ref_df` saving (CSV vs parquet, location/naming) — deferred since Session 29

## Last Session Recap
※ recap: Finalized `plot_spatiotemporal_summary()`'s row-1 panels (fixed spike-frame contours on all frames, dynamic per-frame area counts, no pixel coloring, span only on spike frame) and `get_temporal_traces()` (0-default for missing regions), then deleted the validation diagnostic scripts. Next priority: implement the decided export filename/folder structure and design a results find/browse feature.

---

# Log of the project progress 2026-06-20 Sat (Session 29)
Last working file: `classes/region_analyzer.py`
Last working line: 182 (`def get_temporal_traces`)

## List of modified files
- `utils/params.py` — added `ColumnSorter` dataclass (`UISizes`-style, accessed without instantiation): `CORE_COLUMNS`, `FLUIDIC_COLUMNS`, `IMG_COND_COLUMNS`, `EPHY_COND_COLUMNS`, `OTHER_COLUMNS`, `MEMO_COLUMNS`, `IGNORE_COLUMNS` — derived by actually inspecting column frequency across all 56 `REC_*` tables in `rec_data.db`, not guessed
- `functions/query_databases.py` — **new**: `lookup_rec_from_db(table, db_path, exp_db_path)` batch-queries `rec_data.db` grouped by date table (one query per `REC_{date}`, not per row), reads each table's real columns via `PRAGMA table_info` first (schemas vary by date), diagonally concatenates so missing columns become null; chains `_sort_rec_columns` (drops `IGNORE_COLUMNS`, orders the rest by `ColumnSorter` group priority) and `populate_animal_id_values` (fills missing `ANIMAL_ID` from `exp_info.db`'s `BASIC_INFO` DOR→Animal_ID mapping; handles 1-candidate "fill all" and 2-candidate "fill by elimination" cases) as its last two steps
- `functions/__init__.py` — registered `lookup_rec_from_db`/`populate_animal_id_values` in the lazy-import table
- `controllers/ctrl_data_selector.py` — local `CORE_COLUMNS` tuple replaced with `ColumnSorter.CORE_COLUMNS` import
- `spike_analysis.py` — deleted `_lookup_obj` entirely; `parse_ana_list` now only parses + existence-filters the ana list (returns `entries: pl.DataFrame` with `proc_tiff_path`/`raw_abf_path` columns added, no DB lookup); `__main__` computes `ref_df = lookup_rec_from_db(entries, db, exp_db)` once, then queries it **live per row** (`ref_df.filter(pl.col("Filename") == ...)`) for `OBJ` instead of flattening into a lookup dict — kept intentionally so `ref_df` stays a full, separately-queryable/saveable table rather than being collapsed away; later in the session, `RegionAnalyzer` construction/variable names updated to match its redesigned single-frame API (`spike_frame`, `bright_area`, `dim_area`)
- `classes/spatial_categorization.py` — `fit()`/`_calculate_global_thresholds()` reworked per item 1 (see design note below); `otsu_double`/`li_double` renamed to `base977_otsu`/`base977_li`
- `classes/region_analyzer.py` — major redesign, see item 4 design note below: single-spike-frame input (no more per-frame list/loop), `min_area` removed, dim-region selection now merges every dim component whose bbox overlaps a window around the bright region's own bbox, and a new `get_temporal_traces()` method extracts bright/dim mean-z-score traces across the segment using the spike frame's fixed masks (item 3, partially done — see TODOs)
- `.claude/settings.local.json` — new permission entries accumulated from today's tool calls
- Two temporary diagnostic scripts (`_diag_region_check.py`, `_diag_temporal_traces.py`) were created to visually validate the `RegionAnalyzer` redesign and the new temporal traces against the real test recording `2025_12_15-0013_BIEXP_GAUSS_CAT.tif` — both deleted at session end, their job (visual confirmation) done

## Summary of current progress
- Generalized the old per-row, single-column `_lookup_obj` into a batched, schema-tolerant, multi-column `rec_data.db` lookup, with column ordering now policy-driven (`ColumnSorter`) instead of whatever SQL happened to return
- Added cross-database enrichment: `ANIMAL_ID` gaps in the compiled table get backfilled from `exp_info.db` using a clear, verified 1-or-2-candidate rule (confirmed via real data that no DOR ever has more than 2 animals)
- Relocated all three DB-query functions out of `spike_analysis.py` into `functions/query_databases.py`, matching the project's existing public/private function-module convention
- Rewired `parse_ana_list`/`__main__` end-to-end so `obj` resolution goes through the new batched lookup instead of `_lookup_obj`; verified against `ana_list_20260601_000.txt` (correctly resolved `OBJ='10X'`) — the only failure hit afterward was a missing test `.abf` file on disk, unrelated to the refactor
- Hit a rough patch mid-session: left `parse_ana_list` and `__main__` in a mismatched state (signature/return-type changed in one but not the other) after a couple of rejected edits — caught and fully fixed within this session, verified by re-running the script
- Fixed the split-range thresholding design from Session 28's TODO list (item 1) — testing immediately surfaced that the originally planned approach was statistically degenerate; pivoted to a baseline-mean+2σ cutoff instead (see design note)
- Iteratively redesigned `RegionAnalyzer`'s dim-region selection (item 4) through several rounds of user sketches/diagrams — landed on "merge every dim component whose bbox overlaps a window around the bright region's own bbox," with the window margin visually tuned against real data (`DIM_SEARCH_MARGIN`: 1.0 → 0.5 → 0.2, since the bright region here is tall/thin and a too-large margin let the dim search window run off the edges of the frame, pulling in unrelated noise specks)
- Simplified `RegionAnalyzer` to operate on a single spike frame instead of looping over every frame in the segment, since only the spike frame's result was ever actually used downstream; removed the undiscoverable `min_area` parameter entirely
- Started item 3 (temporal trace analysis): added `get_temporal_traces()`, which reuses the spike frame's fixed bright/dim masks to compute a mean-z-score trace per frame across the whole segment; verified visually against real data — the resulting curves show exactly the expected shape (flat baseline noise, sharp spike-aligned rise, decay back toward baseline)
- Flagged but did not fix: `classes/results_exporter.py` still assumes `RegionAnalyzer`'s old per-frame-list output shape — deferred until the export-wiring TODO is tackled

## Completed TODOs (from Session 28)
- ✅ Complete the rec_data.db query functions for the picked list — done via `lookup_rec_from_db`/`populate_animal_id_values` in `functions/query_databases.py`

## Completed TODOs (from Session 29)
- ✅ Exported `ColumnSorter` via `utils/__init__.py`; switched `functions/query_databases.py` and `controllers/ctrl_data_selector.py` to import it (and `MODELS_DIR`/`REC_DB_PATH`) from `utils` instead of `utils.params` directly
- ✅ Item 1 (split-range thresholding) — see revised design note above; implemented as baseline-mean+2σ cutoff instead of the originally planned two-range split, with `base977_otsu`/`base977_li` renaming

## Design idea logged this session (not yet implemented): export/output redesign
Motivating problem: 214 picked ana_list entries don't mean 214 distinct cells — the same animal/slice/site can appear in multiple recordings (different days, detrend modes, etc.), and the team wants to know the true distinct-cell count plus a way to group/sort exports for downstream stats without making the export folder structure too deep to browse.

Decided direction:
- **Stop using folder depth to encode identity.** A `DOR/Animal_ID/Slice/AT/...` nested folder tree is both too deep to browse and still useless for grouping (you'd have to walk the filesystem to count cells). Split the concern in two:
  1. **Filenames** stay flat under `results/{exp_date}/{category}/` (`categorized/`, `median/`, `regions/`, `spatials/` — categories already exist in `ResultsExporter`), but gain a compact, scannable code: `{exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_{TYPE}.ext`, e.g. `2025_12_15-0014_A1S1RC1_BIEXP_GAUSS_CAT.tif`.
     - `A{n}` = sequential index per **distinct animal within the current export batch** (not the literal `ANIMAL_ID`, which is a long strain code like `neoChAT-204` in real data — confirmed via `exp_info.db.BASIC_INFO`) — purely for fast visual "is this a different cell?" scanning.
     - `S{slice}` = the `SLICE` column verbatim (e.g. `1R`).
     - `C{site}` = derived from `AT` (e.g. `SITE_1` → `C1`, take trailing number after the prefix).
  2. **Full metadata** (real `Animal_ID`, `OBJ`, `SLICE`, `AT`, etc.) goes into figure titles/captions once plotting exists (`classes/plot_results.py`), not into filenames.
  3. **Distinct-cell counting and grouped statistics happen against the SQLite `experiments` table** (`classes/results_exporter.py`), not the filesystem — needs an `ANIMAL_ID` column added (currently missing; only `SLICE`/`AT` exist) plus a derived `CELL_KEY = f"{ANIMAL_ID}_{SLICE}_{AT}"`. Then "214 recordings → N distinct cells" is `df.group_by(["ANIMAL_ID","SLICE","AT"]).len()` against that table.

## Design ideas logged this session (not yet implemented): analysis fixes (higher priority than export)
User explicitly reordered: these 5 analysis-correctness items come **before** the export/output redesign below.

1. ✅ **Done — implementation diverged from the original split-range design.** Originally planned: split `_calculate_global_thresholds()` into two frame ranges (background pass on `[start, spike_frame]`, signal pass on `[spike_frame, end]`). Testing showed that was degenerate — `functions/zscore_img_segs.py` z-scores each pixel against its *own* baseline mean/std, so baseline-only pixels are just ~N(0,1) noise with no real background/signal bimodality; Otsu/Li on that pool just bisects the noise near its center, marking ~half of baseline as "dim". **Actual fix**: `thresh_dim = baseline_pixels.mean() + 2*baseline_pixels.std()` (baseline = frames `[0, spike_frame_idx)`; ~97.7% of pure-noise baseline pixels stay below this cutoff), then `thresh_bright = threshold_otsu/li(all_pixels[all_pixels > thresh_dim])` pooled across the **whole** segment. Methods renamed `otsu_double`/`li_double` → `base977_otsu`/`base977_li` to reflect the new algorithm. `fit()` now takes `spike_frame_idx` (passed from `spike_analysis.py` as `median_segment.shape[0] // 2`). Verified visually against `2025_12_15-0013_BIEXP_GAUSS_CAT.tif` — clean bright blob + coherent dim halo + mostly-black background.
2. **Replace old interactive plots with export-oriented ones** — `classes/plot_results.py`'s `PlotPeaks`/`PlotSegs`/`PlotSpatialDist`/`PlotRegion` are live `QMainWindow` viewers (comboboxes, play buttons), not batch-export-friendly. Build new static-figure functions for `ResultsExporter.export_figure()` instead of adapting the old ones — old ones to be ignored/dropped, not reused.
3. 🟡 **Partially done.** Original plan: use the spike frame's fixed dim/bright masks to compute mean z-score per mask across every frame in the segment, then find peak latency + a recovery-duration metric on the combined trace. **Done so far**: `RegionAnalyzer.get_temporal_traces(segment)` computes `bright_trace`/`dim_trace` from the fixed spike-frame masks — verified visually, correct shape (flat baseline, spike-aligned peak, decay). **Not done yet**: peak-to-peak latency between the two traces, and the recovery-duration metric on `combined = bright_trace + dim_trace` — carried over as a TODO.
4. ✅ **Done — implementation diverged significantly from the original plan after live debugging.** Original plan: find largest bright first, then the dim region must be the largest dim CC whose centroid falls inside the bright region's mask. Testing (via a temporary diagnostic PNG comparing real categorized data) showed this was both too strict (a halo's centroid often falls outside the bright blob's own mask, not inside it) and not what was actually wanted. After several rounds of sketches, the final design: build a search window by expanding the bright region's own bbox by `DIM_SEARCH_MARGIN × its own span` on each side (no manual offset to guess — reuses a quantity already being computed), then **merge every dim connected-component whose bbox overlaps that window at all** (fully or partially) into one combined dim region, rather than picking a single largest dim CC. `DIM_SEARCH_MARGIN` visually tuned against real data: `1.0` and `0.5` both let the window run past the frame edges for an elongated bright region, pulling in unrelated noise; settled on `0.2`. Also simplified `RegionAnalyzer` to a single-spike-frame API (no more per-frame list/loop) and removed the `min_area` parameter (undiscoverable magic number).
5. **Red-channel soma-distance analysis** — replace `_CAT.tif` export with a contour overlay (bright+dim outlines) drawn on the corresponding `EMI=RED` recording (labels the patched cell body), then measure centroid-to-soma distance. Soma position method (user-specified): define an ROI, apply Otsu threshold within it, then `skimage.measure.regionprops` centroid to find the cell center — not fully automatic on the whole frame, and not manual-click.

## What should we do next? (TODOs)
- [ ] Build new export-oriented static-figure plotting functions; drop the old interactive `Plot*` classes from the new pipeline (item 2 above)
- [ ] Finish item 3: peak-to-peak latency between `bright_trace`/`dim_trace`, and a recovery-duration metric on `combined = bright_trace + dim_trace` (trace extraction itself is already done via `RegionAnalyzer.get_temporal_traces()`)
- [ ] Build ROI + Otsu + `regionprops`-centroid soma detection on `EMI=RED` images, then contour-overlay export + centroid-to-soma distance calc (item 5 above)
- [ ] Implement actual `ref_df` saving — format (CSV vs parquet) and location/naming convention were both explicitly deferred today ("don't think about that now"); only the live-query usage was built, not persistence
- [ ] Add `ANIMAL_ID` column + derived `CELL_KEY` to `ResultsExporter`'s `experiments` SQLite table (see export/output design idea above)
- [ ] Implement the compact filename code (`A{n}S{slice}C{site}`) in `ResultsExporter`'s export filenames
- [ ] Wire `ResultsExporter.export_all(...)` into `spike_analysis.py`'s per-row loop — `AbfClip.get_export_data()` / `RegionAnalyzer.get_summary()`/`get_results()` already match its expected input shape, just never called from the new pipeline
- [ ] Decide: should `ResultsExporter` write to the ana_list's `dir_results`, or keep its own separate `results/` root? (open design question, blocks the wiring step above)

## Completed TODOs (from Session 29, continued)
- ✅ Archived `im_dynamics.py`, `batch_process.py`, `test_batch.py` into a new `archive/` folder (no existing references to them elsewhere in the codebase)

## Last Session Recap
※ recap: Fixed thresholding (baseline mean+2σ cutoff, renamed to `base977_otsu`/`base977_li`), then redesigned `RegionAnalyzer` end-to-end — single spike-frame API, `min_area` removed, dim region now a merge of every dim component overlapping a tuned search window around the bright bbox (`DIM_SEARCH_MARGIN=0.2`) — and started temporal trace extraction (`get_temporal_traces()`, verified working). Pending: peak latency + recovery-duration metrics to finish item 3.

---

# Log of the project progress 2026-06-19 Fri (Session 28)
Last working file: `spike_analysis.py`
Last working line: 67 (`def parse_ana_list`)

## List of modified files
- `functions/list_parser.py` — **new**: shared `list_parser(path) -> (table, io_dirs)` + private `_df_builder`. Reads the `Picked: [...]` header into column names, reads every bracket row in full (raises `ValueError` on field-count mismatch instead of silently truncating), and collects `dir_*:` lines into `io_dirs`
- `functions/__init__.py` — registered `list_parser` in the lazy-import table
- `spike_analysis.py` — deleted `_parse_bracket`; `parse_ana_list` now reads `ana_list_*.txt` via `list_parser` and filters/builds entries by column name (`gauss_exist`/`als_exist`/`abf_exist`/`paired_abf`) instead of `parts[:5]`
- `img_proc.py` — deleted `_parse_bracket`; `parse_proc_list` reads `proc_*.txt` via `list_parser` by column name. `update_proc_list_gauss_exists` rewritten to recompute `gauss_exists`/`do_processing`/`detrend_mode` by name and re-serialize each row using the table's actual column order, so `als_exists`/`paired_abf` can't be clobbered by a position shift. Fixed stale "5 fields" docstring
- `als_correct.py` — deleted its own `_parse_bracket`; `parse_proc_list` reads `gauss_exists` by name instead of `parts[1]`
- `controllers/ctrl_align_spike.py`, `controllers/ctrl_img_proc.py`, `functions/file_status.py`, `CLAUDE.md`, `data/ana_list_20260618_000.txt` — carried in the same commit from earlier in this session: `gauss_mode`/`als_mode` renamed to `gauss_exists`/`als_exists` and consolidated onto one proc-file index in `file_status.py`; CLAUDE.md gained explicit "never use chained shell commands" + "default to PowerShell" rules after repeated permission-prompt friction

## Summary of current progress
- Identified that `spike_analysis.py`, `img_proc.py`, and `als_correct.py` each hand-rolled their own positional bracket-row parser (`_parse_bracket`) for the pick/proc/ana list `.txt` format, with at least one (`img_proc.py`) already drifted out of sync with its own docstring — a silent-corruption risk since these functions also write back to disk
- Designed and built one shared `list_parser` that treats the `Picked: [...]` header as the actual schema source instead of decorative text, builds a `pl.DataFrame` from it, and exposes `dir_*` paths via `io_dirs` — named-column access throughout, no more magic indices
- Migrated all three scripts onto it, including the trickier write-back path in `update_proc_list_gauss_exists`
- Verified `list_parser` against real `ana_list`/`proc_list`/`pick_list` data files (correct columns + row counts + `io_dirs`), and verified the write-back round-trip preserves untouched columns (`als_exists`, `paired_abf`) correctly
- User began manual GUI validation of the refactored pipeline; interrupted by this wrap-up

## Completed TODOs (from Session 27)
- (none of Session 27's 3 TODOs were addressed this session — this session's scope was entirely the list-parsing refactor. RegionAnalyzer/export schema review, color-coded status columns, and export completion feedback are unchanged and not re-confirmed here)

## What should we do next? (TODOs)
- [ ] Finish manual GUI validation of the refactored `spike_analysis.py` / `img_proc.py` / `als_correct.py` pipeline (started this session, interrupted by wrap-up)
- [ ] Complete the rec_data.db query functions for the picked list
- [ ] (Deferred, not yet scheduled) Bring `controllers/ctrl_img_proc.py` (`load_pick_list`, `export_proc_list`) and `controllers/ctrl_align_spike.py` (`_load_entries`, `_proc_dir`, `export_ana_list`) onto `list_parser` — same positional-parsing pattern still exists there, left out this session due to GUI risk (hardcoded Qt column indices)

## Last Session Recap
※ recap: Built a shared `list_parser` (functions/list_parser.py) that reads pick/proc/ana list files by column name instead of positional indices, and migrated spike_analysis.py/img_proc.py/als_correct.py (incl. the gauss_exists write-back) onto it; GUI validation was in progress when interrupted, and rec_data.db query functions for the picked list are still to be completed.

---

# Log of the project progress 2026-06-19 Fri (Session 27)
Last working file: `controllers/ctrl_align_spike.py`
Last working line: 111-112 (`all_abf_ready` / `btn_confirm_analyzing_list.setEnabled`)

## List of modified files
- `controllers/ctrl_align_spike.py` — `__init__` now disables `btn_confirm_analyzing_list` by default; `_load_entries()` computes `all_abf_ready = bool(rows) and (df["ABF_READY?"] == "YES").all()` and enables the button only when every loaded row's `ABF_READY?` is `"YES"` (mirrors how `btn_export_proc_list`/`btn_start_processing` gate on `IMG_READY` in `ctrl_img_proc.py`)
- `docs/knowledgebase/scan_once_lookup_many_pattern.md` — **new**: documents the "scan once, look up many times" pattern — why `check_file_status` in both `ctrl_img_proc.py` and `ctrl_align_spike.py` got faster (avoiding `O(rows × files)` re-globbing and N individual `.exists()` round-trips, costly on network-mounted drives); includes the actual before/after code from both controllers and a concrete 3-row/5-file worked example

## Summary of current progress
- Explained (with real before/after code and worked numeric examples) why `check_file_status` is faster: both controllers moved from "ask the filesystem once per row" (re-glob per row in `ctrl_img_proc.py`, individual `.exists()` per row in `ctrl_align_spike.py`) to "scan the directory once, build a dict/set index, then do O(1) lookups per row"
- Wired `btn_confirm_analyzing_list` enable/disable state to the `ABF_READY?` column so it can't be clicked until every entry has its paired ABF file present
- Captured the scan-once-lookup-many pattern as a new knowledgebase entry for future reuse

## Completed TODOs (from Session 26)
- (none of Session 26's 4 TODOs were addressed this session — only "RegionAnalyzer/export schema review" was confirmed as still relevant going forward; the other three were dropped from active tracking)

## What should we do next? (TODOs)
- [ ] Validate the new ideas of analysis output — review whether the bright/dim largest-region stats from `RegionAnalyzer` are meaningful/correct before deciding the export schema and wiring `ResultsExporter` ([[project_spike_analysis_todos]] memory still applies)
- [ ] Add color-coding (e.g. green/red) to YES/No values in the EXIST/READY columns (`GAUSS_EXISTS?`, `ALS_EXISTS?`, `ABF_READY?`, `IMG_READY`, etc.) across check tables for easier visual recognition
- [ ] Export buttons (`export_proc_list`, `export_ana_list`, etc.) give no GUI feedback when done besides a console log — consider adding a completion info dialog or status indicator

## Last Session Recap
※ recap: Explained why `check_file_status` got faster in both controllers (scan-once/index pattern, documented to knowledgebase with worked example) and wired `btn_confirm_analyzing_list` to enable only when all `ABF_READY?` are "YES"; pending: RegionAnalyzer/export schema validation, color-coded status columns, and export completion feedback.

---

# Log of the project progress 2026-06-18 Thu (Session 26)
Last working file: `spike_analysis.py`
Last working line: ~220 (`RegionAnalyzer` / `get_frame_results` output) — unchanged, no code edited this session

## List of modified files
- (none — ops/troubleshooting session only, no repo files touched)

## Summary of current progress
- Diagnosed why `proc_tiffs/` writes on the saion cluster were hitting a hard cap despite `df -h ~` showing 28T free: `/home` (`lan01.emcisilon.oist.jp:/hpc_home`) is a shared 50T Isilon filesystem, but enforces a separate **per-user 50GB quota** invisible to `df` — only surfaces as "disk quota exceeded" on write
- Moved analysis output from `/home/.../proc_tiffs/` to `/work/WickensU/kang/proc_tiffs/`, a separate large-quota work filesystem — confirmed via MobaXterm SFTP that the 175-file BIEXP detrend + Gaussian blur batch now runs cleanly without hitting quota
- No code or config changes made — this was purely cluster storage triage

## Completed TODOs (from Session 25)
- (none — this was an ops aside, Session 25's open TODOs are unaffected)

## What should we do next? (TODOs)
- [ ] Validate the new ideas of analysis output — review whether the bright/dim largest-region stats from `RegionAnalyzer` are meaningful/correct before deciding the export schema and wiring `ResultsExporter` ([[project_spike_analysis_todos]] memory still applies: largest-region detection criteria + export schema both need design review)
- [ ] Wire `btn_run_analysis` in `ctrl_align_spike.py` to call the `spike_analysis.py` pipeline (still unconnected)
- [ ] Remove temp `_MED.tif`/`_CAT.tif` debug saves in `spike_analysis.py` once pipeline is verified
- [ ] Archive `im_dynamics.py`, `batch_process.py`, `test_batch.py` once pipeline is complete (still present in repo root)

## Last Session Recap
※ recap: Diagnosed a saion cluster `/home` 50GB per-user quota (invisible to `df -h`, separate from the 28T shown free on the shared 50T filesystem) and moved pipeline output to `/work/WickensU/kang/proc_tiffs/`, where the 175-file BIEXP+Gaussian batch now runs without hitting quota. No code changed; Session 25's open TODOs (RegionAnalyzer/export schema validation, `btn_run_analysis` wiring, debug-save cleanup, script archiving) remain.

---

# Log of the project progress 2026-06-18 Thu (Session 25)
Last working file: `spike_analysis.py`
Last working line: ~220 (`RegionAnalyzer` / `get_frame_results` output)

## List of modified files
- `views/view_align_spike.py` — removed dead `lbl_run_on`/`le_run_on` row (`Run On:` field); `spike_analysis.py` has no CUDA branch to report, and `ctrl_align_spike.py` never set it
- `spike_analysis.py` — renamed debug TIFF outputs `_median.tif` → `_MED.tif`, `_categorized.tif` → `_CAT.tif`
- `data/pick_list.json` — reset to `[]`
- `docs/knowledgebase/hpc_slurm_gpu_workflow.md` — **new**: day-to-day Slurm GPU job submission routine for `img_proc.py` on saion (node probing loop → `srun --pty bash` reservation → activate venv + run)
- `.claude/settings.local.json` — added `Bash(git status *)` permission
- `CLAUDE.md` — softened separation-line guidance (don't wrap every small subsection in `===` banners)
- Committed as `444c86f "commit for running on cluster"`; `CLAUDE.md`/`settings.local.json` tweaks above are still uncommitted

## Summary of current progress
- Confirmed `spike_analysis.py`'s pipeline (`AbfClip` → `zscore_img_segs` → `spike_centered_median` → `SpatialCategorizer` → `RegionAnalyzer`) has no CUDA/GPU branch — cleaned the GUI accordingly
- Discussed and documented the cluster GPU job workflow (node probing via `srun nvidia-smi` loop, `--time` day-hours format, `sinfo`/`nodelist` lookups) into a new knowledgebase doc
- `SpatialCategorizer` and `RegionAnalyzer` are wired into `spike_analysis.py` and producing bright/dim region stats (area, x/y span) per spike frame — `ResultsExporter` still not wired up

## Completed TODOs (from Session 24)
- ✅ Removed unused `Run On:` field from `view_align_spike.py`

## What should we do next? (TODOs)
- [ ] Validate the new ideas of analysis output — review whether the bright/dim largest-region stats from `RegionAnalyzer` are meaningful/correct before deciding the export schema and wiring `ResultsExporter` ([[project_spike_analysis_todos]] memory still applies: largest-region detection criteria + export schema both need design review)
- [ ] Wire `btn_run_analysis` in `ctrl_align_spike.py` to call the `spike_analysis.py` pipeline (still unconnected)
- [ ] Remove temp `_MED.tif`/`_CAT.tif` debug saves in `spike_analysis.py` once pipeline is verified
- [ ] Archive `im_dynamics.py`, `batch_process.py`, `test_batch.py` once pipeline is complete (still present in repo root)

## Last Session Recap
※ recap: Removed the dead "Run On:" GUI field (spike_analysis.py has no CUDA branch), documented the saion cluster Slurm GPU job workflow into a new knowledgebase file, and confirmed RegionAnalyzer is wired into the pipeline — but its output still needs validation before deciding the ResultsExporter schema.

---
# Log of the project progress 2026-06-16 Mon (Session 24)
Last working file: `spike_analysis.py`
Last working line: ~160

## List of modified files
- `classes/abf_clip.py` — removed `spike_min_height` from `spike_detection()`; removed dead segment accessors (`get_img_segment`, `get_abf_segment`, `get_time_segment`); `get_export_data()` now includes `tiff_full_path`/`abf_full_path` (Option B, keeps `abf_serial`/`img_serial`); removed `min_interval_frames`; renamed `max_interval_frames` → `set_interval_frames`; `set_interval_frames` now auto-derived as `min(mode(min_available_frames), 20)` via two-pass approach in `get_available_spiking_frames()`
- `functions/imaging_segments_zscore_normalization.py` → **renamed** to `functions/zscore_img_segs.py`; function renamed `img_seg_zscore_norm` → `zscore_img_segs`; new signature takes `proc_tiff_path: Path` + `lst_img_frame_ranges`; opens TIFF once for all segments
- `functions/__init__.py` — lazy import updated to `zscore_img_segs` from `.zscore_img_segs`
- `functions/spike_centered_processes.py` — renamed `_median_axis0` → `_cpu_median_axis0`; added `cache=True`
- `spike_analysis.py` — added `zscore_img_segs` + `spike_centered_median` to pipeline; `del lst_zscore` after median; temp median TIFF save to `results_dir`; `_parse_bracket` now returns `(entry, skip_reason)` tuple for diagnostic skip output; added `numpy`/`tifffile` imports

## Summary of current progress
- **`AbfClip` refined**: spike detection uses prominence only (no absolute height); segment accessors removed; `set_interval_frames` fully auto-derived from data mode (capped at 20) — uniform segment lengths guaranteed
- **`zscore_img_segs` redesigned**: renamed from `imaging_segments_zscore_normalization`, new signature takes path + frame ranges, opens TIFF in single pass
- **`spike_centered_processes.py`**: `_cpu_median_axis0` renamed + cached; GPU version discussed and decided against (MAX_S constraint + fallback risk not worth it)
- **`spike_analysis.py` pipeline extended**: `AbfClip` → `zscore_img_segs` → `spike_centered_median` → temp TIFF save; skip diagnostics now show reason (e.g. `abf_exist=No`)

## Completed TODOs (from Session 23)
- ✅ Fixed `AbfClip` spike detection (removed `spike_min_height`, prominence-only)
- ✅ Redesigned `zscore_img_segs` with path-based loading
- ✅ Wired `zscore_img_segs` + `spike_centered_median` into `spike_analysis.py`
- ✅ Unified segment frame lengths via auto-derived `set_interval_frames`

## What should we do next? (TODOs)
- [ ] Redesign and optimize `SpatialCategorizer` and `RegionAnalyzer` before wiring into pipeline
- [ ] Remove temp median TIFF save in `spike_analysis.py` once pipeline is verified
- [ ] Wire `SpatialCategorizer` → `RegionAnalyzer` → `ResultsExporter` into `spike_analysis.py`
- [ ] Wire `btn_run_analysis` in `ctrl_align_spike.py` to call the pipeline
- [ ] Archive `im_dynamics.py`, `batch_process.py`, `test_batch.py` once pipeline is complete

## Last Session Recap
※ recap: Refined AbfClip (prominence-only detection, auto set_interval_frames from mode), redesigned zscore_img_segs with path-based TIFF loading, extended spike_analysis.py pipeline to zscore→median; next is redesigning SpatialCategorizer and RegionAnalyzer.

---

# Log of the project progress 2026-06-15 Sun (Session 23)
Last working file: `classes/abf_clip.py`
Last working line: ~160 (`get_export_data`)

## List of modified files
- `classes/abf_clip.py` — complete rewrite: new `__init__` takes `proc_tiff_path`, `raw_abf_path`, `results_dir`, `detrend_mode`, `normalization`, optional `fs_imgs`/`min_interval_frames`/`max_interval_frames`; removed `load_img()` — frame count via `tifffile` metadata; `self.cs` → module-level `console`; `lst_img_segments`/`lst_abf_segments`/`lst_time_segments` → index tuples (`lst_img_frame_ranges`, `lst_abf_sample_ranges`); `_export_spike_xlsx()` writes `df_Vm`+`df_peaks` to xlsx via openpyxl; 3 new getters `get_img_segment`, `get_abf_segment`, `get_time_segment`; `get_export_data()` derives `exp_date`/`abf_serial`/`img_serial` from path stems
- `spike_analysis.py` — `_parse_bracket` now has 3 silent guards using ana_list existence flags (`gauss_exist`, `als_exist`, `abf_exist`), takes `raw_abfs_dir` param, returns `proc_tiff_path`/`raw_abf_path`; `parse_ana_list` reads all 3 footer keys (`dir_proc_tiffs`, `dir_raw_abfs`, `dir_results`), returns `(entries, results_dir, detrend_mode, normalization)`; `AbfClip` imported and used in `__main__` loop

## Summary of current progress
- **`AbfClip` fully redesigned**: takes direct paths from `parse_ana_list` instead of decomposed `exp_date`/`abf_serial`/`img_serial`; memory-efficient (no full TIFF load — frame count from tifffile header, pixels read page-by-page via getters); spike detection xlsx exported automatically to `results_dir`
- **`spike_analysis.py` parse layer complete**: `_parse_bracket` uses ana_list existence flags as authoritative guards (no disk check); `parse_ana_list` reads all footer keys and returns detrend/normalization as explicit values; `__main__` loop creates an `AbfClip` per entry and reports segment count
- **Tested**: spike detection xlsx verified in Excel — Vm trace + peak markers look correct

## Completed TODOs (from Session 22)
- ✅ Complete ABFClip flow in `spike_analysis.py`
- ✅ Redesign `AbfClip.__init__` to use direct paths

## What should we do next? (TODOs)
- [ ] Complete the full pipeline in `spike_analysis.py` (after AbfClip: `img_seg_zscore_norm` → `spike_centered_median` → `SpatialCategorizer` → `RegionAnalyzer` → `ResultsExporter` → plots, looping over all entries)
- [ ] Wire up `btn_run_analysis` in `ctrl_align_spike.py` to call the pipeline
- [ ] Archive `im_dynamics.py`, `batch_process.py`, `test_batch.py` once pipeline is complete

## Last Session Recap
※ recap: Redesigned `AbfClip` to use direct paths and index-based segment storage (no full TIFF load), and completed `spike_analysis.py` parse layer with ana_list existence guards; next is the full analysis pipeline loop.

---

# Log of the project progress 2026-06-15 Sun (Session 22)
Last working file: `controllers/ctrl_align_spike.py`
Last working line: ~135 (`export_ana_list`)

## List of modified files
- `controllers/ctrl_align_spike.py` — created from scratch: `QFileSystemWatcher` for ABF dir + proc dir, `_load_entries` (table with reordered columns), `check_file_status`, `export_ana_list` (following `export_proc_list` pattern), fixed `abf_name = parts[-1]`
- `views/view_align_spike.py` — added `btn_confirm_analyzing_list` next to `btn_load_proc_list` in a `QHBoxLayout`
- `spike_analysis.py` — renamed `parse_proc_list` → `parse_ana_list`; updated `_parse_bracket`: `parts[3]` for abf, min check `>= 4`; CLI arg `--proc_list` → `--ana_list`; docstring updated to ana_list format
- `.claude/settings.local.json` — fixed ruff wildcard permission entry (`\\\\` not `\\`); the correct PowerShell command is `ruff check <file>` (no full path needed)
- `data/ana_list_20260605_000.txt` — new file generated by `export_ana_list` (test output)

## Summary of current progress
- **Completed `ctrl_align_spike.py`**: full watcher infrastructure (watches ABF dir + proc dir), table population with correct column order (`DOR, TIFF_SERIAL, GAUSS_EXIST?, ALS_EXIST?, ABF_SERIAL, ABF_READY?`), `check_file_status` delegates to `_load_entries`, `export_ana_list` mirrors `export_proc_list` exactly — copies original proc_list, modifies brackets to new 5-field format `[raw_tiff_name, gauss_exist, als_exist, paired_abf, abf_exist]`, appends `dir_raw_abfs` and `dir_results` footer lines, saves to `data/`
- **`view_align_spike.py` layout complete**: "Directory of Raw ABFs" field, Load/Confirm buttons side by side, proc list table, detrend/norm radio groups, export path, Run Analysis button, processing info fields
- **`spike_analysis.py` ana_list naming**: function and CLI renamed to reflect that input is now `ana_list_*.txt` not `proc_*.txt`; `_parse_bracket` reads `parts[3]` for paired ABF in the new format
- **Ruff permission fixed**: correct PowerShell pattern in `settings.local.json`; always use `ruff check <file>` directly (no `& "full\path"`)

## Completed TODOs (from Session 21)
- ✅ Complete the layout of tab spike_alignment
- ✅ Add `QFileSystemWatcher` to `ctrl_align_spike.py`

## What should we do next? (TODOs)
- [ ] Complete the spike_analysis.py pipeline (spike detection, alignment, spatial categorization, region analysis)
- [ ] Wire up `btn_run_analysis` in `ctrl_align_spike.py` to call the pipeline
- [ ] Complete ABFClip flow in `spike_analysis.py`

## Last Session Recap
※ recap: Completed `ctrl_align_spike.py` (watcher, table, export_ana_list) and finalized `view_align_spike.py` layout; next is implementing the spike detection/alignment pipeline in `spike_analysis.py`.

---

# Log of the project progress 2026-06-15 Sun (Session 21)
Last working file: `functions/gaussian_blur.py`

## List of modified files
- `.claude/skills/project-tracker/SKILL.md` — added recap trigger scenario and instructions
- `controllers/ctrl_als_correct.py` — restored `als_run` top-level import to fix UI freeze on proc list load
- `functions/detrend.py` — added `cache=True` to `_cpu_mov`, `_cpu_biexp`
- `functions/gaussian_blur.py` — added `cache=True` to `_cpu_kernel`, `_cpu_conv`, `_cpu_gaussian_blur`

## Summary of current progress
- **Fixed UI freeze on proc list load**: traced to numba initializing on the main thread — `als_run` was moved to a lazy import in a previous session, so numba was no longer pre-loaded at startup; restored top-level import to fix it
- **Added `cache=True` to numba JIT kernels** in `detrend.py` and `gaussian_blur.py` — first run compiles and caches to disk; subsequent startups load cached binary, significantly faster (`als.py` already had this)
- **Updated project-tracker skill**: added recap trigger — when user says "recap", outputs a concise `※ recap:` line from conversation context; wrap-up now also saves the recap line to `docs/continue_from_here.md` for use after `/clear`

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fixed UI freeze on ALS Correction proc list load
- ✅ Numba JIT cache enabled across all CPU kernels

## What should we do next? (TODOs)
- [ ] Complete the flow of ABFClip
- [ ] Complete the layout of tab spike_alignment

## Last Session Recap
※ recap: Fixed a numba-caused UI freeze on ALS proc list load and added `cache=True` to CPU JIT kernels for faster subsequent startups; updated project-tracker skill with recap support.

---

# Log of the project progress 2026-06-10 Tue (Session 20)

## Last working file
- `controllers/ctrl_img_proc.py`

## List of modified files
- `controllers/ctrl_img_proc.py` — multiple fixes (see summary)
- `controllers/ctrl_data_selector.py` — merged schema comment onto `Picked:` line
- `classes/helper_cell_dropdown.py` — generalized to accept callable or list for `menu_options`
- `data/pick_20260601_000.txt` — fixed `Picked:` format
- `data/pick_20260605_000.txt` — fixed `Picked:` format

## Summary of current progress
- **Proc file comment bug fixed**: `ctrl_data_selector.py` was writing `Picked:` and `# [raw_tiff_name, paired_abf]` as two separate lines; merged schema onto `Picked:` line so `ctrl_img_proc.py` cleanly overwrites it on export
- **Added `ALS_EXISTS?` column** to img_proc check table — purely informational, no effect on PROC/MODE; also added to proc file export schema (now 6 columns: `raw_tiff_name, gauss_exists, als_exists, do_processing, detrend_mode, paired_abf`)
- **Fixed double "File status updated" prints** — `directoryChanged` watcher fires mid-write during processing; fixed by disconnecting in `start_processing()` and reconnecting in `_on_processing_done()`
- **Fixed PROC/MODE default logic** — any GAUSS exists → PROC=SKIP/MODE=NONE; no GAUSS → PROC=YES/MODE=BIEXP
- **Fixed MODE dropdown** — NONE no longer selectable when PROC=YES; `CellDropdownDelegate` generalized to accept a callable for context-aware options per row
- **`btn_start_processing` disabled** when any `IMG_READY` is MISSING (same gate as `btn_export_proc_list`)
- Column indices updated: PROC=5, MODE=6 (after inserting `ALS_EXISTS?` at col 4)

## Completed TODOs/Tasks (before new wrap-up)
- ✅ All changes above fully completed within this session

## What should we do next? (TODOs)
- [ ] Complete the flow of ABFClip
- [ ] Complete the layout of tab spike_alignment

---

# Log of the project progress 2026-06-10 Tue (Session 19)

## Last working file
- `functions/check_cuda.py`, `controllers/ctrl_als_correct.py`

## List of modified files
- `functions/check_cuda.py` — moved `_setup_cuda_environment()` call and `from numba import cuda` from module level into `check_cuda()` body; module is now cheap to import (~0ms instead of ~400ms)
- `controllers/ctrl_als_correct.py` — removed `import tifffile` and `from functions import als_run` from module level; both deferred into `_als_test()` method (first use); `from functions import check_cuda` kept at module level (now cheap after above fix)
- `pyproject.toml` — removed `"C90"` (McCabe complexity) from `lint.extend-select`
- `CLAUDE.md` — updated ruff invocation: now just `ruff check <files>` (installed via `uv tool install ruff`, shim at `C:\Users\Kang\scoop\persist\uv\tools\shims\ruff.exe`); removed old full-path instruction
- `.claude/settings.local.json` — removed 11 stale ruff permission entries (old full paths, venv paths, uvx); kept only `"PowerShell(ruff check *)"` and `"Bash(ruff check:*)"` as clean wildcards

## Summary of current progress
- Investigated why startup was still slow despite previous lazy import attempts
- Root cause: `from functions import X` in controllers calls `__getattr__` **immediately** — it bypasses the lazy loading in `functions/__init__.py`, which only defers when accessed via `functions.X` later. So `from numba import cuda` in `check_cuda.py` (module level) was always loading at startup
- Fixed by moving the heavy numba import inside the function body; `als_run` (from `als.py`, also loads numba) deferred by moving its import inside `_als_test()`
- Startup improvement: ~400ms from `numba.cuda` + ~400ms from `als.py` numba imports eliminated from startup path
- Also cleaned up ruff setup for new laptop (ruff now installed via uv, accessible as `ruff` in PATH)

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fixed slow app startup — numba imports deferred to first use

## What should we do next? (TODOs)
- [ ] Consider connecting `pick_confirmed` signal to also switch to ALS Correction tab after image processing completes
- [ ] Fix git safe.directory for this repo on the new laptop (tool subprocess reads different user context — run `git config --global --add safe.directory "D:/MyDB/2_Programs/PG_005"` in your own terminal, not via Claude tool)

---

# Log of the project progress 2026-06-05 Thu (Session 18)

## Last working file
- `controllers/ctrl_als_cal.py` (L38, L152), `views/view_als_cal.py` (L84)

## List of modified files
- `functions/detrend.py` — removed `edge_min` from all 4 kernels; changed output to ΔF/F₀ `(raw - trend) / trend`; removed dead last-frame blocks from `_cpu_mov` / `_gpu_mov`
- `img_proc.py` — GAUSS TIFFs (`_MOV_GAUSS.tif`, `_BIEXP_GAUSS.tif`) saved as float16 (reverted back from float32)
- `classes/bk_worker.py` — renamed `finished = Signal()` → `work_done = Signal()` to avoid QThread.finished conflict (was causing double print)
- `controllers/ctrl_img_proc.py` — updated `.finished.connect` → `.work_done.connect`
- `als_dff0.py` → `als_cal.py` — removed EPSILON + division; subtraction only; output `_CAL.tif`; functions renamed (`process_cal`, `_cal_output_path`)
- `functions/als_baseline.py` → `functions/als.py` — renamed `als_baseline_run` → `als_cal_run`; GPU kernel optimized (eliminated `w[]` array, float64 contamination, interior branch); single-letter variables → descriptive names
- `controllers/ctrl_als_dff0.py` → `controllers/ctrl_als_cal.py` — class `CtrlAlsCal`; `slow_fluc_3d`; `on_dff0_cal`; `btn_dff0_cal`
- `views/view_als_dff0.py` → `views/view_als_cal.py` — class `ViewAlsCal`; `btn_dff0_cal`
- `functions/__init__.py` — `als_cal_run` lazy-imported from `.als`
- `controllers/__init__.py` — `CtrlAlsCal`
- `views/__init__.py` — `ViewAlsCal`
- `main.py` — tab label "ALS Correction"; `tab_als_cal`, `view_als_cal`, `ctrl_als_cal`
- `docs/knowledgebase/als_algorithm_concepts.md` — **new** — full ALS algorithm Q&A: equation, L matrix, weights, Thomas algorithm, iteration concept, GPU kernel, thread/warp/block sizing

## Summary of current progress
- Refactored `als_dff0` → `als_cal` pipeline end-to-end: removed ΔF/F₀ division (ALS output is now pure slow-fluctuation subtraction); all naming updated (`slow_fluc`, `_CAL`, `als_cal_run`)
- Fixed double-print bug: `BackgroundWorker.finished` was shadowing `QThread.finished` C++ signal causing double emission → renamed to `work_done`
- Optimized GPU ALS kernel: eliminated one local array (48 KB → 36 KB per thread), removed float64 contamination, removed interior-loop branch
- Changed detrend output to ΔF/F₀: `(raw - trend) / trend` across all 4 kernels
- Documented the full ALS algorithm in `docs/knowledgebase/als_algorithm_concepts.md`

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Re-evaluate GAUSS TIFF dtype → float16 confirmed adequate for ΔF/F₀ values near zero

## What should we do next? (TODOs)
- [ ] Consider connecting `pick_confirmed` signal to also switch to ALS Correction tab after image processing completes

---

# Log of the project progress 2026-06-05 Thu (Session 17)

## Last working file
- `img_proc.py`

## List of modified files
- `CLAUDE.md` — updated ruff check instruction: run `C:\Users\KANG\.local\bin\ruff.exe check <files>` directly instead of `uv run ruff`
- `controllers/ctrl_als_dff0.py` — disable `btn_cal_dff0_all` before starting dF/F0 worker; re-enable in `_on_dff0_all_done`
- `controllers/ctrl_img_proc.py` — disable `btn_start_processing` before starting img_proc worker; re-enable in `_on_processing_done`; fixed `GAUSS_EXISTS?` → PROC/MODE logic for partial GAUSS existence
- `functions/detrend.py` — removed `.astype(np.float16)` from `mov_detrend` and `biexp_detrend` return paths; both now return float32 cleanly
- `img_proc.py` — GAUSS TIFFs (`_MOV_GAUSS.tif`, `_BIEXP_GAUSS.tif`) now saved as float32 instead of float16 (temporary, pending evaluation)

## Summary of current progress
- Completed all 3 carry-over TODOs from Session 16: disabled run buttons during worker execution, switched GAUSS TIFFs to float32
- Discussed float16 precision: at baseline values (~20000 counts), float16 step size = 16, meaning no sub-integer precision; detrend functions were wastefully converting to float16 mid-pipeline only to be converted back to float32 by gaussian_blur — removed those intermediate casts
- Fixed a logic bug in `check_file_status`: when only one GAUSS type existed (e.g. MOV but not BIEXP), the file was incorrectly marked SKIP/NONE; now correctly marks it YES with the missing mode

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Disable `btn_cal_dff0_all` while dF/F0 worker running; re-enable in `_on_dff0_all_done`
- ✅ Disable `btn_start_processing` while img_proc worker running; re-enable in `_on_processing_done`
- ✅ Removed float16 intermediate cast from `detrend.py` (both `mov_detrend` and `biexp_detrend`)
- ✅ Switched GAUSS TIFF save format from float16 → float32 in `img_proc.py`
- ✅ Fixed `GAUSS_EXISTS?` → PROC/MODE assignment in `check_file_status`

## What should we do next? (TODOs)
- ✅ Re-evaluate whether float32 GAUSS TIFFs should be permanent — reverted to float16 in Session 18

---

# Log of the project progress 2026-06-04 Wed (Session 16)

## Last working file
- `als_dff0.py`

## List of modified files
- `als_dff0.py` — added `emitter=None` to `process_dff0()` and `run()`; emits `{"type": "progress", "i", "total", "file"}` per file and `{"type": "step", "msg"}` for each processing stage (Loading, ALS baseline, dF/F0, Saved)
- `controllers/ctrl_als_dff0.py` — added `use_emitter=True` to `BackgroundWorker`; connected `proc_msgs` → new `_on_dff0_progress()` slot; sets `le_run_on` and `le_als_params` before worker starts; `_on_dff0_all_done()` now sets `le_processing_step` to "All done!"

## Summary of current progress
- Mirrored the `img_proc.py` emitter pattern into `als_dff0.py` and `ctrl_als_dff0.py` so the ALS dF/F0 pipeline now reports live progress to the GUI form fields (`le_run_on`, `le_als_params`, `le_curret_total`, `le_processing_file`, `le_processing_step`) in `view_als_dff0.py`
- Discussed float32 → float16 → float32 dtype pipeline: GAUSS TIFFs are saved as float16 (`img_proc.py:157,190`), read back and cast to float32 in the ALS controller. The float32 array dtype is correct but precision is float16-level (~3.3 decimal digits) — lost precision cannot be recovered. float16 max ≈ 65504 may also clip values between 65504–65535.

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Added emitter pipeline to `als_dff0.py` (`process_dff0` + `run`)
- ✅ Wired emitter in `ctrl_als_dff0.py` → GUI line edits in `view_als_dff0.py`
- ✅ Carry-over from Session 15: `le_run_on` now populated for the ALS tab as well

## What should we do next? (TODOs)
- ✅ Disable `btn_cal_dff0_all` while the dF/F0 worker is running — done in Session 17
- ✅ Disable `btn_start_processing` while the img_proc worker is running — done in Session 17
- ✅ Consider saving GAUSS TIFFs as float32 — evaluated and reverted to float16 in Session 18

---

# Log of the project progress 2026-06-04 Wed (Session 15)

## Last working file
- `functions/gaussian_blur.py`

## List of modified files
- `classes/bk_worker.py` — `proc_msgs = Signal(str)` → `Signal(object)` to support dict payloads
- `img_proc.py` — all `emitter()` calls now emit dicts (`{"type": "progress", ...}` / `{"type": "step", ...}`); removed `.astype(np.float16)` from `tifffile.imread()` — raw TIFFs now loaded as native `uint16`; added `dtype={img.dtype}` to load log
- `controllers/ctrl_img_proc.py` — `_on_progress()` rewritten to parse dict and route to form fields (`le_curret_total`, `le_mode`, `le_processing_file`, `le_processing_step`); proc list filename `proc_pick_*.txt` → `proc_*.txt` (strips `pick_` prefix via `.removeprefix()`)
- `functions/check_cuda.py` — flipped CUDA version preference: now selects 12.x before 11.x
- `functions/gaussian_blur.py` — `_gpu_kernel` (`@cuda.jit`, grid=1) commented out; replaced by `_cpu_kernel` + `cuda.to_device()` in `_gpu_gaussian_blur` to eliminate `NumbaPerformanceWarning`
- `pyproject.toml` — `required-environments` → `environments` (uv field rename)
- `uv.lock` — updated for CUDA 12 packages

## Summary of current progress
- Upgraded CUDA toolkit from 11.8 to 12.8; updated `check_cuda.py` to prefer 12.x
- Switched `proc_msgs` signal payload from `str` to `object` (dict); all emitter calls in `img_proc.py` now emit structured dicts; `_on_progress()` in `ctrl_img_proc.py` routes dict fields to correct GUI widgets
- Fixed raw TIFF loading: was incorrectly converting to `float16` on load; now reads native `uint16` and lets downstream functions handle dtype
- Fixed proc list filename to `proc_YYYYMMDD_NNN.txt` (was `proc_pick_YYYYMMDD_NNN.txt`)
- Eliminated `NumbaPerformanceWarning` (grid size 1) by computing the 37-element Gaussian kernel on CPU via `_cpu_kernel` instead of a `@cuda.jit` kernel

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Rewrote `_on_progress(msg)` to parse dict and route to correct form fields
- ✅ `le_run_on` populated with "GPU (CUDA)" or "CPU (NUMBA-JIT)" in `start_processing()`
- ✅ Fixed raw TIFF loading dtype (uint16)
- ✅ Fixed proc list filename prefix

## What should we do next? (TODOs)
- ✅ Disable `btn_start_processing` while running — done in Session 17

---

# Log of the project progress 2026-06-04 Wed (Session 14)

## Last working file
- `views/view_img_proc.py` (← break here, line 94)

## List of modified files
- `controllers/ctrl_img_proc.py` — `current_pick_list_path` → `pick_list_path`; `out_path` → `proc_list_path`; added `_on_progress(msg)` slot; wired `proc_msgs.connect(_on_progress)` in `start_processing()`; worker created with `use_emitter=True`
- `controllers/ctrl_data_selector.py` — local `current_pick_list_path` → `pick_list_path` in `pick_list_export()`
- `classes/bk_worker.py` — added `proc_msgs = Signal(str)`; added `use_emitter: bool = False` flag; when `True`, injects `emitter=self.proc_msgs.emit` as kwarg into the called function
- `img_proc.py` — `run()` accepts `emitter=None`; `process_mov()` and `process_biexp()` accept `emitter=None`; emitter called at key steps (detrend, gaussian blur, save done); `emitter` forwarded from `run()` into both `process_*` functions
- `views/view_img_proc.py` — user replaced `tb_proc_log` (QTextBrowser) with a structured `QFormLayout` (`lo_proc_info`) containing: `le_run_on`, `lbl_current_total_disp`, `le_processing_file`, `le_processing_step`

## Summary of current progress
- Completed variable renames: `current_pick_list_path` → `pick_list_path`, `out_path` → `proc_list_path`
- Built the full progress reporting pipeline: `proc_msgs = Signal(str)` in `BackgroundWorker` with opt-in `use_emitter=True` flag (so `als_dff0` is unaffected); `emitter` callback injected into `img_proc.run()` and forwarded into `process_mov()` / `process_biexp()`
- User redesigned the progress display in `view_img_proc.py`: replaced plain `QTextBrowser` with a structured form showing Run on / Current/Total / Processing file / Processing step
- ⚠️ `_on_progress` in `ctrl_img_proc.py` still calls `self.view.tb_proc_log.append(msg)` which no longer exists — **not yet wired to the new form fields**

## Completed TODOs/Tasks (before new wrap-up)
- Variable renames: `current_pick_list_path` → `pick_list_path`, `out_path` → `proc_list_path`
- Progress reporting architecture implemented end-to-end (BackgroundWorker signal → emitter callback → img_proc pipeline)
- Previous session TODO "Think about how to hint processing progress in the GUI" — architecture decided and partially implemented

## What should we do next? (TODOs)
- ✅ Rewrite `_on_progress(msg)` to parse emitted messages and route to form fields — done in Session 15
- ✅ Populate `le_run_on` with "GPU" or "CPU" — done in Session 15
- ✅ Disable `btn_start_processing` while running — done in Session 17

---

# Log of the project progress 2026-06-03 Tue (Session 13)

## Last working file
- `controllers/ctrl_img_proc.py` (← break here)

## List of modified files
- `controllers/ctrl_img_proc.py` — `export_checked_list` → `export_proc_list`; output filename `{stem}_checked.txt` → `proc_{stem}.txt`; `load_pick_list` guard updated (reject `proc_` prefix); `start_processing()` refactored: removed dialog, auto-exports proc list if missing, passes `out_path` directly; `_on_processing_done` now calls `check_file_status()` after processing
- `controllers/ctrl_als_dff0.py` — `_brief_path` → `_proc_list_path`; `on_load_checked_brief` → `on_load_proc_list`; `_load_gauss_tiffs_from_brief` → `_load_gauss_tiffs_from_proc_list`; imports updated; log message updated
- `views/view_img_proc.py` — `btn_export_checked_list` → `btn_export_proc_list`, button label "Export Proc List"
- `views/view_als_dff0.py` — `btn_load_checked_brief` → `btn_load_proc_list`, button label "Load Processing List"; removed unused `UISizes` import
- `classes/dialog_get_path.py` — `get_checked_brief()` → `get_proc_list()`; name filter updated to `proc_*.txt`
- `img_proc.py` — `parse_brief` → `parse_proc_list`; `update_brief_gauss_exists` → `update_proc_list_gauss_exists`; `brief_path` → `proc_list_path`; CLI `--brief` → `--proc_list`; section headers and log messages updated
- `als_dff0.py` — same renames as `img_proc.py`; `_parse_brief_for_gauss` → `_parse_proc_list_for_gauss`; CLI `--brief` → `--proc_list`
- `docs/continue_from_here.md` — cleaned up fake/completed TODOs

## Summary of current progress
- Completed full rename of "Export Checked List" / `_checked.txt` → "Export Proc List" / `proc_*.txt` across views and controllers
- Refactored `start_processing()`: no longer uses a file dialog — auto-exports the proc list if not yet saved, then runs the pipeline directly
- `_on_processing_done` now refreshes the pick list table via `check_file_status()` after processing completes
- Full "brief" → "proc_list" / "processing" terminology rename across all pipeline files and controllers
- Confirmed: tab-switch on export (`pick_confirmed` → Image Processing tab) was already wired in `main.py`

## Completed TODOs/Tasks (before new wrap-up)
- `export_checked_list` → `export_proc_list` with new `proc_` filename scheme
- `start_processing()` refactored: dialog removed, auto-export, direct `out_path` pass
- All "brief" naming eliminated across codebase
- Fake/stale TODOs removed from tracker

## What should we do next? (TODOs)
- ✅ Progress hinting in GUI — implemented in Sessions 14–15

---

# Log of the project progress 2026-06-01 Sun (Session 12)

## Last working file
- `controllers/ctrl_img_proc.py` (← break here, line 231)

## List of modified files
- `views/view_data_selector.py` — all "processing_brief" widget names → `pick_list_*`; GroupBox "Processing Brief" → "Pick List"; button "Export Processing Brief" → "Export Pick List"
- `controllers/ctrl_data_selector.py` — methods renamed (`brief_gen` → `pick_list_gen`, `brief_export` → `pick_list_export`, etc.); `_parse_pick_list_header` moved into class; `pick_confirmed = Signal(str)`; added `super().__init__()`; emits path on export
- `controllers/ctrl_img_proc.py` — `load_pick_list` accepts optional `path_str: str = ""`; `current_brief_path` → `current_pick_list_path`; `brief_line` → `pick_line`; `brief_path` → `pick_list_path`; dialog titles updated
- `controllers/ctrl_als_dff0.py` — dialog title "Select a Checked Brief" → "Select a Checked Pick List"
- `img_proc.py` — docstring CLI example `proc_brief_*` → `pick_*`
- `als_dff0.py` — docstring CLI example `proc_brief_*` → `pick_*`
- `main.py` — connected `pick_confirmed` signal to `ctrl_img_proc.load_pick_list`

## Summary of current progress
- Completed full rename of "Processing Brief" / `proc_brief_*` → "Pick List" / `pick_*` across all relevant files
- `pick_confirmed = Signal(str)` added to `CtrlDataSelector`; emits exported file path
- `load_pick_list` in `ctrl_img_proc` now accepts optional path — skips dialog when path is provided (auto-load on export)
- Tab-switching on export (`pick_confirmed` → switch to `tab_im_proc`) discussed but not yet applied to `main.py`

## Completed TODOs/Tasks (before new wrap-up)
- Renamed all "Processing Brief" / `proc_brief_*` references to "Pick List" / `pick_*` across view, controllers, and pipeline scripts
- Wired `pick_confirmed` signal to auto-load pick list in Image Processing tab on export

## What should we do next? (TODOs)
- ✅ Add tab-switch in `main.py` on `pick_confirmed` — confirmed already wired in Session 13
- ✅ Fix `pick_list_path` variable shadowing — resolved in Session 13
- ✅ Redesign `start_processing()` — refactored in Session 13

---

# Log of the project progress 2026-05-27 Wed (Session 11)

## Last working file
- `als_dff0.py`

## List of modified files
- `als_dff0.py` — reorganized to follow the `img_proc.py` pipeline structure more closely:
  - kept `run()` as the GUI/CLI-callable pipeline runner
  - added `process_dff0()` for one-file ALS baseline + dF/F0 processing
  - kept CPU/GPU routing delegated to `functions.als_baseline_run(..., cuda_available)`
  - added ordered de-duplication of GAUSS TIFF paths from the checked brief
  - added timing and memory logs similar to `img_proc.py`
  - added `EPSILON` guard when dividing by baseline to reduce `inf`/`nan` risk
  - kept CLI fallback behavior with `check_cuda() if check_cuda is not None else (False, "CUDA not available")`

## Summary of current progress
- Read and compared: `img_proc.py`, `functions/detrend.py`, `functions/gaussian_blur.py`, `functions/als_baseline.py`, `als_dff0.py`, `controllers/ctrl_img_proc.py`, `controllers/ctrl_als_dff0.py`
- Confirmed hardware routing pattern: controllers / CLI call `check_cuda()` once → `run()` receives `cuda_available` → pipeline functions pass the flag down → dispatchers choose GPU or CPU
- Confirmed fallback exists for all four entry points

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Reorganized `als_dff0.py` to mirror `img_proc.py` pipeline structure

## What should we do next? (TODOs)
- (none)

---

# Log of the project progress 2026-05-26 Mon (Session 10)
Last working file: `views/view_img_proc.py`
Last working line: 69

## List of modified files:
- `img_proc.py` — removed `log_path` param and console-swap from `run()`
- `als_dff0.py` — removed `log_path` param and console-swap from `run()`
- `controllers/ctrl_img_proc.py` — removed `_proc_log_path`, `fileChanged` watcher, `_on_log_changed`, all log file logic; `_on_processing_done` simplified
- `controllers/ctrl_als_dff0.py` — removed `log_path` creation and passing in `on_cal_dff0_all()`
- `views/view_img_proc.py` — removed old `tb_console`/`lbl_console`; `tb_proc_log` (QTextBrowser) now present in block 2
- `classes/bk_worker.py` — simplified by user
- `functions/tau_estimate.py` — modified by user
- `data/*_log.txt` (x3) — deleted stale log files

## Summary of current progress
- Removed entire `_log.txt` + `QFileSystemWatcher` mechanism from codebase
- `console.log()` now outputs to terminal only — clean and simple
- `tb_proc_log` (QTextBrowser) exists in `view_img_proc.py` block 2 but is not yet wired to receive messages
- Architecture clarified: `run()` = GUI-callable pipeline runner; `if __name__ == "__main__":` = CLI entry point

## Completed TODOs/Tasks (before new wrap-up)
- Removed all `_log.txt`, `tb_console`, and `QFileSystemWatcher` code from: `img_proc.py`, `als_dff0.py`, `ctrl_img_proc.py`, `ctrl_als_dff0.py`, `view_img_proc.py`

## What should we do next? (TODOs)
- (none)
