# Log of the project progress 2026-06-20 Sat (Session 29)
Last working file: `spike_analysis.py`
Last working line: 156 (`region_analyzer = RegionAnalyzer(categorizer.categorized_frames, obj=obj)`)

## List of modified files
- `utils/params.py` — added `ColumnSorter` dataclass (`UISizes`-style, accessed without instantiation): `CORE_COLUMNS`, `FLUIDIC_COLUMNS`, `IMG_COND_COLUMNS`, `EPHY_COND_COLUMNS`, `OTHER_COLUMNS`, `MEMO_COLUMNS`, `IGNORE_COLUMNS` — derived by actually inspecting column frequency across all 56 `REC_*` tables in `rec_data.db`, not guessed
- `functions/query_databases.py` — **new**: `lookup_rec_from_db(table, db_path, exp_db_path)` batch-queries `rec_data.db` grouped by date table (one query per `REC_{date}`, not per row), reads each table's real columns via `PRAGMA table_info` first (schemas vary by date), diagonally concatenates so missing columns become null; chains `_sort_rec_columns` (drops `IGNORE_COLUMNS`, orders the rest by `ColumnSorter` group priority) and `populate_animal_id_values` (fills missing `ANIMAL_ID` from `exp_info.db`'s `BASIC_INFO` DOR→Animal_ID mapping; handles 1-candidate "fill all" and 2-candidate "fill by elimination" cases) as its last two steps
- `functions/__init__.py` — registered `lookup_rec_from_db`/`populate_animal_id_values` in the lazy-import table
- `controllers/ctrl_data_selector.py` — local `CORE_COLUMNS` tuple replaced with `ColumnSorter.CORE_COLUMNS` import
- `spike_analysis.py` — deleted `_lookup_obj` entirely; `parse_ana_list` now only parses + existence-filters the ana list (returns `entries: pl.DataFrame` with `proc_tiff_path`/`raw_abf_path` columns added, no DB lookup); `__main__` computes `ref_df = lookup_rec_from_db(entries, db, exp_db)` once, then queries it **live per row** (`ref_df.filter(pl.col("Filename") == ...)`) for `OBJ` instead of flattening into a lookup dict — kept intentionally so `ref_df` stays a full, separately-queryable/saveable table rather than being collapsed away
- `.claude/settings.local.json` — new permission entries accumulated from today's tool calls

## Summary of current progress
- Generalized the old per-row, single-column `_lookup_obj` into a batched, schema-tolerant, multi-column `rec_data.db` lookup, with column ordering now policy-driven (`ColumnSorter`) instead of whatever SQL happened to return
- Added cross-database enrichment: `ANIMAL_ID` gaps in the compiled table get backfilled from `exp_info.db` using a clear, verified 1-or-2-candidate rule (confirmed via real data that no DOR ever has more than 2 animals)
- Relocated all three DB-query functions out of `spike_analysis.py` into `functions/query_databases.py`, matching the project's existing public/private function-module convention
- Rewired `parse_ana_list`/`__main__` end-to-end so `obj` resolution goes through the new batched lookup instead of `_lookup_obj`; verified against `ana_list_20260601_000.txt` (correctly resolved `OBJ='10X'`) — the only failure hit afterward was a missing test `.abf` file on disk, unrelated to the refactor
- Hit a rough patch mid-session: left `parse_ana_list` and `__main__` in a mismatched state (signature/return-type changed in one but not the other) after a couple of rejected edits — caught and fully fixed within this session, verified by re-running the script

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
3. **New temporal trace analysis** — use the **spike frame's** dim/bright masks (fixed, not per-frame) to compute mean z-score within each mask across every frame in the segment → `dim_trace`/`bright_trace`. Find each trace's peak → latency = time between bright-peak and dim-peak. `combined = bright_trace + dim_trace` → measure recovery duration (time to decay back near baseline; exact closeness threshold still TBD).
4. **Constrain dim region to spatially relate to bright region** — `RegionAnalyzer._find_largest` currently finds largest-dim and largest-bright independently (could be unrelated blobs). Fix: find largest bright first, then the dim region must be the largest dim CC **whose centroid falls inside the bright region's mask** (discard other dim CCs even if larger).
5. **Red-channel soma-distance analysis** — replace `_CAT.tif` export with a contour overlay (bright+dim outlines) drawn on the corresponding `EMI=RED` recording (labels the patched cell body), then measure centroid-to-soma distance. Soma position method (user-specified): define an ROI, apply Otsu threshold within it, then `skimage.measure.regionprops` centroid to find the cell center — not fully automatic on the whole frame, and not manual-click.

## What should we do next? (TODOs)
- [ ] Build new export-oriented static-figure plotting functions; drop the old interactive `Plot*` classes from the new pipeline (item 2 above)
- [ ] Implement spike-frame-anchored `dim_trace`/`bright_trace` temporal analysis + peak-to-peak latency + recovery-duration metric (item 3 above)
- [ ] Constrain `RegionAnalyzer`'s dim-region selection to require centroid-inside-bright-mask (item 4 above)
- [ ] Build ROI + Otsu + `regionprops`-centroid soma detection on `EMI=RED` images, then contour-overlay export + centroid-to-soma distance calc (item 5 above)
- [ ] Implement actual `ref_df` saving — format (CSV vs parquet) and location/naming convention were both explicitly deferred today ("don't think about that now"); only the live-query usage was built, not persistence
- [ ] Add `ANIMAL_ID` column + derived `CELL_KEY` to `ResultsExporter`'s `experiments` SQLite table (see export/output design idea above)
- [ ] Implement the compact filename code (`A{n}S{slice}C{site}`) in `ResultsExporter`'s export filenames
- [ ] Wire `ResultsExporter.export_all(...)` into `spike_analysis.py`'s per-row loop — `AbfClip.get_export_data()` / `RegionAnalyzer.get_summary()`/`get_results()` already match its expected input shape, just never called from the new pipeline
- [ ] Decide: should `ResultsExporter` write to the ana_list's `dir_results`, or keep its own separate `results/` root? (open design question, blocks the wiring step above)
- [ ] Archive `im_dynamics.py`, `batch_process.py`, `test_batch.py` once the new pipeline + exporter wiring is confirmed working (long-standing carry-over TODO)

## Last Session Recap
※ recap: Exported `ColumnSorter` via `utils/__init__.py` (small cleanup), then designed (no code) two sets of fixes: 5 analysis-correctness items (split-range thresholding, export-ready plots, spike-frame-anchored dim/bright temporal traces + latency/recovery metrics, centroid-in-bright dim constraint, RED-channel soma-distance analysis) prioritized **ahead of** the export/output redesign (cell-grouping via DB not folders, compact batch-relative filename codes). Nothing implemented yet — next session starts with item 1 (split-range thresholding).

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
