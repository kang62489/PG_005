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
