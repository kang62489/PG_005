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
- [ ] Re-evaluate whether float32 GAUSS TIFFs should be permanent — compare file size vs precision tradeoff on real data (float32 = 2× disk space vs float16)

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
- [ ] Disable `btn_cal_dff0_all` while the dF/F0 worker is running; re-enable in `_on_dff0_all_done`
- [ ] Disable `btn_start_processing` while the img_proc worker is running; re-enable in `_on_processing_done` (carry-over from Session 15)
- [ ] Consider saving GAUSS TIFFs as float32 instead of float16 — float16 clips values > 65504 and loses sub-integer precision; evaluate if this matters for ALS accuracy

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
- [ ] *(Optional)* Disable `btn_start_processing` while running; re-enable in `_on_processing_done`

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
- [ ] **[NEXT]** Rewrite `_on_progress(msg)` in `ctrl_img_proc.py` to parse emitted messages and route to the correct form fields:
  - `[i/total] filename  MODE=...` → parse into `lbl_current_total_disp` ("i/total") + `le_processing_file` (filename)
  - `  Detrending...` / `  Gaussian blur...` / `  ✓ Saved...` → `le_processing_step`
- [ ] Populate `le_run_on` with "GPU" or "CPU" when `start_processing()` is called
- [ ] Optionally: disable `btn_start_processing` while running; re-enable in `_on_processing_done`

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
- [ ] **[NEXT]** Think about how to hint processing progress in the GUI — e.g. how many files are left, which file is currently being processed (progress bar, status label, button state, etc.)

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
- [ ] **[NEXT]** Add tab-switch in `main.py`: connect `pick_confirmed` → `lambda _: self.w_main.setCurrentWidget(self.tab_im_proc)`
- [ ] Fix the variable name `pick_list_path` in `ctrl_img_proc.start_processing()` — currently shadowed/confusing with the local dialog result
- [ ] Redesign `start_processing()` in `ctrl_img_proc.py` — review flow and responsibilities

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
- Read and compared:
  - `img_proc.py`
  - `functions/detrend.py`
  - `functions/gaussian_blur.py`
  - `functions/als_baseline.py`
  - `als_dff0.py`
  - `controllers/ctrl_img_proc.py`
  - `controllers/ctrl_als_dff0.py`
- Confirmed hardware routing pattern:
  - controllers / CLI call `check_cuda()` once
  - `run()` receives `cuda_available`
  - pipeline functions pass the flag down
  - function dispatchers (`mov_detrend`, `biexp_detrend`, `gaussian_blur_run`, `als_baseline_run`) choose GPU or CPU
- Confirmed fallback exists for:
  - `btn_start_processing` in `controllers/ctrl_img_proc.py`
  - `btn_run_als_test` in `controllers/ctrl_als_dff0.py`
  - `btn_cal_dff0_all` in `controllers/ctrl_als_dff0.py`
  - CLI entry points in `img_proc.py` and `als_dff0.py`
- Note: `btn_run_als_test` has CPU fallback, but currently ignores the CUDA diagnostic message (`cuda_available, _ = check_cuda()`).

## Validation
- `uv run ruff check als_dff0.py` passed
- `uv run python -m py_compile als_dff0.py` passed
- Did not run actual TIFF processing because it would execute the heavy ALS pipeline on real image stacks.

## What should we do next? (TODOs)
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
