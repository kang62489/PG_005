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
