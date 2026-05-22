# Log of the project progress 2026-05-22 Thu (Session 8)
Last working file: `controllers/ctrl_img_proc.py`

## List of modified files:
- `classes/__init__.py` — added `BackgroundWorker` to `_LAZY_IMPORTS`
- `classes/bk_worker.py` — new: `BackgroundWorker(QThread)` wraps any callable, emits `finished`
- `controllers/__init__.py` — added `CtrlAlsDff0`
- `controllers/ctrl_als_dff0.py` — new: wires `cb_switch_roi` → switches `QStackedLayout` index
- `controllers/ctrl_img_proc.py` — wired `btn_start_processing`; added processing log system (`_proc_log_path`, `_on_log_changed`, `_on_processing_done`)
- `functions/__init__.py` — moved `check_cuda`/`test_cuda` from eager try/except to `_LAZY_IMPORTS`
- `functions/check_cuda.py` — `check_cuda()` now returns `(bool, str)`; keeps all `console.log()` unchanged, adds `messages.append()` alongside each
- `img_proc.py` — `run()` accepts `log_path: Path | None = None`; swaps module-level `console` with line-buffered file Console for duration of run; CLI entry point updated to unpack `(bool, str)` from `check_cuda()`
- `views/__init__.py` — added `ViewAlsDff0`
- `views/view_als_dff0.py` — new: block 1 (load checked brief, table); block 2 (ALS config, QStackedLayout with 5 MplCanvas)
- `views/view_img_proc.py` — (no new changes this session)

## Summary of current progress
- `btn_start_processing` fully wired: opens dialog → check_cuda → BackgroundWorker → run_img_proc
- Real-time processing log: cuda messages + all `console.log()` in `img_proc.py` written to `{brief_stem}_log.txt` in `data/`; `QFileSystemWatcher.fileChanged` → `_on_log_changed` → `tb_console` updates live
- Startup speed improved: numba (`check_cuda`), matplotlib (`MplCanvas`), numpy/tifffile (`img_proc`) all deferred
- `view_als_dff0.py` block 2 implemented: ALS config controls, `cb_switch_roi` ComboBox, 5 `MplCanvas` in `QStackedLayout`
- `ctrl_als_dff0.py` wires ROI switching
- **Verified working on a real file** (2025_11_08-0028, BIEXP mode, RTX 3070, CUDA=True)

## Key architecture notes
- `check_cuda()` returns `(bool, str)` — caller writes the str to log before starting worker
- `img_proc.run()` uses `global console` + `console = Console(file=log_file, ...)` to redirect all `console.log()` in `process_biexp`/`process_mov` to file automatically (no signature changes needed in those functions)
- `_proc_log_path` initialized in `__init__`; log file added to `dirs_watcher` at start, removed on done
- `BackgroundWorker` already forwards `**kwargs` so `log_path=` is passed through transparently

## What should we do next? (TODOs)
- [ ] **[NEXT]** Wire `btn_run_als_test` and `btn_cal_dff0_all` in `ctrl_als_dff0.py`
- [ ] **[NEXT]** Load checked brief in `ctrl_als_dff0.py` → populate `tv_checked_brief`
- [ ] MODE Option A: rename BIEXP/MOV → BIEXP_DFF0/MOV_DFF0 for full pipeline; BIEXP/MOV = blur only (no ALS)
  - Update MODE dropdown options in `_set_mode_delegate`: ["BIEXP", "BIEXP_DFF0", "MOV", "MOV_DFF0", "NONE"]
  - Update `_on_proc_changed` default mode (currently hardcoded "BIEXP")
  - Split `process_biexp()` and `process_mov()` in `img_proc.py` into blur-only vs full pipeline
  - Update `run()` dispatcher in `img_proc.py`
- [ ] Update `GAUSS_EXISTS?` check → also check for `_DFF0.tif` existence
- [ ] `_gpu_mov` coalesced memory — still non-coalesced (separate future task)
