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
- [ ] **[NEXT]** Fix `console.log` message display in `tb_proc_log` (`views/view_img_proc.py`)
