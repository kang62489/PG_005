# Log of the project progress 2026-04-28 Mon (Session 3)

Last working file: run_biexp_detrend.py
Last working line: ~1 (reviewing only)

# List of modified files:
- demo_als_biexp.py (deleted)
- demo_biexp_detrend.py (deleted)
- run_als_1d.py (removed savefig call + out_png variable)
- run_biexp_detrend.py (removed noqa: INP001)
- classes/helper_checkable_dropdown.py (removed unnecessary noqa: N802)
- classes/plot_results.py (removed unnecessary noqa: C901, PLR0915)
- classes/results_exporter.py (removed unnecessary noqa: ANN401)
- functions/kmeans.py (removed unnecessary noqa: C901, PLR0912, PLR0915)
- functions/spike_centered_processes.py (removed unnecessary noqa: E741)

## Summary of current progress
- Reviewed all 4 standalone scripts (demo_als_biexp, demo_biexp_detrend, run_als_1d, run_biexp_detrend)
- Deleted demo_als_biexp.py and demo_biexp_detrend.py — no longer needed
- Kept run_als_1d.py and run_biexp_detrend.py for future integration into CPU/GPU pipeline
- Removed 7 unnecessary noqa comments across multiple files (RUF100)
- Removed savefig calls from run_als_1d.py
- Fixed 3 C408 ruff issues (dict() → literal) in demo files before deletion
- Discussed differences: biexp detrend (run_biexp_detrend.py) vs moving-average detrend (cpu/gpu_process.py)

## Completed TODOs (from last session)
- (none — this session was focused on code review and cleanup)

## What should we do next? (TODOs)
- [ ] Wire btn_start_processing — run processing based on PROC column values
- [ ] Plan merging biexp detrend into CPU/GPU pipeline (run_biexp_detrend.py → functions/)
