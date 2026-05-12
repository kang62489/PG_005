# Log of the project progress 2026-05-13 Tue

Last working file: controllers/ctrl_img_proc.py
Last working line: ~255

# List of modified files:
- batch_process.py
- classes/abf_clip.py
- classes/helper_cell_dropdown.py
- classes/model_from_dataframe.py
- controllers/ctrl_img_proc.py (<- Break here, ~line 255)
- im_preprocess.py
- run_als_baseline.py
- run_biexp_detrend.py
- test_batch.py

## Summary of current progress
- Filtered out `_checked.txt` files from `load_pick_list` dialog
- Added MODE column (BIEXP/MOV/BOTH/NONE) to the pick list table with its own dropdown delegate
- `_on_proc_changed`: MODE auto-updates when PROC changes (YES→BIEXP, SKIP→NONE)
- MODE cell is disabled (not editable) when PROC is SKIP
- `export_checked_list` now writes `[filename, PROC, MODE]` format
- Both `load_pick_list` and `export_checked_list` now parse between `Picked:` and `Total...` lines
- Delegate auto-commits on selection (`activated` signal); fixed "editor not belong to view" warning
- Unified ALL_CAPS file suffix convention: `_MOV_CAL`, `_MOV_GAUSS`, `_BIEXP_CAL`, `_BIEXP_GAUSS`, `_BIEXP_BASELINE`, `_BIEXP_DFF0` across all scripts
- `classes/abf_clip.py` `load_img` default updated to `"MOV_GAUSS"`

## Completed TODOs (from last session)
- Added DETREND/MODE column to check_list — covered by new MODE column
- Consistent file naming convention enforced across all scripts

## What should we do next? (TODOs)
- [ ] **[NEXT]** Modify and merge `im_preprocess.py` and `run_biexp_detrend.py` into a MODE-selection-oriented structure — single entry point that dispatches to MOV or BIEXP pipeline based on MODE value
- [ ] Wire `btn_start_processing` — trigger processing based on PROC/MODE column values from checked list
- [ ] Make preprocessing scripts load filenames from `*_checked.txt` instead of hardcoded lists
- [ ] Plan merging biexp detrend into CPU/GPU pipeline (`run_biexp_detrend.py` → `functions/`)
- [ ] Implement ALS baseline estimation for dF/F0 calculation
- [ ] Decouple Gaussian blur from detrending — separate functions
- [ ] Archive original Mov pipeline — keep usable but clearly separated for PI reference
