# Log of the project progress 2026-04-27 Sun

Last working file: controllers/ctrl_img_preproc.py
Last working line: ~95

# List of modified files:
- controllers/ctrl_data_selector.py
- controllers/ctrl_dor_query.py
- controllers/ctrl_img_preproc.py (<- Break here, ~line 95)
- views/view_img_preproc.py

## Summary of current progress
- Set all `tv_` table views in DOR Query and Data Selector tabs to uneditable (`NoEditTriggers`)
- Merged `_init_dir_fields()` directly into `__init__` in `ctrl_img_preproc.py`
- Added `setup_block_2()` with `w_preview_corr` plot container placeholder in `view_img_preproc.py`
- Learned and documented Polars expressions (`docs/knowledgebase/polars_expressions.md`)
- Identified multiple bugs in `check_file_status()` — not yet applied

## Completed TODOs (from last session)
- ✅ Set all tv_ tableviews in Query by DOR and Data Selector to uneditable

## What should we do next? (TODOs)
- [ ] Apply `check_file_status()` fix (map_elements, correct variable names, remove .clone())

## Extra Notes / Ideas
- ⚠️ **Re-think `check_file_status()`** — there are multiple types of preprocessed TIFFs: **Cal**, **Gauss (mov)**, and **BiExp**. The current logic only checks for one `_preproc.tif` file. Need to redesign status checking to handle all three types.
