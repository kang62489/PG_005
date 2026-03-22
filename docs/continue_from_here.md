# Log of the project progress 2026-03-20 Fri 22:16:57

Last working file: controllers/ctrl_check_list.py
Last working line: 53 (end of load_pick_list method)

## List of modified files (this session):
- controllers/ctrl_check_list.py (new — implemented `load_pick_list()`)

## Summary of current progress (based on modified files, existing plans)
- Implemented `load_pick_list()` in `CtrlCheckList` (Step 1 of the plan at `.claude/plans/spicy-strolling-quokka.md`)
  - Reads `data/pick_list.json` via `pd.read_json(..., orient="records", dtype=str)`
  - Parses `Filename` → `DOR` (date part) + `TIFF_SERIAL` (serial number, no .tif)
  - Builds check DataFrame with columns: `DOR`, `TIFF_SERIAL`, `IMG_READY`, `PREPROC`, `PREPROC_READY`
  - Sets `ModelFromDataFrame` on `tv_check_list`
  - Handles empty pick list case gracefully

## Completed TODOs/Tasks
- [x] Plan for `CtrlCheckList` — approved by user (previous session)
- [x] Implement `load_pick_list()` — Step 1 of the plan

## What should we do next? (TODOs)
- [ ] Test `load_pick_list()` in the GUI (load some pick list entries, click "Load Pick List")
- [ ] Implement "Start Check" — verify file existence on disk and populate `IMG_READY`, `PREPROC`, `PREPROC_READY`
- [ ] Implement browse directories (raw_images, raw_abfs, processed_images)
- [ ] Implement "Remove Selected" / "Clear List" functionality
- [ ] Consider ABF_SERIAL / ABF_READY columns (currently ignored)

## Messages from you
- (none)
