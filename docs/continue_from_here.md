# Log of the project progress 2026-04-15 Tue 17:30:00

Last working file: controllers/ctrl_dor_query.py
Last working line: ~235

# List of modified files
- classes/__init__.py
- classes/dialog_confirm.py (new file)
- classes/dialog_get_path.py (new file)
- controllers/ctrl_dor_query.py (<- Break here, line ~235)
- views/view_dor_query.py
- utils/params.py

## Summary of current progress

### classes/__init__.py — Added DialogGetPath export
- Added direct (non-lazy) import for `DialogGetPath` alongside `DialogConfirm`
- Added `DialogGetPath` to `__all__`

### ctrl_dor_query.py — btn_scan_files & te_file_structure fully wired
- `scan_files()` implemented: opens `DialogGetPath`, guards DOR mismatch (folder name must contain DOR string), recursively scans with `rglob("*")`, counts file extensions via `collections.Counter`, writes summary to `# Folder Structure` section in `Data_{dor}.md`
- `load_data_md` now parses `# Folder Structure` and displays content in `te_file_structure`; shows `"no scanning results"` if section is empty or missing
- `btn_scan_files` enabled/disabled in sync with log file existence
- Mismatch between selected folder and DOR shown in `te_file_structure`

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Wire up `btn_scan_files` → `scan_files()` method
- ✅ Parse and display `# Folder Structure` in `te_file_structure` on load
- ✅ Export `DialogGetPath` from `classes/__init__.py`
- ✅ DOR mismatch guard with message shown in `te_file_structure`
- ✅ Fixed scan to be recursive (`rglob` instead of `glob`)

## What should we do next? (TODOs)
- [ ] Transfer loading of rec_tables to `ctrl_pick_list.py` (currently commented out in `ctrl_dor_query.py`)
