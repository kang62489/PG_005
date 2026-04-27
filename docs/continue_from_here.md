# Log of the project progress 2026-04-27 Sun

Last working file: controllers/ctrl_img_proc.py
Last working line: ~145

# List of modified files:
- controllers/__init__.py (renamed CtrlImgPreproc → CtrlImgProc)
- controllers/ctrl_img_proc.py (<- Break here, ~line 145)
- views/__init__.py (renamed ViewImgPreproc → ViewImgProc)
- views/view_img_proc.py (renamed ViewImgPreproc → ViewImgProc)
- main.py (renamed all preproc references → proc, tab label updated)

## Summary of current progress
- Renamed ImgPreproc → ImgProc across all files (ctrl, view, __init__ x2, main.py)
- Redesigned check_file_status(): split into _raw_tiff_ready, _cal_exists, _gauss_exists
- Used regex CAL_PATTERN (Biexp|Mov) to detect processing type from filenames
- Fixed load_pick_list() empty case not clearing self.df_check_list
- Added PROC column (YES/SKIP) using chained .with_columns() after GAUSS_EXISTS? is resolved
- Fixed polars bug: pl.col referencing a newly computed column in the same with_columns() call

## Completed TODOs (from last session)
- Apply check_file_status() fix (map_elements, correct variable names, remove .clone())
- Re-think check_file_status() to handle Cal, Gauss (Mov), and BiExp separately

## What should we do next? (TODOs)
- [ ] Implement editable PROC column (dropdown YES/SKIP) via QStyledItemDelegate + ModelFromDataFrame editable support
- [ ] Wire btn_start_processing to run processing based on PROC column values
- [ ] Add setup_block_2 preview plot for bleach correction

## Extra Notes / Ideas
- 💡 Consider adding a DirWatcher (QFileSystemWatcher on proc dir) for automatic file status refresh
