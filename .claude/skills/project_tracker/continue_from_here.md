# Log of the project progress 2026-03-04 Wed 16:04:07

List of modified files:
- `controllers/ctrl_dor_query.py` (refactored: QStandardItemModel → pandas + QSqlTableModel; clear tv_injections on DOR switch; resizeColumnsToContents for tv_injections)

## What have we done? (Summary of current progress)
- Full rewrite of `ctrl_dor_query.py`:
  - Replaced `QStandardItemModel`/`QStandardItem` loops with `QSqlTableModel` + column hide loop
  - Added `QSqlDatabase` (opened once in `__init__`) shared by both animal and injection models
  - Added `self.df_animals` / `self.df_injections` as pandas DataFrames (read via `pd.read_sql`)
  - `load_injections` now reads `animal_id` from `df_animals.iloc[row]` instead of Qt model item
  - `tv_injections.resizeColumnsToContents()` added after hide loop for auto column width
  - `tv_injections.setModel(None)` + reset `df_injections` at top of `load_animals` to clear stale data on DOR switch

## What should we do next? (TODOs)
- [ ] **Add `load_rec_summary` in `ctrl_dor_query.py`** ← start here!
  - Triggered by DOR selection (inside `load_animals`)
  - Data source: `data/rec_data.db` (need to inspect table/columns first)
  - Need to add a display widget (`tv_rec_summary`) to `ViewDorQuery` as well
- [ ] Fix remaining ruff problem in `utils/params.py` (DTZ005 was fixed; one more violation still unidentified)
- [ ] Complete layouts of other tabs: `tab_abf_preview`, `tab_analysis_list`
- [ ] Wire up filter panel buttons (All / Clear / Reset All Filters) in `controllers/ctrl_pick_raws.py`
- [ ] Populate filter list widgets (OBJ, EXC, EMI) with data from database

## Messages from you
- For `load_rec_summary`: check `data/rec_data.db` schema first (tables + columns), then add widget to view, then implement controller method.
