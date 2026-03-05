# Log of the project progress 2026-03-05 Thu 17:00:00

List of modified files:
- `views/view_dor_query.py` (typo fix: `filter_comumns` → `filter_columns`, lines 28, 74)
- `controllers/ctrl_dor_query.py` (typo fix: `filter_comumns` → `filter_columns`, lines 43, 65, 84, 101, 116)

## What have we done? (Summary of current progress)
- Fixed typo `filter_comumns` → `filter_columns` in 7 places across 2 files (view + controller for DOR query tab)

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
- (none)
