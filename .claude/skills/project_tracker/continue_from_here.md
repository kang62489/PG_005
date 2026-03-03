# Log of the project progress 2026-03-03 Mon

List of modified files:
- `views/view_pick_raws.py` (new — filter panel + rec summary layout)

## What have we done? (Summary of current progress)
- Implemented filter panel UI in `views/view_pick_raws.py` (`setup_block_1`):
  - `QGroupBox("Filter Panel")` containing 3 columns: OBJ, EXC, EMI
  - Each column has a `QListWidget` + `[All]` `[Clear]` buttons
  - `Reset All Filters` button at the end of the panel
- User revised layout: split into `setup_block_1` (filter panel) and `setup_block_2` (rec summary + add button)
- `setup_block_2` has `tv_rec_summary` (QTableView) and `btn_add_to_analysis_list` (QPushButton)
- Discussed how to access dynamically created buttons from a controller:
  - Option 1: `self` dicts (`self.btn_all["OBJ"]`)
  - Option 2: `setObjectName` + `findChild`
  - Option 3 (best combo): `setattr(self, f"btn_all_{col}", btn)` — sets both `self.btn_all_OBJ` attribute AND pairs naturally with `setObjectName` for QSS styling

## What should we do next? (TODOs)
- [ ] **Complete the layouts of current tabs first** ← start here!
  - `tab_dor_query` ("Query by DOR") — query the database by DOR
  - `tab_abf_preview` ("ABF Quick Check") — quick preview of ABF files
  - `tab_analysis_list` ("Analysis List") — list of analyses to run/review
- [ ] Wire up filter panel buttons (All / Clear / Reset All Filters) in `controllers/ctrl_pick_raws.py`
- [ ] Populate filter list widgets (OBJ, EXC, EMI) with data from database
- [ ] Fix documentation discrepancies in `docs/PROJECT_SUMMARY.md` and `docs/DEPENDENCY_DIAGRAM.md`

## Messages from you
- Complete the layouts of current tabs first.
