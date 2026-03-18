# Log of the project progress 2026-03-18 Wed 16:17:00

List of modified files (this session):
- controllers/ctrl_dor_query.py (fixed bug: `check_pick_list` was saving unsorted data instead of sorted `self.df_pick_list`)

55+ files total modified (uncommitted) — includes formatting changes across codebase + GUI work from prior sessions.

## What have we done? (Summary of current progress)
- Initiated GUI building on the `gui` branch (ACID - Analyzer for Cholinergic Influence Domain)
- Created main window with QTabWidget layout (4 tabs: Query by DOR, Check Pick List, Image Preprocessing, Spike Alignment Analysis)
- Built DOR query tab with controller `CtrlDorQuery`:
  - Loads DORs from `exp_info.db`
  - Loads animals & injections on selection
  - `tv_rec_summary` with filter dropdowns (OBJ, EXC, EMI) + column visibility toggle
  - Filter buttons (All / Clear / Reset) all wired up
  - `pick_selected` button saves selected rows to `pick_list.json` (sorted by Filename)
- Fixed `CheckableDropdown` checkbox double-click issue
- Fixed bug in `check_pick_list`: was saving unsorted `selected_row_data` instead of sorted `self.df_pick_list`
- Reviewed `ctrl_dor_query.py` for redundant `str()`/`dtype=str` — all are justified
- Codebase-wide formatting/style cleanup

## What should we do next? (TODOs)
- [ ] **Finish the "Open Pick List" button** — create a dialog showing the current pick list ← start here!
- [ ] **Refactor controllers to receive `main` instead of `view`** — be careful about view parent widget references
- [ ] **Create a shared table model for `pick list`** accessible from multiple tabs
- [ ] Complete layouts of other tabs: `tab_im_preproc`, `tab_spike_align`
- [ ] Wire up `ViewCheckList` to display the shared pick list model

## Messages from you
- Finish the button open pick list - a dialog showing pick list.
- Considering passing main instances to Ctrls, but also need to be careful about the current parent of view instances.
- Complete the table model which should be accessible to several tabs.
