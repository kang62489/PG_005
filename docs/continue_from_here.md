# Log of the project progress 2026-03-19 Wed 10:00:00

List of modified files:
- classes/dialog_pick_list.py (new - Pick List dialog with QFileSystemWatcher for auto-update)
- classes/model_from_dataframe.py (new - QAbstractTableModel wrapper for pd.DataFrame)
- controllers/ctrl_dor_query.py (refactored check_pick_list, added _load_pick_list logic inline)
- data/pick_list.json (updated with picked rows)

## What have we done? (Summary of current progress)
- Created `ModelFromDataFrame` — a thin `QAbstractTableModel` wrapper for `pd.DataFrame`
  - Implements `rowCount`, `columnCount`, `data`, `headerData`
  - `headerData` returns `AlignLeft | AlignVCenter` for `TextAlignmentRole` to fix header alignment
- Created `DialogPickList` — a dialog showing the current pick list
  - Uses `QFileSystemWatcher` to auto-reload table when `pick_list.json` changes
  - `resize_to_table_content` sizes dialog to fit columns + accounts for scrollbar width
- Refactored `check_pick_list` in `CtrlDorQuery`:
  - Fixed bug: empty JSON (`[]`) caused `KeyError: 'Filename'`
  - Flattened nested `if` → two flat blocks (load, then merge)
  - Renamed variable to `df_saved` for clarity

## What should we do next? (TODOs)
- [ ] **Complete `tab_check_list`** ← start here!

## Messages from you
- Continue to complete tab_check_list
