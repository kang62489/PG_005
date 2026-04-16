# Log of the project progress 2026-04-16 Wed 17:00:00

Last working file: controllers/ctrl_data_selector.py
Last working line: ~end of file

# List of modified files:
- controllers/__init__.py
- controllers/ctrl_dor_query.py
- controllers/ctrl_data_selector.py (renamed from ctrl_check_list.py)
- views/__init__.py
- views/view_data_selector.py (renamed from view_check_list.py)
- main.py
- docs/knowledgebase/signal_slot_cross_controller.md (new)

## Summary of current progress

### Signal/Slot cross-controller wiring
- `CtrlDorQuery` now inherits `QObject` and declares `dor_changed = Signal(str)`
- Signal is emitted in `load_animals()` whenever DOR selection changes
- `Main` connects `ctrl_dor_query.dor_changed` → `ctrl_data_selector.on_dor_changed`
- `CtrlDataSelector.on_dor_changed` stores `current_dor` and calls `load_rec_summary(dor)`

### Code migration from ctrl_dor_query → ctrl_data_selector
- Moved and uncommented: `load_rec_summary`, `apply_filters`, `reset_all_filters`,
  `toggle_shown_columns`, `check_pick_list`, `pick_selected`, `clear_pick_list`, `open_pick_list`
- Moved signal connections and `COLUMNS_TO_PICK` constant
- `rec_data_db` is now opened in `CtrlDataSelector.__init__`

### Rename refactor
- Tab title: "Check Pick List" → "Data Selector"
- Files: ctrl_check_list.py → ctrl_data_selector.py, view_check_list.py → view_data_selector.py
- Classes: CtrlCheckList → CtrlDataSelector, ViewCheckList → ViewDataSelector
- Widget names: tv_check_list → tv_data_selector, lbl_check_list → lbl_data_selector

### Documentation
- Created `docs/knowledgebase/signal_slot_cross_controller.md` explaining the signal/slot pattern

## Completed TODOs (from last session)
- ✅ Transfer rec_table loading code to ctrl_data_selector.py

## What should we do next? (TODOs)
- [ ] Re-arrange the layout of tab Data Selector (view_data_selector.py)
- [ ] Complete the functions in ctrl_data_selector.py (wire up buttons, test flow end-to-end)
