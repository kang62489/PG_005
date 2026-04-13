# Log of the project progress 2026-04-13 Sun 12:00
Last working file: controllers/ctrl_dor_query.py
Last working line: 48

# List of modified files:
- controllers/ctrl_dor_query.py
- views/view_dor_query.py

## Summary of current progress

### Bug fixes in ctrl_dor_query.py
- Fixed infinite recursion: `update_preview` was calling `te_editing.setMarkdown()` on itself → now renders to `te_preview`
- Fixed Qt crash on `---` input: Qt tries to parse YAML frontmatter and crashes → prepend `\n` to text starting with `---` as workaround

### Layout redesign in view_dor_query.py
- Removed old `te_editing` / `te_preview` markdown editor panels
- New layout under `lo_data_folder_props` (QHBoxLayout):
  - **Left**: QFormLayout (Last Updated, System, Keywords) + `te_insert_log` + Insert/Clear buttons
  - **Right**: `cb_log_date` (QComboBox) + `te_log_contents` (read-only QTextEdit)
- Reviewed and confirmed layout structure; ready to wire up controller functions

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Fix `update_preview` infinite recursion bug
- ✅ Fix Qt `---` YAML frontmatter crash
- ✅ Redesign `view_dor_query.py` layout
- ✅ Review new layout

## What should we do next? (TODOs)
- Wire up controller functions for the new layout in `ctrl_dor_query.py`

## Extra Notes (Plan for ctrl_dor_query.py next session)
1. Select date in DOR → check and load `logs/Data_{DOR}.md`
2. Fill UIs (QDateEdit, QTextEdit) based on loaded markdown
3. Determine the inserting text format for `te_insert_log`
4. Set guards for the above functions
5. Complete `cb_log_date` to jump to and display a certain block in the loaded markdown
