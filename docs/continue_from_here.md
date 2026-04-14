# Log of the project progress 2026-04-14 Mon

Last working file: controllers/ctrl_dor_query.py
Last working line: ~120

# List of modified files:
- controllers/ctrl_dor_query.py (<- Break here, line ~120)
- views/view_dor_query.py
- utils/__init__.py
- utils/params.py
- logs/Data_2022_07_20.md

## Summary of current progress

### ctrl_dor_query.py — load_data_md implemented
- Connected `load_data_md` to `lw_dor.currentTextChanged` signal
- `load_data_md` reads `DATA_{dor}.md` from LOG_DIR
- Extracts `System` and `Keywords` using `next()` + `split(":", 1)` (safe for time-containing strings)
- Gets file last modified time via `stat().st_mtime`, formatted with weekday (`%Y-%b-%d %a (%H:%M:%S)`)
- Handles missing log file: prints red message via rich console, sets UI fields to "No log found"

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Connect `load_data_md` to `lw_dor` signal
- ✅ Parse System and Keywords from markdown
- ✅ Display last modified time (with weekday) in `le_last_modified`
- ✅ Handle missing log file gracefully

## What should we do next? (TODOs)
- Wire up `cb_log_date` — populate with date block headers from the markdown log
- Display the corresponding log block in `te_log_contents` when a date is selected in `cb_log_date`
- Implement insert log functionality (`te_insert_log` + Insert/Clear buttons)
