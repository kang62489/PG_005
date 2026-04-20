# Log of the project progress 2026-04-17 Thu

Last working file: controllers/ctrl_dor_query.py
Last working line: ~358 (update_project method)

# List of modified files:
- controllers/ctrl_dor_query.py
- views/view_dor_query.py
- utils/params.py
- logs/Data_2022_07_20.md
- logs/Data_2022_07_27.md

## Summary of current progress

### DOR Query tab improvements

1. **Added "Project" field** — new `le_prj` QLineEdit; reads/writes `Project:` from `Data_{DOR}.md` frontmatter
2. **Renamed** `te_file_structure` → `te_folder_structure` (naming consistency)
3. **Layout fix** — moved Insert Log section from left panel to right panel
4. **Adjusted UI sizes** in `params.py`: `TE_DESCRIPTIONS_HEIGHT=240`, `TE_FINDINGS_HEIGHT=60` (no fixed height), `TE_FOLDER_STRUCTURE_HEIGHT=40`
5. **Updated new-file template** — new `Data_{DOR}.md` now includes `Project:` in frontmatter and `# Extra Info` section

## Completed TODOs (from last session)
- ✅ Add `Data_{dor}.md` files to the data directory and fix their formats (partially — updated existing log files)

## What should we do next? (TODOs)
- [ ] Rename `ABF_NUMBER` → `PAIRED_ABF` in all REC summary Excel files, then update `functions/xlsx_reader.py` and `controllers/ctrl_dor_query.py`
- [ ] Backfill existing `Data_{DOR}.md` log files to include the new `Project:` field in frontmatter
- [ ] Organize `rec_data.db` and raw `.rec` files
