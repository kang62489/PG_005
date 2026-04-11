# Log of the project progress 2026-04-02 Thu 21:00
Last working file: test_batch.py
Last working line: 45

# List of modified files:
- batch_process.py
- classes/results_exporter.py
- test_batch.py
- docs/DEPENDENCY_DIAGRAM.md
- docs/PROJECT_SUMMARY.md

## Summary of current progress

### Branch merge: `new_detrend` → `gui`
- Rebased `new_detrend` onto `gui` to linearize history
- Resolved a conflict in `docs/continue_from_here.md` during rebase
- Completed squash merge on `gui` branch (all `new_detrend` commits collapsed into one)

### Changes brought in by `new_detrend` merge:
- **pandas → polars**: Replaced across `functions/kmeans.py`, `classes/abf_clip.py`, `classes/dialog_pick_list.py`, `classes/model_from_dataframe.py`, `classes/plot_results.py`
- **`pyproject.toml`**: `pandas` removed, `polars>=1.39.3` added
- **New experimental detrend scripts** (standalone, not in pipeline):
  - `run_als_1d.py` — ALS baseline correction for 1D Excel traces
  - `run_biexp_detrend.py` — per-pixel bi-exponential detrend for TIFF stacks
  - `demo_als_biexp.py` / `demo_biexp_detrend.py` — visual demos

### Reorganized results output structure
- `classes/results_exporter.py`: `export_all()` now saves to `results/{exp_date}/{zscores,categorized,regions,spatials}/` instead of flat `results/files/`; returns a `dict` of paths keyed by subfolder name
- `batch_process.py`: figure exports updated to use `exp_dir["spatials"]` and `exp_dir["regions"]`
- `test_batch.py`: updated printout; fixed C401 ruff warnings (set comprehensions)

### Docs updated
- `docs/DEPENDENCY_DIAGRAM.md` — pandas→polars, added experimental detrend scripts section
- `docs/PROJECT_SUMMARY.md` — same updates, timestamp updated to 2026-04-02

## Completed TODOs/Tasks (before new wrap-up)
- ✅ Squash merge `new_detrend` into `gui`
- ✅ Update `docs/PROJECT_SUMMARY.md` and `docs/DEPENDENCY_DIAGRAM.md`
- ✅ Reorganize results output into date-based subfolders

## What should we do next? (TODOs)
- Try to complete the GUI (Phase 1–5 per plan `.claude/plans/swift-plotting-porcupine.md`) OR organize `rec_data.db`

## Messages from you
- Re-sectioned and mounted slides since 15:00 today — go rest! 🙌
