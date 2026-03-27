# Log of the project progress 2026-03-23 Mon

Last working file: run_biexp_detrend.py
Last working line: end of file

## List of modified files (this session):
- `demo_als_biexp.py` (new — ALS + bi-exp concept demo with simulated data)
- `demo_biexp_detrend.py` (new — guard demo on real tif files)
- `run_biexp_detrend.py` (new — full ALS template detrend processing script)
- `im_preprocess.py` (minor edits)

## Summary of current progress (based on modified files, existing plans)
- Explored ALS (Asymmetric Least Squares) for baseline estimation of spiking fluorescence signals
- Designed a detrend pipeline: ALS on mean trace (bright pixels only) → baseline template T → per-pixel linear fit (y ≈ a·T + b) → subtract
- Discovered that 0028/0029 baselines are U-shaped (not bi-exp), so ALS template fitting (shape-agnostic) was adopted instead of bi-exp model
- Processed 3 test files and saved detrended stacks to `processed_images/` as `*_BiexpCal.tif`
- Results still need visual verification in Fiji

## Completed TODOs/Tasks (before new wrap-up)
- [x] Test `load_pick_list()` in the GUI (from previous session)
- [x] ALS concept exploration and pipeline design
- [x] Implement ALS template detrend script (`run_biexp_detrend.py`)
- [x] Process 3 test files: 2026_03_20-0028, 2026_03_20-0029, 2025_06_11-0002

## What should we do next? (TODOs)
- [ ] Visually verify detrended results in Fiji — check if U-shape baseline is removed
- [ ] Investigate why 0028/0029 have a U-shaped baseline (z-drift? laser fluctuation?)
- [ ] Tune ALS parameters (lambda, p) per recording if needed
- [ ] Integrate ALS template detrend into main pipeline (replace cpu_detrend / gpu_detrend)
- [ ] Add guard checks back for batch processing safety (e.g. flat baseline detection)
- [ ] Continue GUI work (CtrlCheckList — "Start Check", browse dirs, remove/clear list)

## Messages from you
- (none)
