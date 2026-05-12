# Log of the project progress 2026-05-11 Sun (Session 6)

Last working file: controllers/ctrl_img_proc.py
Last working line: ~143

# List of modified files:
- None (review/orientation session only)

## Summary of current progress
- Fixed VSCode IDE issues (external, not code-related)
- Re-oriented in the GUI codebase on the `gui` branch
- Reviewed ctrl_img_proc.py and view_img_proc.py — GUI tab for image processing
- Clarified that pick_list.json and pick_list.xlsx both live in data/
- Clarified that data/pick_list.xlsx is deleted and data/pick_list.json is reset to [] on every app startup (clear_pick_list called in CtrlDataSelector.__init__)

## Completed TODOs (from last session)
- None (this session was review/orientation only)

## What should we do next? (TODOs)
- [ ] Wire btn_start_processing — run biexp processing based on PROC column values
- [ ] Plan merging biexp detrend into CPU/GPU pipeline (run_biexp_detrend.py -> functions/)
- [ ] Implement ALS baseline estimation for dF/F0 calculation
- [ ] Consider cropping image edges to reduce non-uniform illumination effect on tau estimation
- [ ] Decouple Gaussian blur from detrending — separate functions for detrend and blur
- [ ] Archive original Mov pipeline (im_preprocess.py + cpu/gpu_process.py) — keep usable but clearly separated for PI reference
- [ ] Apply im_preprocess_to_be_mod.py changes to main im_preprocess.py
