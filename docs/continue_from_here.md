# Log of the project progress 2026-05-13 Tue (Session 3)

Last working file: img_proc.py
Last working line: ~176

# List of modified files:
- functions/detrend.py (NEW)
- functions/gaussian_blur.py (NEW)
- functions/tau_estimate.py (NEW)
- functions/__init__.py (updated exports)
- functions/gpu_gauss.py (fixed: unified _gpu_conv with axis parameter)
- img_proc.py (completed from broken draft)
- archive/ (NEW folder)

## Summary of current progress
- Completed full refactor: 6 fragmented files → 3 unified modules
  - `detrend.py`: `_cpu_mov`, `_gpu_mov`, `mov_detrend`, `_cpu_biexp`, `_gpu_biexp`, `biexp_detrend`, `align_to_min`
  - `gaussian_blur.py`: `_cpu_kernel`, `_cpu_conv`, `_cpu_gaussian_blur`, `_gpu_kernel`, `_gpu_conv`, `_gpu_gaussian_blur`, `gaussian_blur_run`
  - `tau_estimate.py`: `biexp`, `make_p0_bounds`, `_fit_pixel`, `sample_tau`
- Rewrote biexp detrend with Numba JIT (`_cpu_biexp`) + CUDA kernel (`_gpu_biexp`)
- Fixed GPU Gaussian blur: replaced separate `convolve_horizontal` / `convolve_vertical` with unified `_gpu_conv(axis)` — no warp divergence, consistent with CPU design
- Completed `img_proc.py` — parses `_checked.txt` brief, routes by MODE (MOV/BIEXP/BOTH/NONE), runs successfully on 3 real TIFF files (1200×1024×1024) on RTX 3070
- Archived old scripts: `im_preprocess.py`, `run_biexp_detrend.py`, `cpu_binning.py`, `kmeans.py` → `archive/`
- Deleted: `cpu_detrend.py`, `gpu_detrend.py`, `cpu_gauss.py`, `gpu_gauss.py`, `cpu_process.py`, `gpu_process.py`

## Completed TODOs (from last session)
- [x] Write `img_proc.py` complete structure (parse_brief, process_mov, process_biexp, run, __main__)
- [x] Create biexp Numba JIT detrend + scipy tau estimation (now in functions/detrend.py + tau_estimate.py)
- [x] Decouple Gaussian blur from detrending — separate functions
- [x] Archive original MOV/BIEXP pipeline scripts

## What should we do next? (TODOs)
- [ ] **[NEXT]** Adjust BIEXP detrend calculation — something in the math/output needs fixing
- [ ] **[NEXT]** Answer user questions about writing CUDA kernels by hand
- [ ] Wire `btn_start_processing` in `ctrl_img_proc.py` → call `run(brief_path, cuda_available)`
- [ ] Implement ALS baseline estimation for dF/F0 calculation
- [ ] Run full verification: MOV mode and BOTH mode (only BIEXP tested so far)
