# Log of the project progress 2026-05-14 Wed (Session 4)

Last working file: functions/detrend.py
Last working line: ~290

# List of modified files:
- functions/detrend.py (major optimizations)
- img_proc.py (float16 pipeline, align_to_min cleanup)

## Summary of current progress

### float16 pipeline unification
- `img_proc.py`: imread now loads as `float16` instead of `float32` (~2.3 GB saved at load)
- All `tifffile.imwrite` calls save as `float16` (removed `np.clip` + `uint16`)
- `biexp_detrend` and `mov_detrend` both return `float16`

### detrend.py internal dtype: float64 → float32
- `biexp_detrend`: `basis_matrix`, `basis_pinv`, `img_flat` all changed from float64 → float32
- `_gpu_biexp` kernel: `coeffs` local array and all accumulators changed float64 → float32
- Halves memory footprint of biexp computation (~9.4 GB → ~4.7 GB for img_flat)

### GPU biexp coalesced memory access
- `_gpu_biexp` kernel: transposed indexing `img_flat[frame_idx, pixel_idx]` instead of `img_flat[i, t]`
- `biexp_detrend` dispatcher: GPU path uses `(n_frames, n_pixels)` layout (no `.T`); CPU path keeps `(n_pixels, n_frames)` for cache-friendliness

### _gpu_mov sliding window
- Replaced O(window × frames) loop + `cuda.local.array(2048)` with two-pass sliding window: O(frames) per pixel, no local array allocation

### _cpu_mov fixes
- Fixed two nested `prange` → `range` (Numba does not support nested prange)
- Replaced `np.mean(slice)` with sliding window (same logic as GPU)

### align_to_min — real bottleneck found
- Initially changed to `np.median` (slow — sorts 1200 values × 1M pixels on CPU, ~80s!)
- Reverted to `stack.mean(axis=0)` — fast O(n), good enough for stable fluorescence baseline
- Added `floor=3.0` parameter: every pixel's baseline lands at ~3.0 (avoids dF/F0 division-by-zero)
- Changed from relative alignment to per-pixel normalization: `stack - means + floor`
- Moved `align_to_min` call inside `biexp_detrend` (was previously in `img_proc.py`) — now consistent with `mov_detrend`
- Removed `align_to_min` import from `img_proc.py`

## Completed TODOs (from last session)
- [x] Adjust BIEXP detrend calculation — align_to_min now normalizes correctly, values ~3.0
- [x] Answer user questions about writing CUDA kernels by hand

## What should we do next? (TODOs)
- [ ] **[NEXT]** Wire `btn_start_processing` in `ctrl_img_proc.py` → call `run(brief_path, cuda_available)`
- [ ] **[NEXT]** `_gpu_mov` still has non-coalesced memory access — consider transposing to `(n_frames, n_pixels)` same as biexp (separate future task)
- [ ] Implement ALS baseline estimation for dF/F0 calculation
- [ ] Run full verification: MOV mode and BOTH mode
