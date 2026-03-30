# Log of the project progress 2026-03-31 Tue 02:53:10
Last working file: classes/plot_results.py
Last working line: 571

# List of modified files:
- classes/abf_clip.py
- classes/plot_results.py
- functions/spike_centered_processes.py
- batch_process.py
- im_dynamics.py
- pyproject.toml

## Summary of current progress (based on modified files, existing plans)
- Raised `minimal_required_interval_frames` from 3→4 in `abf_clip.py` so all kept segments are uniform 9-frame shape
- Replaced NaN-padded loop + `nanmedian` with a numba `@njit(parallel=True)` kernel `_median_axis0` in `spike_centered_processes.py` for faster median computation (requires `pip install "numba>=0.60.0"`)
- Added `numba>=0.60.0` to `pyproject.toml` dependencies
- Fixed `PlotSpatialDist` colorbar: tick labels changed to `BK/Dim/Bright` with `rotation=90`
- Fixed Vm trace axes span: `gs[2, :]` → `gs[2, :n_frames]` so right edge aligns with Frame 4
- Added magnification (`self.obj`) to spatial plot suptitle
- Fixed `batch_process.py` and `im_dynamics.py` to pass the correct `objective`/`OBJECTIVE` to `PlotSpatialDist` instead of relying on the `"10X"` default

## Completed TODOs/Tasks (before new wrap-up)
- Uniform segment length fix (A1 + A2 from plan)
- Colorbar UI fixes (B1 + B2 from plan)
- Magnification shown correctly in spatial plot title

## What should we do next? (TODOs)
- (none specified)

## Messages from you
- (none)
