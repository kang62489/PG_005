---
keywords: cuda, numba-cuda, nvrtc, nvjitlink, compute capability, volta, v100, saion, uv, CCSupportError, ERROR_UNSUPPORTED_ARCH
files_changed: pyproject.toml
severity: major
---

# 2026-09-22

## Problem Description

Running `img_proc.py` on the saion cluster's Tesla V100 nodes (compute capability 7.0)
failed inside the GPU detrend kernel, on a machine that had previously worked per
`docs/resolved_problems/cuda_env_setup_for_numba_on_linux_hpc.md`. Two errors appeared
in sequence, one after fixing the other — both from the same underlying cause.

### Symptoms

- Job ran fine through image loading and tau estimation, then crashed at the
  `biexp_detrend` GPU kernel launch (`functions/detrend.py:121`).
- First error: `numba.cuda.cudadrv.error.CCSupportError: GPU compute capability 7.0
  is not supported(requires >=7.5)` — raised during nvrtc compilation
  (`numba_cuda/numba/cuda/cudadrv/nvrtc.py: find_closest_arch`).
- After pinning a fix for the first error, a second error appeared at the link stage:
  `cuda.bindings.nvjitlink.nvJitLinkError: ERROR_UNSUPPORTED_ARCH (17)` — raised in
  `numba_cuda/numba/cuda/cudadrv/driver.py: Linker.complete()`.
- Confirmed via `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` that
  the target node (`saion-gpu21`) is a Tesla V100, compute capability 7.0.

### Example Error

```
numba.cuda.cudadrv.error.CCSupportError: GPU compute capability 7.0 is not supported(requires >=7.5)
```

```
cuda.bindings.nvjitlink.nvJitLinkError: ERROR_UNSUPPORTED_ARCH (17)
```

## Root Cause

`pyproject.toml` pins `numba-cuda>=0.23.0` with no upper bound, and does not pin any
of its transitive CUDA component dependencies (`cuda-bindings`, `cuda-core`,
`cuda-nvrtc`, `nvidia-nvjitlink-cu12`, etc). `uv` resolved these to their latest
available versions, which turned out to be **CUDA 13.0** builds.

CUDA 13.0 removed offline-compilation support for GPU architectures below compute
capability 7.5 — this drops Maxwell, Pascal, and **Volta** GPUs, confirmed via
NVIDIA's blog post ["Navigating GPU Architecture Support: A Guide for NVIDIA CUDA
Developers"](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/).

At the time of this issue, the saion cluster's idle GPU nodes were:

- `gpu-v100` partition (`saion-gpu15-16, 18-22`) — Tesla V100, compute capability 7.0
- `gpu-p100` partition — Tesla P100, compute capability 6.0
- `gpu-a100` / `short-a100` partition (`saion-gpu23-26`) — A100, compute capability 8.0,
  but all four nodes were in `drng` (draining) state and unavailable

So both the compile stage (nvrtc) and the link stage (nvjitlink) independently
enforced the CC ≥ 7.5 cutoff, each needing its own CUDA 12.x pin to fix, since they
ship as separate PyPI packages.

## Solution

### Files Changed

- `pyproject.toml` — added two explicit dependency pins.

### Commands Run

```bash
uv add "nvidia-cuda-nvrtc-cu12"
uv sync

uv add "nvidia-nvjitlink-cu12"
uv sync
```

Note: the first attempt used the package name `cuda-nvrtc-cu12` (without the
`nvidia-` prefix), which failed with "not found in the package registry". The
correct PyPI package name is `nvidia-cuda-nvrtc-cu12`.

### Why This Fixes It

Pinning `nvidia-cuda-nvrtc-cu12` forces `uv` to resolve a CUDA 12.x nvrtc build
instead of drifting to the CUDA 13.x default, restoring compile-stage support for
compute capability 7.0. Pinning `nvidia-nvjitlink-cu12` does the same for the
separate link-stage component. Both were needed because each ships as an
independent PyPI package with its own version resolution.

### Useful Diagnostic Commands

```bash
# List partitions and node states
sinfo

# Check GPU model + compute capability on a specific node (short reservation, no long lock)
srun --partition=gpu --gres=gpu:1 --nodelist=<node> --time=00:01:00 \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Check an installed package's version
uv pip show <package>

# List installed CUDA-related packages
uv pip list | grep -i cuda
```

## Flag for Future

`numba-cuda>=0.23.0` in `pyproject.toml` still has no upper bound on its transitive
CUDA dependencies beyond the two now-pinned packages. A future `uv sync` could still
re-resolve other CUDA components (e.g. `cuda-bindings`, `cuda-core` themselves) to
13.x and reintroduce a similar failure. Consider pinning the full CUDA 12.x
component set explicitly if this recurs.

## Related

See `docs/resolved_problems/cuda_env_setup_for_numba_on_linux_hpc.md` (2026-05-03)
for the original CUDA/numba/driver setup on this cluster, which established
`saion-gpu18` (V100, driver 570.195.03) as a working node at that time. This issue
is a regression that occurred after that setup, caused by unpinned transitive CUDA
package versions drifting to CUDA 13.0 over time.
