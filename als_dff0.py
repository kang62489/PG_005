"""
als_dff0.py  --  ALS baseline estimation + dF/F0 calculation pipeline.
=======================================================================
Reads a processing list, collects all existing *_GAUSS.tif files,
runs ALS baseline estimation on each, and saves *_DFF0.tif.

Usage:
    python als_dff0.py --proc_list data/proc_pick_20260512_002.txt [--lam 100] [--p 0.05] [--n_iter 10]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile
from rich.console import Console

from functions import als_baseline_run, check_cuda, get_memory_usage
from img_proc import parse_proc_list, update_proc_list_gauss_exists

# ── Configuration ─────────────────────────────────────────────────────────────

EPSILON = np.float32(1e-6)
console = Console()


# ── Proc list parsing ─────────────────────────────────────────────────────────


def _gauss_paths_from_bracket(parts: list[str], proc_dir: Path) -> list[Path]:
    """Return existing GAUSS TIFF paths for one parsed bracket-line parts list."""
    if len(parts) < 2:
        return []
    stem = Path(parts[0]).stem
    gauss_exists = parts[1]
    paths: list[Path] = []
    if gauss_exists in ("BIEXP", "BIEXP & MOV"):
        candidate = proc_dir / f"{stem}_BIEXP_GAUSS.tif"
        if candidate.exists():
            paths.append(candidate)
    if gauss_exists in ("MOV", "BIEXP & MOV"):
        candidate = proc_dir / f"{stem}_MOV_GAUSS.tif"
        if candidate.exists():
            paths.append(candidate)
    return paths


def _parse_proc_list_for_gauss(proc_list_path: Path) -> tuple[list[Path], Path]:
    """Return all existing *_GAUSS.tif paths and proc_dir from an updated proc list."""
    _, _, proc_dir = parse_proc_list(proc_list_path)
    update_proc_list_gauss_exists(proc_list_path, proc_dir)

    gauss_paths: list[Path] = []
    seen_paths: set[Path] = set()
    in_picked = False

    for line in proc_list_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Picked:"):
            in_picked = True
        elif in_picked:
            if stripped.startswith("["):
                parts = [p.strip() for p in stripped.strip("[]").split(",")]
                for path in _gauss_paths_from_bracket(parts, proc_dir):
                    resolved_path = path.resolve()
                    if resolved_path not in seen_paths:
                        seen_paths.add(resolved_path)
                        gauss_paths.append(path)
            elif not stripped.startswith("#"):
                in_picked = False

    return gauss_paths, proc_dir


# ── Processing functions ───────────────────────────────────────────────────────


def _dff0_output_path(tiff_path: Path) -> Path:
    """Return output path by replacing the GAUSS suffix with DFF0."""
    return tiff_path.with_stem(tiff_path.stem.replace("_GAUSS", "_DFF0"))


def process_dff0(tiff_path: Path, cuda_available: bool, lam: float, p: float, n_iter: int, emitter=None) -> None:
    """ALS baseline + dF/F0 for one *_GAUSS.tif. Saves *_DFF0.tif."""
    t0 = time.time()

    if emitter:
        emitter({"type": "step", "msg": f"Loading {tiff_path.name}..."})
    console.log(f"[cyan]Loading {tiff_path.name}...")
    stack = tifffile.imread(tiff_path)
    console.log(f"  Shape {stack.shape}  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    if emitter:
        emitter({"type": "step", "msg": f"Computing ALS baseline (lam={lam:g}, p={p:g}, n_iter={n_iter})..."})
    console.log(f"  Computing ALS baseline (lam={lam:g}, p={p:g}, n_iter={n_iter})...")
    baseline = als_baseline_run(stack, lam, p, n_iter, cuda_available)
    console.log(f"  Baseline done  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    if emitter:
        emitter({"type": "step", "msg": "Computing dF/F0..."})
    console.log("  Computing dF/F0...")
    safe_baseline = np.where(np.abs(baseline) > EPSILON, baseline, EPSILON)
    dff0 = ((stack - baseline) / safe_baseline).astype(np.float16)

    out_path = _dff0_output_path(tiff_path)
    tifffile.imwrite(out_path, dff0)
    if emitter:
        emitter({"type": "step", "msg": f"✓ Saved {out_path.name}  ({time.time() - t0:.1f}s)"})
    console.log(f"  Saved {out_path.name}  ({time.time() - t0:.1f}s)")

    del stack, baseline, safe_baseline, dff0


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run(
    proc_list_path: Path,
    cuda_available: bool,
    lam: float,
    p: float,
    n_iter: int,
    emitter=None,
) -> None:
    """Parse processing list and compute dF/F0 for each *_GAUSS.tif."""
    gauss_paths, _proc_dir = _parse_proc_list_for_gauss(proc_list_path)
    console.log(f"Processing list: {proc_list_path.name}")
    console.log(f"  {len(gauss_paths)} GAUSS TIFF(s) to process  (cuda={cuda_available})")
    console.log(f"  ALS: lam={lam:g}, p={p:g}, n_iter={n_iter}")

    total = len(gauss_paths)
    for i, tiff_path in enumerate(gauss_paths, 1):
        if not tiff_path.exists():
            console.log(f"[DROPPED] {tiff_path.name} not found")
            continue

        if emitter:
            emitter({"type": "progress", "i": i, "total": total, "file": tiff_path.name})
        console.log(f"\n{'=' * 60}")
        console.log(f"{tiff_path.name} [{i}/{total}]")
        process_dff0(tiff_path, cuda_available, lam, p, n_iter, emitter=emitter)

    console.log(f"\n{'=' * 60}")
    console.log("All done!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS baseline + dF/F0 pipeline")
    parser.add_argument("--proc_list", required=True, type=Path, help="Path to processing list file (proc_*.txt)")
    parser.add_argument("--lam", type=float, default=1e2, help="ALS smoothness (default: 100)")
    parser.add_argument("--p", type=float, default=0.05, help="ALS asymmetry (default: 0.05)")
    parser.add_argument("--n_iter", type=int, default=10, help="ALS iterations (default: 10)")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda() if check_cuda is not None else (False, "CUDA not available")
    console.log(_cuda_msg)
    run(args.proc_list, _cuda_available, args.lam, args.p, args.n_iter)
