"""
als_dff0.py  --  ALS baseline estimation + dF/F0 calculation pipeline.
=======================================================================
Reads a checked processing brief, collects all existing *_GAUSS.tif files,
runs ALS baseline estimation on each, and saves *_DFF0.tif.

Usage:
    python als_dff0.py --brief data/proc_brief_20260512_002_checked.txt [--lam 100] [--p 0.05] [--n_iter 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from rich.console import Console

from functions import als_baseline_run, check_cuda
from img_proc import parse_brief, update_brief_gauss_exists

# ── Configuration ─────────────────────────────────────────────────────────────

console = Console()


# ── Brief parsing ─────────────────────────────────────────────────────────────


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


def _parse_brief_for_gauss(brief_path: Path) -> tuple[list[Path], Path]:
    """Return all existing *_GAUSS.tif paths and proc_dir from an updated brief."""
    _, _, proc_dir = parse_brief(brief_path)
    update_brief_gauss_exists(brief_path, proc_dir)

    gauss_paths: list[Path] = []
    in_picked = False

    for line in brief_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Picked:"):
            in_picked = True
        elif in_picked:
            if stripped.startswith("["):
                parts = [p.strip() for p in stripped.strip("[]").split(",")]
                gauss_paths.extend(_gauss_paths_from_bracket(parts, proc_dir))
            elif not stripped.startswith("#"):
                in_picked = False

    return gauss_paths, proc_dir


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run(
    brief_path: Path,
    cuda_available: bool,
    lam: float,
    p: float,
    n_iter: int,
) -> None:
    """Parse brief and compute dF/F0 for each *_GAUSS.tif."""
    gauss_paths, _proc_dir = _parse_brief_for_gauss(brief_path)
    console.log(f"Brief: {brief_path.name}")
    console.log(f"Found {len(gauss_paths)} GAUSS TIFF(s) to process  (cuda={cuda_available})")

    for tiff_path in gauss_paths:
        console.log(f"\n{'=' * 60}")
        console.log(f"[cyan]Processing {tiff_path.name}...")
        stack = tifffile.imread(tiff_path).astype(np.float32)
        console.log(f"  Shape {stack.shape}")

        console.log("  Computing ALS baseline...")
        baseline = als_baseline_run(stack, lam, p, n_iter, cuda_available)

        console.log("  Computing dF/F0...")
        dff0 = ((stack - baseline) / baseline).astype(np.float16)

        out_path = tiff_path.with_stem(tiff_path.stem.replace("_GAUSS", "_DFF0"))
        tifffile.imwrite(out_path, dff0)
        console.log(f"  Saved {out_path.name}")

    console.log(f"\n{'=' * 60}")
    console.log("All done!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS baseline + dF/F0 pipeline")
    parser.add_argument("--brief", required=True, type=Path, help="Path to _checked.txt brief file")
    parser.add_argument("--lam", type=float, default=1e2, help="ALS smoothness (default: 100)")
    parser.add_argument("--p", type=float, default=0.05, help="ALS asymmetry (default: 0.05)")
    parser.add_argument("--n_iter", type=int, default=10, help="ALS iterations (default: 10)")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda()
    console.log(_cuda_msg)
    run(args.brief, _cuda_available, args.lam, args.p, args.n_iter)
