"""
als_correct.py  --  ALS slow-fluctuation removal pipeline.
=========================================================
Reads a processing list, collects all existing *_GAUSS.tif files,
subtracts the ALS-estimated slow fluctuation from each, and saves *_ALS.tif.

Usage:
    python als_correct.py --proc_list data/proc_pick_20260512_002.txt [--lam 100] [--p 0.05] [--n_iter 10]
"""

# Standard library imports
import argparse
import time
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile
from rich.console import Console

# Local application imports
from functions import als_run, check_cuda, get_memory_usage
from img_proc import update_proc_list_gauss_exists

# ── Configuration ─────────────────────────────────────────────────────────────

console = Console()


# ── Proc list parsing ─────────────────────────────────────────────────────────


def _parse_bracket(stripped: str, proc_dir: Path) -> list[Path]:
    """Return existing GAUSS TIFF paths for one bracket line."""
    parts = [p.strip() for p in stripped.strip("[]").split(",")]
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


def parse_proc_list(proc_list_path: Path) -> tuple[list[Path], Path]:
    """Return all existing *_GAUSS.tif paths and proc_dir from an updated proc list."""
    proc_dir: Path | None = None
    for line in proc_list_path.read_text().splitlines():
        if line.startswith("dir_proc_tiffs:"):
            proc_dir = Path(line.split(":", 1)[1].strip())
            break
    if proc_dir is None:
        msg = f"Missing dir_proc_tiffs in {proc_list_path}"
        raise ValueError(msg)
    update_proc_list_gauss_exists(proc_list_path, proc_dir)

    text = proc_list_path.read_text()
    gauss_paths: list[Path] = []
    seen_paths: set[Path] = set()
    in_picked = False

    for line in text.splitlines():
        if line.strip().startswith("Picked:"):
            in_picked = True
            continue
        if in_picked:
            if line.strip().startswith("["):
                for path in _parse_bracket(line.strip(), proc_dir):
                    resolved_path = path.resolve()
                    if resolved_path not in seen_paths:
                        seen_paths.add(resolved_path)
                        gauss_paths.append(path)
            else:
                in_picked = False

    return gauss_paths, proc_dir


# ── Processing functions ───────────────────────────────────────────────────────


def _als_output_path(tiff_path: Path) -> Path:
    """Return output path by replacing the GAUSS suffix with ALS."""
    return tiff_path.with_stem(tiff_path.stem.replace("_GAUSS", "_ALS"))


def process_als(tiff_path: Path, cuda_available: bool, lam: float, p: float, n_iter: int, emitter=None) -> None:
    """ALS slow-fluctuation subtraction for one *_GAUSS.tif. Saves *_ALS.tif."""
    t0 = time.time()

    if emitter:
        emitter({"type": "step", "msg": f"Loading {tiff_path.name}..."})
    console.log(f"[cyan]Loading {tiff_path.name}...")
    stack = tifffile.imread(tiff_path)
    console.log(f"  Shape {stack.shape}  dtype={stack.dtype}  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    if emitter:
        emitter({"type": "step", "msg": f"Computing ALS baseline (lam={lam:g}, p={p:g}, n_iter={n_iter})..."})
    console.log(f"  Computing ALS baseline (lam={lam:g}, p={p:g}, n_iter={n_iter})...")
    slow_fluc = als_run(stack, lam, p, n_iter, cuda_available)
    console.log(f"  Slow fluctuation estimated  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    if emitter:
        emitter({"type": "step", "msg": "Subtracting slow fluctuation..."})
    console.log("  Subtracting slow fluctuation...")
    corrected = (stack - slow_fluc).astype(np.float16)

    out_path = _als_output_path(tiff_path)
    tifffile.imwrite(out_path, corrected)
    if emitter:
        emitter({"type": "step", "msg": f"✓ Saved {out_path.name}  ({time.time() - t0:.1f}s)"})
    console.log(f"  Saved {out_path.name}  ({time.time() - t0:.1f}s)")

    del stack, slow_fluc, corrected


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run(
    proc_list_path: Path,
    cuda_available: bool,
    lam: float,
    p: float,
    n_iter: int,
    emitter=None,
) -> None:
    """Parse processing list and subtract slow fluctuation for each *_GAUSS.tif."""
    gauss_paths, _proc_dir = parse_proc_list(proc_list_path)
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
        process_als(tiff_path, cuda_available, lam, p, n_iter, emitter=emitter)

    console.log(f"\n{'=' * 60}")
    console.log("All done!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS slow-fluctuation removal pipeline")
    parser.add_argument("--proc_list", required=True, type=Path, help="Path to processing list file (proc_*.txt)")
    parser.add_argument("--lam", type=float, default=1e2, help="ALS smoothness (default: 100)")
    parser.add_argument("--p", type=float, default=0.05, help="ALS asymmetry (default: 0.05)")
    parser.add_argument("--n_iter", type=int, default=10, help="ALS iterations (default: 10)")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda() if check_cuda is not None else (False, "CUDA not available")
    console.log(_cuda_msg)
    run(args.proc_list, _cuda_available, args.lam, args.p, args.n_iter)
