"""
img_proc.py  --  Unified image preprocessing pipeline entry point.
==================================================================
Reads a proc list, routes each file by mode:
  MOV   -> moving-average detrend + Gaussian blur
  BIEXP -> bi-exponential detrend + Gaussian blur
  BOTH  -> BIEXP then MOV
  NONE  -> skip

Proc list format (column names declared on the 'Picked:' line):
  [raw_tiff_name, gauss_exists, als_exists, do_processing, detrend_mode, paired_abf]

Usage:
    python img_proc.py --proc_list data/proc_pick_20260512_002.txt
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
from functions import (
    biexp_detrend,
    check_cuda,
    gaussian_blur_run,
    get_memory_usage,
    list_parser,
    mov_detrend,
    sample_tau,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Gaussian blur
SIGMA = 6.0

# Setup rich console for logging
console = Console()


# ── Proc list parsing ─────────────────────────────────────────────────────────


def parse_proc_list(proc_list_path: Path) -> tuple[list[dict], Path, Path]:
    """
    Parse a proc list file (proc_*.txt).

    Returns:
        (entries, raw_dir, proc_dir) where each entry is
        {"file": str, "proc": str, "mode": str}.
        Entries with do_processing == "SKIP" or detrend_mode == "NONE" are excluded.
    """
    table, io_dirs = list_parser(proc_list_path)

    missing = [k for k in ("dir_raw_tiffs", "dir_proc_tiffs") if k not in io_dirs]
    if missing:
        msg = f"Missing {', '.join(missing)} in {proc_list_path}"
        raise ValueError(msg)
    raw_dir = Path(io_dirs["dir_raw_tiffs"])
    proc_dir = Path(io_dirs["dir_proc_tiffs"])

    entries = [
        {"file": row["raw_tiff_name"], "proc": row["do_processing"], "mode": row["detrend_mode"]}
        for row in table.iter_rows(named=True)
        if row["do_processing"] != "SKIP" and row["detrend_mode"] != "NONE"
    ]

    return entries, raw_dir, proc_dir


# ── Proc list update ──────────────────────────────────────────────────────────


def _refresh_gauss_row(row: dict[str, str], proc_dir: Path) -> dict[str, str]:
    """Recompute gauss_exists, do_processing, and detrend_mode for one row from actual files in proc_dir."""
    stem = Path(row["raw_tiff_name"]).stem
    has_biexp = (proc_dir / f"{stem}_BIEXP_GAUSS.tif").exists()
    has_mov = (proc_dir / f"{stem}_MOV_GAUSS.tif").exists()
    gauss_exists = (
        "BIEXP & MOV" if has_biexp and has_mov
        else "BIEXP" if has_biexp
        else "MOV" if has_mov
        else "No"
    )
    do_processing = "YES" if gauss_exists == "No" else "SKIP"
    detrend_mode = "BIEXP" if do_processing == "YES" else "NONE"
    return {**row, "gauss_exists": gauss_exists, "do_processing": do_processing, "detrend_mode": detrend_mode}


def update_proc_list_gauss_exists(proc_list_path: Path, proc_dir: Path) -> None:
    """Rewrite gauss_exists, do_processing, and detrend_mode based on actual files in proc_dir."""
    table, _io_dirs = list_parser(proc_list_path)
    refreshed_rows = iter(_refresh_gauss_row(row, proc_dir) for row in table.iter_rows(named=True))

    lines = proc_list_path.read_text().splitlines()
    updated = []
    for line in lines:
        if line.strip().startswith("["):
            row = next(refreshed_rows)
            line = "[" + ", ".join(row[col] for col in table.columns) + "]"
        updated.append(line)
    proc_list_path.write_text("\n".join(updated))


# ── Processing functions ───────────────────────────────────────────────────────


def process_mov(file: str, raw_dir: Path, proc_dir: Path, cuda_available: bool, emitter=None) -> None:
    """Moving-average detrend + Gaussian blur. Saves *_MOV_GAUSS.tif."""
    stem = Path(file).stem
    t0 = time.time()

    console.log(f"[cyan]Loading {file}...")
    img = tifffile.imread(raw_dir / file)
    console.log(f"  Shape {img.shape}  dtype={img.dtype}  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    if emitter:
        emitter({"type": "step", "msg": "Detrending (MOV)..."})
    console.log("  Detrending (MOV)...")
    detrended = mov_detrend(img, cuda_available)

    if emitter:
        emitter({"type": "step", "msg": "Gaussian blur..."})
    console.log("  Gaussian blur...")
    blurred = gaussian_blur_run(detrended, SIGMA, cuda_available)
    tifffile.imwrite(proc_dir / f"{stem}_MOV_GAUSS.tif", blurred.astype(np.float16))
    if emitter:
        emitter({"type": "step", "msg": f"✓ Saved {stem}_MOV_GAUSS.tif  ({time.time() - t0:.1f}s)"})
    console.log(f"  Saved {stem}_MOV_GAUSS.tif  ({time.time() - t0:.1f}s)")

    del img, detrended, blurred


def process_biexp(file: str, raw_dir: Path, proc_dir: Path, cuda_available: bool, emitter=None) -> None:
    """Bi-exp detrend + Gaussian blur. Saves *_BIEXP_CAL.tif and *_BIEXP_GAUSS.tif."""
    stem = Path(file).stem
    t0 = time.time()

    console.log(f"[cyan]Loading {file}...")
    img = tifffile.imread(raw_dir / file)
    console.log(f"  Shape {img.shape}  dtype={img.dtype}  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    if emitter:
        emitter({"type": "step", "msg": "Sampling pixels for tau estimation..."})
    console.log("  Sampling pixels for tau estimation...")
    tau1, tau2 = sample_tau(img)
    console.log(f"  tau1={tau1:.1f}  tau2={tau2:.1f}  ({time.time() - t0:.1f}s)")

    if emitter:
        emitter({"type": "step", "msg": "Detrending (BIEXP)..."})
    console.log("  Detrending (BIEXP)...")
    detrended = biexp_detrend(img, tau1, tau2, cuda_available)

    if emitter:
        emitter({"type": "step", "msg": "Gaussian blur..."})
    console.log("  Gaussian blur...")
    blurred = gaussian_blur_run(detrended, SIGMA, cuda_available)
    del detrended
    tifffile.imwrite(proc_dir / f"{stem}_BIEXP_GAUSS.tif", blurred.astype(np.float16))
    if emitter:
        emitter({"type": "step", "msg": f"✓ Saved {stem}_BIEXP_GAUSS.tif  ({time.time() - t0:.1f}s)"})
    console.log(f"  Saved {stem}_BIEXP_GAUSS.tif  ({time.time() - t0:.1f}s)")

    del img, blurred


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run(proc_list_path: Path, cuda_available: bool, emitter=None) -> None:
    """Parse proc list and process each file according to its MODE."""
    entries, raw_dir, proc_dir = parse_proc_list(proc_list_path)
    proc_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Processing list: {proc_list_path.name}")
    console.log(f"  raw  -> {raw_dir}")
    console.log(f"  proc -> {proc_dir}")
    console.log(f"  {len(entries)} file(s) to process  (cuda={cuda_available})")

    total = len(entries)
    for i, entry in enumerate(entries, 1):
        file = entry["file"]
        mode = entry["mode"]
        fpath = raw_dir / file

        if not fpath.exists():
            console.log(f"[DROPPED] {file} not found")
            continue

        if emitter:
            emitter({"type": "progress", "i": i, "total": total, "file": file, "mode": mode})
        console.log(f"\n{'=' * 60}")
        console.log(f"{file}  MODE={mode} [{i}/{total}]")

        if mode == "MOV":
            process_mov(file, raw_dir, proc_dir, cuda_available, emitter=emitter)
        elif mode == "BIEXP":
            process_biexp(file, raw_dir, proc_dir, cuda_available, emitter=emitter)
        elif mode == "BOTH":
            process_biexp(file, raw_dir, proc_dir, cuda_available, emitter=emitter)
            process_mov(file, raw_dir, proc_dir, cuda_available, emitter=emitter)
        else:
            console.log(f"  Unknown mode '{mode}', skipping")

    console.log(f"\n{'=' * 60}")
    update_proc_list_gauss_exists(proc_list_path, proc_dir)
    console.log("All done!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image preprocessing pipeline")
    parser.add_argument("--proc_list", required=True, type=Path, help="Path to proc list file (proc_*.txt)")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda() if check_cuda is not None else (False, "CUDA not available")
    console.log(_cuda_msg)
    run(args.proc_list, _cuda_available)
