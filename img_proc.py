"""
img_proc.py  --  Unified image preprocessing pipeline entry point.
==================================================================
Reads a proc list, routes each file by mode:
  MOV   -> moving-average detrend + Gaussian blur
  BIEXP -> bi-exponential detrend + Gaussian blur
  BOTH  -> BIEXP then MOV
  NONE  -> skip

Proc list format (5 fields per entry):
  [filename, gauss_exists, do_processing, detrend_mode, paired_abf]

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
    mov_detrend,
    sample_tau,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Gaussian blur
SIGMA = 6.0

# Setup rich console for logging
console = Console()


# ── Proc list parsing ─────────────────────────────────────────────────────────


def _parse_bracket(proc_list_line: str) -> dict | None:
    """Parse one '[filename, gauss_exists, do_processing, detrend_mode, ...]' bracket line.

    Proc list format (5 fields):
        [filename, gauss_exists, do_processing, detrend_mode, paired_abf]

    Returns a dict {"file", "proc", "mode"} or None if fields are missing,
    do_processing is SKIP, or detrend_mode is NONE.
    """
    parts = [p.strip() for p in proc_list_line.strip("[]").split(",")]
    if len(parts) < 5:
        return None
    filename, proc, mode = parts[0], parts[3], parts[4]
    if proc == "SKIP" or mode == "NONE":
        return None
    return {"file": filename, "proc": proc, "mode": mode}


def parse_proc_list(proc_list_path: Path) -> tuple[list[dict], Path, Path]:
    """
    Parse a proc list file (proc_*.txt).

    Extracts the "Picked:" block as a list of entries and reads
    dir_raw_tiffs / dir_proc_tiffs from the footer.

    Returns:
        (entries, raw_dir, proc_dir) where each entry is
        {"file": str, "proc": str, "mode": str}.
        Entries with MODE == "NONE" are excluded.
    """
    text = proc_list_path.read_text()
    entries: list[dict] = []
    raw_dir: Path | None = None
    proc_dir: Path | None = None
    in_picked = False

    for line in text.splitlines():
        if line.strip().startswith("Picked:"):
            in_picked = True
            continue

        if in_picked:
            if line.strip().startswith("["):
                entry = _parse_bracket(line.strip())
                if entry:
                    entries.append(entry)
            else:
                in_picked = False

        if line.startswith("dir_raw_tiffs:"):
            raw_dir = Path(line.split(":", 1)[1].strip())
        elif line.startswith("dir_proc_tiffs:"):
            proc_dir = Path(line.split(":", 1)[1].strip())

    if raw_dir is None or proc_dir is None:
        msg = f"Missing dir_raw_tiffs or dir_proc_tiffs in {proc_list_path}"
        raise ValueError(msg)

    return entries, raw_dir, proc_dir


# ── Proc list update ──────────────────────────────────────────────────────────


def update_proc_list_gauss_exists(proc_list_path: Path, proc_dir: Path) -> None:
    """Rewrite gauss_exists, do_processing, and detrend_mode (cols 1, 3, 4) based on actual files in proc_dir."""
    lines = proc_list_path.read_text().splitlines()
    updated = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            parts = [p.strip() for p in stripped.strip("[]").split(",")]
            if len(parts) >= 5:
                stem = Path(parts[0]).stem
                has_biexp = (proc_dir / f"{stem}_BIEXP_GAUSS.tif").exists()
                has_mov = (proc_dir / f"{stem}_MOV_GAUSS.tif").exists()
                gauss_exists = (
                    "BIEXP & MOV" if has_biexp and has_mov
                    else "BIEXP" if has_biexp
                    else "MOV" if has_mov
                    else "No"
                )
                parts[1] = gauss_exists
                parts[3] = "YES" if gauss_exists == "No" else "SKIP"
                parts[4] = "BIEXP" if parts[3] == "YES" else "NONE"
                line = "[" + ", ".join(parts) + "]"
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
