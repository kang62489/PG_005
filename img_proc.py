"""
img_proc.py  --  Unified image preprocessing pipeline entry point.
==================================================================
Reads a checked processing brief, routes each file by mode:
  MOV   -> moving-average detrend + Gaussian blur
  BIEXP -> bi-exponential detrend + Gaussian blur
  BOTH  -> BIEXP then MOV
  NONE  -> skip

Checked brief format (5 fields per entry):
  [filename, gauss_exists, do_processing, detrend_mode, paired_abf]

Usage:
    python img_proc.py --brief data/proc_brief_20260512_002_checked.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile
from rich.console import Console

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


# ── Brief parsing ─────────────────────────────────────────────────────────────


def _parse_bracket_entry(brief_line: str) -> dict | None:
    """Parse one '[filename, gauss_exists, do_processing, detrend_mode, ...]' bracket line.

    Checked brief format (5 fields):
        [filename, gauss_exists, do_processing, detrend_mode, paired_abf]

    Returns a dict {"file", "proc", "mode"} or None if fields are missing,
    do_processing is SKIP, or detrend_mode is NONE.
    """
    parts = [p.strip() for p in brief_line.strip("[]").split(",")]
    if len(parts) < 4:
        return None
    filename, proc, mode = parts[0], parts[2], parts[3]
    if proc == "SKIP" or mode == "NONE":
        return None
    return {"file": filename, "proc": proc, "mode": mode}


def parse_brief(brief_path: Path) -> tuple[list[dict], Path, Path]:
    """
    Parse a checked processing brief (_checked.txt).

    Extracts the "Picked:" block as a list of entries and reads
    dir_raw_tiffs / dir_proc_tiffs from the footer.

    Returns:
        (entries, raw_dir, proc_dir) where each entry is
        {"file": str, "proc": str, "mode": str}.
        Entries with MODE == "NONE" are excluded.
    """
    text = brief_path.read_text()
    entries: list[dict] = []
    raw_dir: Path | None = None
    proc_dir: Path | None = None
    in_picked = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("Picked:"):
            in_picked = True
            continue

        if in_picked:
            if stripped.startswith("["):
                entry = _parse_bracket_entry(stripped)
                if entry:
                    entries.append(entry)
            elif not stripped.startswith("#"):
                in_picked = False

        if line.startswith("dir_raw_tiffs:"):
            raw_dir = Path(line.split(":", 1)[1].strip())
        elif line.startswith("dir_proc_tiffs:"):
            proc_dir = Path(line.split(":", 1)[1].strip())

    if raw_dir is None or proc_dir is None:
        msg = f"Missing dir_raw_tiffs or dir_proc_tiffs in {brief_path}"
        raise ValueError(msg)

    return entries, raw_dir, proc_dir


# ── Processing functions ───────────────────────────────────────────────────────


def process_mov(file: str, raw_dir: Path, proc_dir: Path, cuda_available: bool) -> None:
    """Moving-average detrend + Gaussian blur. Saves *_MOV_GAUSS.tif."""
    stem = Path(file).stem
    t0 = time.time()

    console.log(f"[cyan]Loading {file}...")
    img = tifffile.imread(raw_dir / file).astype(np.float16)
    console.log(f"  Shape {img.shape}  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    console.log("  Detrending (MOV)...")
    detrended = mov_detrend(img, cuda_available)
    # tifffile.imwrite(proc_dir / f"{stem}_MOV_CAL.tif", detrended.astype(np.float16))
    # console.log(f"  Saved {stem}_MOV_CAL.tif  ({time.time() - t0:.1f}s)")

    console.log("  Gaussian blur...")
    blurred = gaussian_blur_run(detrended, SIGMA, cuda_available)
    tifffile.imwrite(proc_dir / f"{stem}_MOV_GAUSS.tif", blurred.astype(np.float16))
    console.log(f"  Saved {stem}_MOV_GAUSS.tif  ({time.time() - t0:.1f}s)")

    del img, detrended, blurred


def process_biexp(file: str, raw_dir: Path, proc_dir: Path, cuda_available: bool) -> None:
    """Bi-exp detrend + Gaussian blur. Saves *_BIEXP_CAL.tif and *_BIEXP_GAUSS.tif."""
    stem = Path(file).stem
    t0 = time.time()

    console.log(f"[cyan]Loading {file}...")
    img = tifffile.imread(raw_dir / file).astype(np.float16)
    console.log(f"  Shape {img.shape}  memory={get_memory_usage():.2f} GB  ({time.time() - t0:.1f}s)")

    console.log("  Sampling pixels for tau estimation...")
    tau1, tau2 = sample_tau(img)
    console.log(f"  tau1={tau1:.1f}  tau2={tau2:.1f}  ({time.time() - t0:.1f}s)")

    console.log("  Detrending (BIEXP)...")
    detrended = biexp_detrend(img, tau1, tau2, cuda_available)
    # tifffile.imwrite(proc_dir / f"{stem}_BIEXP_CAL.tif", detrended.astype(np.float16))
    # console.log(f"  Saved {stem}_BIEXP_CAL.tif  ({time.time() - t0:.1f}s)")

    console.log("  Gaussian blur...")
    blurred = gaussian_blur_run(detrended, SIGMA, cuda_available)
    del detrended
    tifffile.imwrite(proc_dir / f"{stem}_BIEXP_GAUSS.tif", blurred.astype(np.float16))
    console.log(f"  Saved {stem}_BIEXP_GAUSS.tif  ({time.time() - t0:.1f}s)")

    del img, blurred


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run(brief_path: Path, cuda_available: bool, log_path: Path | None = None) -> None:
    """Parse brief and process each file according to its MODE."""
    global console
    _log_file = None
    if log_path:
        _log_file = log_path.open("a", encoding="utf-8", buffering=1)
        _original_console = console
        console = Console(file=_log_file, highlight=False, no_color=True)
    try:
        entries, raw_dir, proc_dir = parse_brief(brief_path)
        proc_dir.mkdir(parents=True, exist_ok=True)

        console.log(f"Brief: {brief_path.name}")
        console.log(f"  raw  -> {raw_dir}")
        console.log(f"  proc -> {proc_dir}")
        console.log(f"  {len(entries)} file(s) to process  (cuda={cuda_available})")

        for entry in entries:
            file = entry["file"]
            mode = entry["mode"]
            fpath = raw_dir / file

            if not fpath.exists():
                console.log(f"[SKIP] {file} not found")
                continue

            console.log(f"\n{'=' * 60}")
            console.log(f"{file}  MODE={mode}")

            if mode == "MOV":
                process_mov(file, raw_dir, proc_dir, cuda_available)
            elif mode == "BIEXP":
                process_biexp(file, raw_dir, proc_dir, cuda_available)
            elif mode == "BOTH":
                process_biexp(file, raw_dir, proc_dir, cuda_available)
                process_mov(file, raw_dir, proc_dir, cuda_available)
            else:
                console.log(f"  Unknown mode '{mode}', skipping")

        console.log(f"\n{'=' * 60}")
        console.log("All done!")
    finally:
        if _log_file is not None:
            console = _original_console
            _log_file.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image preprocessing pipeline")
    parser.add_argument("--brief", required=True, type=Path, help="Path to _checked.txt brief file")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda() if check_cuda is not None else (False, "CUDA not available")
    console.log(_cuda_msg)
    run(args.brief, _cuda_available)
