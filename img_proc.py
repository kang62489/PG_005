"""
img_proc.py  --  Unified image preprocessing pipeline entry point.
==================================================================
Reads a checked processing brief, routes each file by mode:
  MOV   -> moving-average detrend + Gaussian blur
  BIEXP -> bi-exponential detrend + Gaussian blur
  BOTH  -> BIEXP then MOV
  NONE  -> skip

Usage:
    python img_proc.py --brief data/proc_brief_20260512_002_checked.txt
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import tifffile
from rich.console import Console

from functions import (
    als_baseline_run,
    biexp_detrend,
    check_cuda,
    gaussian_blur_run,
    get_memory_usage,
    mov_detrend,
    sample_tau,
)

SIGMA = 6.0
ALS_LAM = 1e2
ALS_P = 0.05
ALS_N_ITER = 10
console = Console()


# ── Brief parsing ─────────────────────────────────────────────────────────────


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

        if stripped == "Picked:":
            in_picked = True
            continue

        if in_picked:
            if stripped.startswith("["):
                m = re.match(r"\[(.+?),\s*(\w+),\s*(\w+)\]", stripped)
                if m:
                    filename, proc, mode = m.groups()
                    if mode != "NONE":
                        entries.append({"file": filename.strip(), "proc": proc, "mode": mode})
            else:
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

    console.log("  ALS baseline...")
    baseline = als_baseline_run(blurred, ALS_LAM, ALS_P, ALS_N_ITER, cuda_available)
    tifffile.imwrite(proc_dir / f"{stem}_MOV_BASELINE.tif", baseline.astype(np.float16))
    console.log(f"  Saved {stem}_MOV_BASELINE.tif  ({time.time() - t0:.1f}s)")

    console.log("  Computing dF/F0...")
    dff0 = ((blurred.astype(np.float32) - baseline) / baseline).astype(np.float16)
    tifffile.imwrite(proc_dir / f"{stem}_MOV_DFF0.tif", dff0)
    console.log(f"  Saved {stem}_MOV_DFF0.tif  ({time.time() - t0:.1f}s)")

    del img, detrended, blurred, baseline, dff0


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
    tifffile.imwrite(proc_dir / f"{stem}_BIEXP_GAUSS.tif", blurred.astype(np.float16))
    console.log(f"  Saved {stem}_BIEXP_GAUSS.tif  ({time.time() - t0:.1f}s)")

    console.log("  ALS baseline...")
    baseline = als_baseline_run(blurred, ALS_LAM, ALS_P, ALS_N_ITER, cuda_available)
    tifffile.imwrite(proc_dir / f"{stem}_BIEXP_BASELINE.tif", baseline.astype(np.float16))
    console.log(f"  Saved {stem}_BIEXP_BASELINE.tif  ({time.time() - t0:.1f}s)")

    console.log("  Computing dF/F0...")
    dff0 = ((blurred.astype(np.float32) - baseline) / baseline).astype(np.float16)
    tifffile.imwrite(proc_dir / f"{stem}_BIEXP_DFF0.tif", dff0)
    console.log(f"  Saved {stem}_BIEXP_DFF0.tif  ({time.time() - t0:.1f}s)")

    del img, detrended, blurred, baseline, dff0


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run(brief_path: Path, cuda_available: bool) -> None:
    """Parse brief and process each file according to its MODE."""
    entries, raw_dir, proc_dir = parse_brief(brief_path)
    proc_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"[bold green]Brief: {brief_path.name}")
    console.log(f"  raw  -> {raw_dir}")
    console.log(f"  proc -> {proc_dir}")
    console.log(f"  {len(entries)} file(s) to process  (cuda={cuda_available})")

    for entry in entries:
        file = entry["file"]
        mode = entry["mode"]
        fpath = raw_dir / file

        if not fpath.exists():
            console.log(f"[yellow][SKIP] {file} not found")
            continue

        console.log(f"\n[bold]{'=' * 60}")
        console.log(f"[bold]{file}  MODE={mode}")

        if mode == "MOV":
            process_mov(file, raw_dir, proc_dir, cuda_available)
        elif mode == "BIEXP":
            process_biexp(file, raw_dir, proc_dir, cuda_available)
        elif mode == "BOTH":
            process_biexp(file, raw_dir, proc_dir, cuda_available)
            process_mov(file, raw_dir, proc_dir, cuda_available)
        else:
            console.log(f"[yellow]  Unknown mode '{mode}', skipping")

    console.log(f"\n[bold green]{'=' * 60}")
    console.log("[bold green]All done!")


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image preprocessing pipeline")
    parser.add_argument("--brief", required=True, type=Path, help="Path to _checked.txt brief file")
    args = parser.parse_args()

    _cuda_available = check_cuda() if check_cuda is not None else False
    run(args.brief, _cuda_available)
