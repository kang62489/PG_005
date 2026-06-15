"""
spike_analysis.py  --  Spike-aligned image analysis pipeline.
=============================================================
Reads an analysis list (ana_list_*.txt), loads the appropriate processed TIFF
(*_GAUSS.tif or *_ALS.tif) together with its paired ABF file, runs spike
detection and alignment, spatial categorization, region analysis, and exports
results.

Ana list format (5 fields per bracket entry):
  [raw_tiff_name, gauss_exist, als_exist, paired_abf, abf_exist]

Detrend mode selects which processed TIFF prefix is loaded:
  BIEXP -> *_BIEXP_GAUSS.tif  or  *_BIEXP_ALS.tif
  MOV   -> *_MOV_GAUSS.tif    or  *_MOV_ALS.tif

Usage:
    python spike_analysis.py --ana_list data/ana_list_20260601_000.txt [--detrend BIEXP] [--use_als]
"""

import argparse
from pathlib import Path

from rich.console import Console

from classes import AbfClip

console = Console()


# ── Ana list parsing ──────────────────────────────────────────────────────────


def _parse_bracket(
    stripped: str,
    proc_dir: Path,
    raw_abfs_dir: Path,
    detrend_mode: str,
    use_als: bool,
) -> dict | None:
    """Return {"proc_tiff_path", "raw_abf_path"} for one bracket line, or None if guards fail."""
    parts = [p.strip() for p in stripped.strip("[]").split(",")]
    if len(parts) < 5:
        return None
    raw_tiff_name, gauss_exist, als_exist, paired_abf, abf_exist = parts[:5]

    if use_als and als_exist != "YES":
        return None
    if not use_als and gauss_exist != "YES":
        return None
    if abf_exist != "YES":
        return None

    stem = Path(raw_tiff_name).stem
    suffix = "_ALS.tif" if use_als else "_GAUSS.tif"

    return {
        "proc_tiff_path": proc_dir / f"{stem}_{detrend_mode}{suffix}",
        "raw_abf_path": raw_abfs_dir / paired_abf,
    }


def parse_ana_list(
    ana_list_path: Path,
    detrend_mode: str = "BIEXP",
    use_als: bool = False,
) -> tuple[list[dict], Path, str, str]:
    """Parse an ana list file and return entries with resolved paths.

    Args:
        ana_list_path: Path to the ana list file (ana_list_*.txt).
        detrend_mode:  Which detrend variant to load — "BIEXP" or "MOV".
        use_als:       If True, load *_ALS.tif; otherwise load *_GAUSS.tif.

    Returns:
        (entries, results_dir, detrend_mode, normalization) where each entry is:
            {"proc_tiff_path": Path, "raw_abf_path": Path}
        Only entries passing all existence guards are included.
    """
    lines = ana_list_path.read_text().splitlines()

    proc_dir: Path | None = None
    raw_abfs_dir: Path | None = None
    results_dir: Path | None = None

    for line in lines:
        if line.startswith("dir_proc_tiffs:"):
            proc_dir = Path(line.split(":", 1)[1].strip())
        elif line.startswith("dir_raw_abfs:"):
            raw_abfs_dir = Path(line.split(":", 1)[1].strip())
        elif line.startswith("dir_results:"):
            results_dir = Path(line.split(":", 1)[1].strip())

    missing = [
        k
        for k, v in {
            "dir_proc_tiffs": proc_dir,
            "dir_raw_abfs": raw_abfs_dir,
            "dir_results": results_dir,
        }.items()
        if v is None
    ]
    if missing:
        msg = f"Missing footer keys in {ana_list_path}: {', '.join(missing)}"
        raise ValueError(msg)

    entries: list[dict] = []
    in_picked = False

    for line in lines:
        if line.strip().startswith("Picked:"):
            in_picked = True
            continue
        if in_picked:
            if line.strip().startswith("["):
                entry = _parse_bracket(line.strip(), proc_dir, raw_abfs_dir, detrend_mode, use_als)
                if entry:
                    entries.append(entry)
            else:
                in_picked = False

    normalization = "ALS" if use_als else "GAUSS"
    return entries, results_dir, detrend_mode, normalization


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike-aligned image analysis pipeline")
    parser.add_argument("--ana_list", required=True, type=Path, help="Path to ana list file (ana_list_*.txt)")
    parser.add_argument("--detrend", choices=["BIEXP", "MOV"], default="BIEXP", help="Detrend mode (default: BIEXP)")
    parser.add_argument("--use_als", action="store_true", help="Load *_ALS.tif instead of *_GAUSS.tif")
    args = parser.parse_args()

    entries, results_dir, detrend_mode, normalization = parse_ana_list(args.ana_list, args.detrend, args.use_als)
    console.print(f"Found {len(entries)} entries in {args.ana_list.name}")

    for entry in entries:
        console.print(f"\n[cyan]{entry['proc_tiff_path'].name}  +  {entry['raw_abf_path'].name}[/cyan]")
        clip = AbfClip(
            proc_tiff_path=entry["proc_tiff_path"],
            raw_abf_path=entry["raw_abf_path"],
            results_dir=results_dir,
            detrend_mode=detrend_mode,
            normalization=normalization,
        )
