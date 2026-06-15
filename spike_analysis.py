"""
spike_analysis.py  --  Spike-aligned image analysis pipeline.
=============================================================
Reads a processing list, loads the appropriate processed TIFF (*_GAUSS.tif or
*_ALS.tif) together with its paired ABF file, runs spike detection and alignment,
spatial categorization, region analysis, and exports results.

Proc list format (5 fields per entry):
  [raw_tiff_name, gauss_exists, do_processing, detrend_mode, paired_abf]

Detrend mode selects which processed TIFF prefix is loaded:
  BIEXP -> *_BIEXP_GAUSS.tif  or  *_BIEXP_ALS.tif
  MOV   -> *_MOV_GAUSS.tif    or  *_MOV_ALS.tif

Usage:
    python spike_analysis.py --proc_list data/proc_20260601_000.txt [--detrend BIEXP] [--use_als]
"""

import argparse
from pathlib import Path

from utils.params import RAW_ABFS_DIR

# ── Proc list parsing ─────────────────────────────────────────────────────────


def _parse_bracket(stripped: str, proc_dir: Path, detrend_mode: str, use_als: bool) -> dict | None:
    """Return {"tiff_path", "abf_path"} for one bracket line, or None if the target TIFF is missing."""
    parts = [p.strip() for p in stripped.strip("[]").split(",")]
    if len(parts) < 5:
        return None
    stem = Path(parts[0]).stem
    abf_name = parts[4]

    suffix = "_ALS.tif" if use_als else "_GAUSS.tif"
    candidate = proc_dir / f"{stem}_{detrend_mode}{suffix}"
    if not candidate.exists():
        return None

    return {
        "tiff_path": candidate,
        "abf_path": RAW_ABFS_DIR / abf_name,
    }


def parse_proc_list(
    proc_list_path: Path,
    detrend_mode: str = "BIEXP",
    use_als: bool = False,
) -> tuple[list[dict], Path]:
    """Return spike analysis entries and proc_dir from a proc list file.

    Args:
        proc_list_path: Path to the proc list file (proc_*.txt).
        detrend_mode:   Which detrend variant to load — "BIEXP" or "MOV".
        use_als:        If True, load *_ALS.tif; otherwise load *_GAUSS.tif.

    Returns:
        (entries, proc_dir) where each entry is:
            {"tiff_path": Path, "abf_path": Path}
        Only entries whose target TIFF exists on disk are included.
    """
    proc_dir: Path | None = None
    for line in proc_list_path.read_text().splitlines():
        if line.startswith("dir_proc_tiffs:"):
            proc_dir = Path(line.split(":", 1)[1].strip())
            break
    if proc_dir is None:
        msg = f"Missing dir_proc_tiffs in {proc_list_path}"
        raise ValueError(msg)

    text = proc_list_path.read_text()
    entries: list[dict] = []
    in_picked = False

    for line in text.splitlines():
        if line.strip().startswith("Picked:"):
            in_picked = True
            continue
        if in_picked:
            if line.strip().startswith("["):
                entry = _parse_bracket(line.strip(), proc_dir, detrend_mode, use_als)
                if entry:
                    entries.append(entry)
            else:
                in_picked = False

    return entries, proc_dir

# ABFClip



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike-aligned image analysis pipeline")
    parser.add_argument("--proc_list", required=True, type=Path, help="Path to proc list file (proc_*.txt)")
    parser.add_argument("--detrend", choices=["BIEXP", "MOV"], default="BIEXP", help="Detrend mode (default: BIEXP)")
    parser.add_argument("--use_als", action="store_true", help="Load *_ALS.tif instead of *_GAUSS.tif")
    args = parser.parse_args()

    entries, proc_dir = parse_proc_list(args.proc_list, args.detrend, args.use_als)
    print(f"Found {len(entries)} entries in {args.proc_list.name}")
    for e in entries:
        print(f"  {e['tiff_path'].name}  +  {e['abf_path'].name}")
