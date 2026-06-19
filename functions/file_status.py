## Modules
# Standard library imports
import re
from pathlib import Path

PROC_FILE_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<mode>BIEXP|MOV)_(?P<kind>GAUSS|ALS)\.tif$")


def build_filename_index(dir_path: Path, pattern: str) -> set[str]:
    """Scan dir_path once, return the set of filenames matching pattern."""
    return {f.name for f in dir_path.glob(pattern)}


def raw_tiff_ready(index: set[str], dor: str, tiff_serial: str) -> str:
    """Check READY/MISSING for a DOR/TIFF_SERIAL pair against a pre-built filename index."""
    return "READY" if f"{dor}-{tiff_serial}.tif" in index else "MISSING"


def abf_ready(index: set[str], abf_name: str) -> str:
    """Check YES/No for a paired ABF filename against a pre-built filename index."""
    return "YES" if abf_name in index else "No"


def build_proc_file_index(dir_path: Path) -> dict[str, dict[str, list[str]]]:
    """Scan dir_path once, indexing GAUSS/ALS modes by '{dor}-{tiff_serial}' stem."""
    index: dict[str, dict[str, list[str]]] = {}
    for f in dir_path.glob("*.tif"):
        m = PROC_FILE_PATTERN.match(f.name)
        if not m:
            continue
        index.setdefault(m["stem"], {"GAUSS": [], "ALS": []})[m["kind"]].append(m["mode"])
    return index


def gauss_exists(index: dict[str, dict[str, list[str]]], dor: str, tiff_serial: str) -> str:
    """Return which detrend mode(s) have a GAUSS file, or 'No' if none."""
    gauss_list = index.get(f"{dor}-{tiff_serial}", {}).get("GAUSS", [])
    if not gauss_list:
        return "No"
    if "BIEXP" in gauss_list and "MOV" in gauss_list:
        return "BIEXP & MOV"
    return gauss_list[0]


def als_exists(index: dict[str, dict[str, list[str]]], dor: str, tiff_serial: str) -> str:
    """Return which detrend mode(s) have an ALS file, or 'No' if none."""
    als_list = index.get(f"{dor}-{tiff_serial}", {}).get("ALS", [])
    if not als_list:
        return "No"
    if "BIEXP" in als_list and "MOV" in als_list:
        return "BIEXP & MOV"
    return als_list[0]


def gauss_ready(index: dict[str, dict[str, list[str]]], dor: str, tiff_serial: str, detrend: str) -> str:
    """Check YES/No for one specific detrend mode's GAUSS file against a pre-built proc file index."""
    return "YES" if detrend in index.get(f"{dor}-{tiff_serial}", {}).get("GAUSS", []) else "No"


def als_ready(index: dict[str, dict[str, list[str]]], dor: str, tiff_serial: str, detrend: str) -> str:
    """Check YES/No for one specific detrend mode's ALS file against a pre-built proc file index."""
    return "YES" if detrend in index.get(f"{dor}-{tiff_serial}", {}).get("ALS", []) else "No"
