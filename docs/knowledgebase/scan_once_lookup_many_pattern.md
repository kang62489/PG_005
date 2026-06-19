---
keywords: scan once, lookup many, glob, exists, stat, network mount, O(rows x files), index, set membership, dict lookup, filesystem round-trip, check_file_status
related: signal_slot_cross_controller.md
---

# 2026-06-19

## "Scan once, look up many times" — why `check_file_status` got faster

### The problem
Two controllers (`ctrl_img_proc.py`, `ctrl_align_spike.py`) build a table of N picked
entries, and for each row need to know whether some related file exists on disk
(a GAUSS/ALS processed TIFF, a paired ABF file). The naive way is to ask the
filesystem **inside the per-row loop** — either by re-globbing the whole directory
per row, or by calling `.exists()` per row. Both approaches repeat filesystem work
N times instead of once, and on network-mounted drives the *latency per call*
dominates, not the data size.

### Two flavors of the same mistake

**1. Re-globbing per row** (`ctrl_img_proc.py`, old `_gauss_exists`):
```python
def _gauss_exists(self, dir_path: Path, dor: str, tiff_serial: str) -> str:
    examine_file_gauss = list(dir_path.glob(f"{dor}-{tiff_serial}*_GAUSS*.tif"))
```
`Path.glob()` lists the *entire directory* and pattern-matches every filename,
every single call. With 200 rows and 2,000 files in the folder, calling this
(plus the equivalent `_als_exists`) per row means **400 directory scans**, each
re-walking 2,000 files → **800,000 comparisons** total. This is the
`O(rows × files)` blowup.

**2. Individual `.exists()` per row** (`ctrl_align_spike.py`, old `_load_entries`):
```python
"GAUSS_EXIST?": "YES" if (proc_dir / f"{tiff_stem}_{detrend}_GAUSS.tif").exists() else "No",
"ALS_EXIST?":   "YES" if (proc_dir / f"{tiff_stem}_{detrend}_ALS.tif").exists() else "No",
"ABF_READY?":   "YES" if (self._raw_abfs_dir() / abf_name).exists() else "No",
```
Each `.exists()` is its own `stat()` syscall — one round trip per file. At 200
rows × 3 checks = **600 individual round trips**. On an SMB/NFS share at ~5 ms
latency each, that's **~3 seconds spent just waiting**, even though almost no
data is actually transferred.

### Worked example with real numbers (`ctrl_img_proc.py`)
Say the pick list has **3 rows** and `dir_processed` contains these **5 files**:
```
20260101-001_BIEXP_GAUSS.tif
20260101-001_BIEXP_ALS.tif
20260101-002_MOV_GAUSS.tif
20260102-001_BIEXP_GAUSS.tif
20260102-001_BIEXP_ALS.tif
```

**Old code** — `_gauss_exists` + `_als_exists` each re-glob the directory, per row:

| Row | `_gauss_exists` call | `_als_exists` call |
|---|---|---|
| `20260101-001` | lists all 5 files | lists all 5 files |
| `20260101-002` | lists all 5 files | lists all 5 files |
| `20260102-001` | lists all 5 files | lists all 5 files |

= **6 directory listings**, each re-scanning all 5 files → **30 filename comparisons**
for only 3 rows.

**New code** — `_build_proc_file_index` lists the directory **once**, regex-matches
each of the 5 filenames **once**, and builds:
```python
{"20260101-001": {"GAUSS": ["BIEXP"], "ALS": ["BIEXP"]},
 "20260101-002": {"GAUSS": ["MOV"],   "ALS": []},
 "20260102-001": {"GAUSS": ["BIEXP"], "ALS": ["BIEXP"]}}
```
Each row then does `index.get(key)` — an O(1) dict lookup, no filesystem access.

= **1 directory listing** (5 regex matches) + 6 free dict lookups, instead of 6
listings + 30 comparisons. Scale this 3-row/5-file toy example up to 200
rows/2,000 files and the gap becomes 400 scans (800,000 comparisons) vs. 1 scan.

### The fix: scan once, build an index/set, then do O(1) lookups per row

**`ctrl_img_proc.py`** — `_build_proc_file_index` walks the directory **once**,
regex-matches every filename **once**, and builds a dict keyed by
`"{dor}-{tiff_serial}"`:
```python
def _build_proc_file_index(self, dir_path: Path) -> dict[str, dict[str, list[str]]]:
    """Scan dir_path once, indexing GAUSS/ALS modes by '{dor}-{tiff_serial}' stem."""
    index: dict[str, dict[str, list[str]]] = {}
    for f in dir_path.glob("*.tif"):
        m = PROC_FILE_PATTERN.match(f.name)
        if not m:
            continue
        index.setdefault(m["stem"], {"GAUSS": [], "ALS": []})[m["kind"]].append(m["mode"])
    return index
```
Each row then does `index.get(key, {}).get("GAUSS", [])` — an O(1) dict lookup,
zero filesystem access. Total cost: **1 directory scan** (regardless of row count)
+ cheap dict lookups per row.

**`ctrl_align_spike.py`** — list each relevant directory **once** into a `set`
of filenames, then test membership instead of stat-ing each file:
```python
proc_files = {f.name for f in proc_dir.glob("*.tif")}
abf_files = {f.name for f in self._raw_abfs_dir().glob("*.abf")}
...
"GAUSS_EXIST?": "YES" if f"{tiff_stem}_{detrend}_GAUSS.tif" in proc_files else "No",
"ABF_READY?": "YES" if abf_name in abf_files else "No",
```
`filename in some_set` is an in-memory hash lookup — no syscall. Total cost:
**2 directory listings total** (not 3×N round trips).

### The general principle
> If a loop needs to ask "does X exist / what category is X" for many items
> against the *same* fixed collection (a directory's contents, a DB table, etc.),
> don't ask the source once per item. Scan/query the source **once**, build an
> in-memory index (`dict` or `set`), then do O(1) lookups inside the loop.
>
> This matters most when each individual "ask" carries fixed latency overhead
> (network filesystem stat/glob, DB round-trip, HTTP request) — in that case the
> *number of calls* dominates cost, not the amount of data per call.

### When this pattern applies elsewhere in this codebase
Watch for any per-row `.exists()`, `.glob()`, or repeated directory listing
inside a loop over picked/processed entries — the same index-once pattern
applies. See [[signal_slot_cross_controller.md]] for how these controllers also
avoid repeated work via `QFileSystemWatcher` removal in favor of explicit
"Refresh Status" buttons.
