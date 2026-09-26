# Spontaneous analysis refinement (2026-09-26)

Scope: `spontaneous_analysis.py`, `classes/sp_zone_analyzer.py`, `functions/plot_results.py` (Step 6),
`functions/fit_hist.py` (threshold fit), `run_on_saion.slm`.
Executed phase by phase -- user checks after each phase (assembly-line style). No code is touched before approval.

Test set (scratch, foreground runs): `output/test9/proc_test9.txt` = subset of `data/proc_20260922_000.txt`:

| Recording | Why |
|---|---|
| `2025_06_11-0003` | normal GACh3.0, 44 zones -- regression reference |
| `2025_11_27-0009` | 1 zone = 1.86e6 µm² (whole FOV) -> giant-hotspot filter |
| `2025_11_27-0007` | 2 zones, median 3.34 Hz vs 30 s period -> artifact check |
| `2025_11_27-0026` | threshold 0.095 (others ~0.18), 103 zones, 99 proximity |
| `2026_01_08-0037` | normal, 44 zones, 437 detections -> page-count / timing check for new maps |

Before any edit: run the current code on the test set -> `output/test9/before/` (baseline for comparisons).

---

## Phase 1 -- Output layout + mask size (TODO G)

**Why the MASK is 1.26 GB:** `save()` writes the full per-frame mask uncompressed as uint8:
1200 frames × 1024 × 1024 × 1 byte = 1,258,291,200 B (+ header = the 1,258,490,490 B seen).

Changes:
1. `_ZONE_MASK.tif` -> `spontaneous/mask/`, written only with the new `--save_mask` flag (default OFF),
   and with `compression="zlib"` (binary 0/255 data compresses very well; still opens in ImageJ/Fiji).
2. `_ZONES.npz` (footprints + contours) -> `spontaneous/footprints/`.
3. `_ZONES.xlsx` and `_ZONE_MAPS.tif` directly in `spontaneous/` (the `zone_maps/` subfolder goes away).
4. `spontaneous_stats.png` no longer exported (call + import removed from `spontaneous_analysis.py`;
   `plot_zone_stats()` kept in `plot_results.py` for later). `spontaneous_summary.xlsx` stays.
5. `SpontaneousZoneAnalyzer.save(out_dir, stem, save_mask=False, debug=False)` builds the subfolders.
6. Module docstring + `run_on_saion.slm` comment updated to the new layout.

Verify: xlsx / npz contents identical to `before/`; mask (when `--save_mask`) pixel-identical, size reported.

---

## Phase 2 -- Drop giant hotspots before grouping (TODO H)

10X FOV = 1024 × 1024 px; 80 % = 838,861 px (≈ 1.49e6 µm² at 1.333 µm/px).

Changes:
1. CONFIG (Step 2): `MAX_HOTSPOT_FRAC = 0.8  # px fraction of the frame: larger merged hotspots are artifacts (10X)`.
2. In `spatiotemporally_connect_hotspots()`, after `merge_adjacent_hotspots()` (so merged fragments are judged
   as one hotspot), skip `area > MAX_HOTSPOT_FRAC * H * W` next to the existing `area < th_small_hotspots` check.
3. Log `  N giant hotspot(s) dropped (> 80 % of frame) in frames [...]` (no summary column -- user dropped it).
4. **Explore (user idea, 2026-09-26): quantile clip of hotspot areas, e.g. keep (0.001, 0.999).**
   Goal: remove small speckles and giant hotspots from over-exposed first frames.
   Scratch first (`output/test9/hotspot_areas.py`): per-recording and pooled area distributions,
   area vs frame index, what (0.001, 0.999) would drop. Open points: per recording vs pooled quantiles,
   and whether it replaces or adds to the 80 % rule. Decide after looking at the plots.

Note: this is a per-hotspot filter. A zone (union of footprints over many frames) can still grow large;
check the test set after the filter before deciding whether a zone-level check is also needed.

Verify: `2025_11_27-0009` whole-FOV zone gone; `2025_06_11-0003` / `2026_01_08-0037` unchanged vs `before/`.

---

## Phase 3 -- New `_ZONE_MAPS.tif` (TODO I)

### 3a. z-score (shared scale)
- `fit_hist.py`: split the fit out of `find_background_threshold()` -> new `fit_background(stack) -> (center, sigma)`;
  `find_background_threshold()` becomes `center + sigma_ratio * sigma` on top of it (same value, same callers).
- Analyzer stores `bg_center`, `bg_sigma` (threshold unchanged = `bg_center + sigma_ratio * bg_sigma`).
- z image = `(frame - bg_center) / bg_sigma` (`img_zscore_convert` already exists) -> threshold sits at z = sigma_ratio.
- One display range per recording: `vmin, vmax` = min / max of z over **all pages' backgrounds**
  (max projection + every detection frame), applied to every page, one colorbar labelled "z".

### 3b. Page 1 -- overview
- Background = **max projection of all frames** (z-scored).
- All zones as translucent fills + id labels (as now) + scale bar.
- Title: `{stem}, {sensor} | N zones -- a trace-corr, b proximity, c isolated | thr = peak 0.120 + 1.5σ (0.034) = 0.171`.

### 3c. Pages 2.. -- one page per frame with ≥ 1 detected hotspot (replaces the one-page-per-zone pages)
- Background = that frame, z-scored, shared vmin/vmax.
- Contour of every zone that a hotspot in this frame belongs to (zone colour, width 2.5) + zone id label;
  thin white outline of the frame's own detected footprint so "zone vs actual hotspot" is visible (approved).
- Title: `frame 123 (6.15 s) | zones 3, 7 | thr = 0.171 (peak + 1.5σ) | max z = 4.82`
  (max z = maximum z inside that frame's detected footprints; `meanproj` text removed).

### 3d. CLI (approved 2026-09-26)
- Remove `--proj` (page 1 is always max proj) and `--color` (z pages use gray + colorbar).

### 3e. Risk: page count / time / size
Pages = frames with detections (up to a few hundred). Today: 60 pages ≈ 29 s, ≈ 5 MB/page raw RGB at 11 in × 120 dpi.
- Step 1: measure on `2026_01_08-0037` (437 detections) -- pages, seconds, MB.
- If too slow/big, options (pick after measuring): smaller figure (8 in), or draw pages directly with numpy
  (colormap LUT + `find_contours` rasterised) instead of matplotlib -- much faster.

Verify: open the TIFF in Fiji for the 5 test recordings; confirm page 1 background, contours match the hotspot,
titles, and identical colour scale across pages.

---

## Phase 4 -- Merge zones that are really the same place (TODO J)

Today: proximity grouping (115 px, complete linkage) runs **only on tracks left over after trace-corr grouping**,
so a trace-corr zone, a proximity zone, and an isolated track can sit on top of each other as 3 zones.

### 4a. Explore first (scratch `output/test9/zone_overlap.py`)
For every zone pair on the test set compute:
- centroid distance (px),
- IoU = |A∩B| / |A∪B|,
- overlap coefficient = |A∩B| / min(|A|, |B|) -- catches a small zone inside a big one, which IoU misses.
Output: histograms + top-pair table + side-by-side contour PNGs of candidate pairs -> user picks metric + threshold.

### 4b. Then implement (after your pick)
- New Step 3b in `sp_zone_analyzer.py`: `merge_close_zones()` -- union-find over pairs passing the rule,
  merged zone = union of member tracks, `source` e.g. `merged (trace_corr #2 + isolated #5)`, stats recomputed.
- CONFIG: chosen metric + threshold. Starting candidate (to be confirmed by 4a): overlap coefficient ≥ 0.5.
- Log `  N zones -> M after merging close zones`.

Verify: zone counts before/after per test recording; new ZONE_MAPS pages to eyeball merged zones.

---

## After each phase
- `ruff check` on touched files; neat-refactor on touched .py files before calling the phase done.
- State every output path; foreground runs only; scratch in `output/test9/`.
- Full re-run on saion only after all phases are accepted.
