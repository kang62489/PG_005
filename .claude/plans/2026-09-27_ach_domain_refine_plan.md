# ach_domain_analysis.py refinement plan (2026-09-27)

Goal: tidy the spike-aligned outputs, tighten the hotspot threshold, add striatum-masked flow display +
DV / ML flow direction + per-recording flow-pattern ratio, label hotspots stim-induced / spontaneous (ABF CH2),
then compare hotspot areas (estim-induced vs patched spontaneous vs natural spontaneous zones).

Workflow: phase by phase, user checks after each phase. Scratch runs -> `output/test01/<phase>/`.
Test set (build `output/test01/ana_test01.txt`): `2025_06_11-0003` (ABF `2025_06_11_0004`),
`2025_11_27-0018` (paired ABF: look up in rec_data.db / proc list), `2025_12_15-0012` (ABF `2025_12_15_0008`).
First run of the CURRENT code -> `output/test01/before/` (baseline for every later compare).

---

## Phase 1 -- Clean `output/` (item 1)  [x] done 2026-09-27 (user ran it; scratch numbering restarts at `test01`)

## Phase 2 -- Output layout (items 6 + 5), must be data-identical to `before/`  [x] done 2026-09-27 (identical; CAT 17.8 MB -> 118 KB)
- Item 6: `{ana_list}_cells.xlsx` -> `results/` (was `results/spikes/`), `ach_domain_analysis.py` `run()` (~line 502).
- Item 5: CAT TIFF `tifffile.imwrite(..., compression="zlib")` in `ResultsExporter._export_categorized_stack()`
  (`classes/results_exporter.py:388`), same as `SpZoneAnalyzer.save()` (`classes/sp_zone_analyzer.py:187`).
- Check: CAT pixel data, MED, DB rows, PNGs identical; only CAT file size + xlsx location change.

## Phase 3 -- Threshold (items 2 + 3), changes results  [x] done 2026-09-27 (user: keep; reliability 60X 92->19 %, 10X 92->45 %, 40X 100->80 %, all still significant)
- Item 2: `BASELINE_SIGMA_MULT` 1.5 -> 2.0 (`classes/spatial_categorization.py:38`). Shared on purpose:
  `classes/spike_reliability.py:33` imports it, so reliability and the final CAT stay in sync (user, Q1).
- Item 3: remove bright objects < 2700 µm² after the morphological open/close (`_apply_morphological()`),
  converted to px per objective with the RegionAnalyzer px/µm table (`region_analyzer.py:37-39`):
  10X ~1,519 px, 40X ~24,300 px, 60X ~54,675 px. `SpatialCategorizer` needs the objective / um_per_pixel
  passed in (both call sites: `ach_domain_analysis.py` Step 4 and `SpikeReliabilityChecker`).
- Run -> `output/test01/after_thr/`, compare against `before/` (reliability %, cluster areas, significant flag).

## Phase 4 -- Flow (items 10 + 11 + 12 + 9)
- 4a (items 10 + 11) [x] done 2026-09-27, display only (data identical). `--stbd` (default bd_{tag}.json next to
  the ana list), `striatum_of()` in ach_domain_analysis.py.
  FLOW.png 3 rows: MED z + arrows (striatum / full FOV), CAT mask + arrows, speed (striatum / full FOV).
  STREAMLINES.png 2 rows: MED z + streamlines (striatum / full FOV), CAT mask + streamlines + pattern label.
  MED z display: trimmed baseline z, range z 1 -> median over pairs of the max z inside each keep_mask; gray colorbar.
  Larger fonts (titles 13, suptitle 16, colorbar 13/11, scale bar 12).
- 4b (item 12) next, then 4c (item 9).
- Items 10 / 11: FLOW + STREAMLINES PNGs masked with the STRIATUM mask from `data/bd_{proc_list}.json`
  (`striatum_outline_px` -> `outline_mask()`, `functions/st_boundary.py:285`); no outline (40X / 60X) ->
  unmasked full FOV (user, Q6). DISPLAY ONLY:
  | Use | Mask |
  |-----|------|
  | TV-L1 flow (`hotspot_flow.py:46`) | none (unchanged) |
  | Pattern label fit (`region_analyzer.py:243` -> `flow_pattern.py`) | CAT keep mask (unchanged) |
  | FLOW + STREAMLINES PNGs | striatum (full FOV if no outline) |
  Need a way to find the bd json for the run (e.g. `--stbd` like `spontaneous_analysis.py`).
- Item 12: per significant recording, count the 5 pair labels (FLOW_OFFSETS spike-1->spike ... +3->+4);
  None pairs out of the denominator. e.g. None / source / aniso / aniso / sink -> 50 % aniso, 50 % src/sink.
  New `experiments` columns `n_flow_labelled`, `flow_aniso_pct`, `flow_srcsink_pct`; stats block gets a
  per-objective table (10X vs 40X / 60X). Expectation: 10X more hybrid, 40X / 60X more anisotropic (FOV).
  (FOV-crop control dropped by user.)
- Item 9: drift vector -> DV / ML using `dorsal_vec` / `medial_vec` from the bd json; report nearest pole +
  tilt (<= 45°) toward the adjacent pole, e.g. `V 43° M`, `M 10° D` (user OK). `flow_pairs` columns
  `dv_ml_pole`, `dv_ml_tilt_deg`, `dv_ml_toward` (None when no orientation / label None).

## Phase 5 -- Stim-induced vs spontaneous label (item 7)
- `AbfClip`: read CH2 = `abf_dataset[1]` (0 = Vm, 3 = TTL frame cut); detect pulse trains inside the TTL window.
- Per RECORDING property (user, Q4a): trains present -> `estim_induced`, else `spontaneous`. Does NOT change
  how the MED is built. Store as an `experiments` column (e.g. `hotspot_origin`).

## Phase 6 -- Area comparison (item 8), plan-mode details first
- Claim: a single spike of a single neuron can release a large hotspot -> a local striatal area can be
  modulated by one neuron.
- Groups: MED hotspot area (spike / spike+1 µm²) of `estim_induced` vs `spontaneous` recordings, vs natural
  spontaneous zone areas from `spontaneous_analysis` (10X only) for the same files.
- Natural-zone filter (user, Q4b): per zone, MIN distance from its centroid (`zone_centroids`,
  `sp_zone_analyzer.py:161`) to the MED hotspot centroid(s); < 150 px (~200 µm at 10X) -> patched neuron's
  zone, excluded; else natural spontaneous. Check whether zone centroids are saved in `footprints/*.npz`
  or the xlsx.
