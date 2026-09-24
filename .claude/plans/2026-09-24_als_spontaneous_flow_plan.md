# Plan: Reorganize TODOs, add spontaneous-zone analysis, refresh the spike-aligned pipeline

## Context

The paper now needs two lines of evidence, both built on `*_BIEXP_ALS.tif` in `proc_tiffs/`:

- **Spontaneous ACh.** Hotspots tile most of the striatum and fire out of phase, at different frequencies. The code for this is `../PG_010/sp_ach_zones.py`, a standalone script that is not yet part of PG_005.
- **Spike-induced ACh.** Hotspots are local and restricted. The current ring-latency metric is weak. It should be replaced by the pre-masked TV-L1 flow result from Session 62 (`prototype_flow_analysis.py`).

At the same time, the involved scripts get the `sp_ach_zones.py` style: short docstrings and step-banner blocks (`# ===== STEP N -- ...`), so the workflow reads top to bottom.

Work proceeds "assembly-line": one phase at a time, verified on real data, with confirmation before each edit.

---

## Reorganized TODO list (replaces the six from Session 61)

| # | New TODO | Absorbs old item(s) | Phase |
|---|----------|---------------------|-------|
| A | Default the pipeline to ALS | (new, your point 1) | 0 |
| B | Merge sp_ach_zones → spontaneous analysis (10X, sensor-aware, GPU) | #1 wave-vs-hotspots, #4 (spontaneous side) | 1 |
| C | Reliability: success/failure Vm ±50 ms + AP threshold | #4 (induced side), Session 61 zoom-in | 2 |
| D | RegionAnalyzer: drop ring latency, add pre-masked flow | #5 locality argument | 3 |
| E | Flow metrics (quantify divergence / outflow) | #5 locality argument | 4 |
| F | Neatness refactor of touched scripts | (new) | done inside each phase |
| — | Still open, not in this plan | #2 separate by objective, #3 median vs segments, #6 dataset for Jeff | later |

## Execution order (agreed)

1. **First:** copy this plan to `.claude/plans/` (project rule). Write the reorganized TODO table above into `docs/continue_from_here.md` as a new "Refined TODOs" block at the top. Stop for the user to check.
2. **Then one phase at a time** (0 → 1 → 2 → 3 → 4). Stop after each phase with: what changed, verification output, and export paths. Wait for the user's OK before the next phase.

---

## Phase 0: ALS as default

- `ach_domain_analysis.py`: replace `--use_als` with `--use_gauss` (ALS is the default). Change the default of `run(..., use_als=True)`.
- `controllers/ctrl_align_spike.py` `_use_als()`: default the GUI toggle to ALS (check the view's checkbox default in `views/view_align_spike.py`).

---

## Phase 1: Spontaneous zones (sp_ach_zones → PG_005)

### Files

- **`classes/spontaneous_zones.py`** (new): `SpontaneousZoneAnalyzer(stack, obj, sensor, fps=20)`, with three public step methods mirroring the original:
  - `detect()`: background threshold (Gaussian fit to the histogram's left side) → cleaned mask
  - `group()`: per-frame hotspots → merge fragments → chain tracks → trace-corr grouping → centroid grouping
  - `map()`: zones → masks, contours, per-zone stats
  - The union-find, `merge_adjacent_hotspots`, and clustering helpers move over unchanged (logic first, speed later).

- **`functions/zone_kernels.py`** (new): CPU (`@njit(parallel=True)`) and GPU (`@cuda.jit`) versions of the hot loops, dispatched like `functions/fit_bg_hist.py` / `functions/detrend.py`:
  1. Threshold + 3×3 opening + closing, per frame. This is `ndimage.binary_opening`/`binary_closing` today, and is the full-stack cost.
  2. The histogram for the background fit. **Reuse** the GPU histogram kernel already in `functions/fit_bg_hist.py`.
  3. `detection_traces`: mean over each footprint for every frame, via a gather + segmented-sum kernel.
  - `fill_holes`, `ndimage.label`, `regionprops`, and linkage stay on CPU (scipy); they are per-frame/per-track and cheap.

- **`functions/plot_results.py`**: `plot_zone_overlay(...) -> Figure` and `plot_single_zone(...) -> Figure`. These are headless, following the project rule. Also add `plot_zone_stats(...) -> Figure`: histograms of zone area (µm²) and frequency (Hz), split by sensor.

- **`spontaneous_analysis.py`** (new CLI runner, root level, like `ach_domain_analysis.py`):
  - Reads an ana/pick list (reuses `list_parser`, `lookup_rec_from_db`).
  - Keeps `OBJ == "10X"` rows only.
  - Reads `SENSOR` (already a core column in `rec_data.db`; see `utils/params.py`). Unknown or other values are reported and still kept, tagged with the raw value.
  - Does not require a paired ABF.

### Output: `results/spontaneous/`, readable names

```
results/spontaneous/
├── {rec}_ALS_ZONES.xlsx        # sheets below
├── {rec}_ALS_ZONE_MASK.tif     # was mask_*.tif
├── {rec}_ALS_ZONES.npz         # footprints + contours in one file
├── zone_maps/{rec}/01_all_zones.png, 02_zone01.png, ...
└── spontaneous_summary.xlsx    # one row per recording + pooled zone table, with sensor column
```

| Old (corr_groups_*.xlsx) | New |
|---|---|
| sheet `groups` | `trace_corr_groups` |
| sheet `centroid_groups` | `proximity_groups` |
| sheet `resting` | `isolated_tracks` |
| sheet `zone_sizes` | `zone_stats` |
| `joint_label` / `labels` | `track_id` / `track_ids` |
| `frames` | `active_frames` |
| `group` | `group_id` |
| `zone`, `origin` | `zone_id`, `source` (e.g. `trace_corr #3`) |
| `n_members` | `n_tracks` |
| `pixel_area` | `area_px` + **`area_um2`** (10X: 0.75 px/µm, from `PIXEL_SCALE`) |
| `n_events`, `period_s`, `freq_hz` | `n_events`, `mean_period_s`, `mean_freq_hz` |

- `detections_raw_*.csv` and the separate `hotspot_footprints` npz are dropped by default. A `--debug` flag keeps them.

### Verification

1. **Parity first.** Run the port on the 7 PG_010 stacks (all already in PG_005 `proc_tiffs/`). Compare against `../PG_010/results/`: same threshold, identical mask TIF, same zone count and zone pixel areas.
2. **Then enable GPU.** The mask must be identical between CPU and GPU. Traces must match within float tolerance. Report timing.

---

## Phase 2: Reliability success/failure Vm + AP threshold

- **`functions/ap_threshold.py`** (new): `find_ap_threshold(time_ms, vm, dvdt_thresh=20.0) -> (t_ms, v_mV) | None`. This is the first sample before the peak where dV/dt ≥ 20 mV/ms, the standard Sekerli/Naundorf-style criterion. The ABF sampling rate (≥10 kHz) is ample for it.
  - *Alternative:* the peak of d²V/dt² (noisier). The 20 mV/ms criterion is recommended; it is a tunable constant.

- **`functions/plot_results.py`**: `plot_vm_success_vs_failure(detected_vm, failure_vm, thresholds, title) -> Figure`. It ports the prototype's 1×2 layout: peak-aligned, ±50 ms, grid on, threshold dots, and the mean±SD threshold per group in each title.

- **`classes/spike_reliability.py`**: new method `export_vm_groups(exporter, vm_segments, ...)`. It saves to `reliability/{stem}_VM_SUCCESS_FAIL.png`, next to the montage.

- **`ach_domain_analysis.py`**: one call right after `export_montage`.

- **Verify:** regenerate for `ana_20260915_000.txt`. Traces must match the Session 61 `output/reliability_group/*.png`. Threshold dots must sit at the AP upstroke knee.

---

## Phase 3: Replace latency with pre-masked flow

- **Remove** from `classes/region_analyzer.py`: `compute_ring_traces`, `compute_cluster_trace`, `_peak_offset_from_spike`, `get_peak_latency_ms`, `get_temporal_traces`, and the ring/trace parts of `_build_clusters`. Centroid and `R_lat` stay for display.
- **Remove** `plot_full_trace` / `latency/` export. Also update the ring references in `plot_spatiotemporal_summary` (`plot_results.py:507`).
- **Add `functions/hotspot_flow.py`**: `premasked_flow(med, cat, idx_from, idx_to)`, ported as-is from the prototype (skimage TV-L1, CPU only; no numba TV-L1 is worth writing).
- **Add to `RegionAnalyzer`**: `self.flow_pairs`, holding the 5 pairs (spike−1→spike … spike+3→spike+4) as `{label, u, v, keep_mask}`.
- **Add to `plot_results.py`**: `plot_flow_panels(...)`, ported from the prototype. It exports to the new `flow/` folder as `{stem}_FLOW.png` (replaces `latency/`).
- **DB:** stop writing `peak_latency_ms`. The column stays so old DBs don't break, following the existing `_ensure_columns` pattern. Remove it from `compute_region_stats` / `build_stats_report`.
- **Verify:** the 6 recordings of `ana_20260915_000.txt` with ALS. `FLOW.png` must match `output/test5/optical_flow/*_FLOW_PREMASKED.png`. MED/CAT/area results must be unchanged vs. before Phase 3.

---

## Phase 4: Flow metrics ("further analysis")

Recommended per-pair metrics, computed only inside the pre-mask:

| Metric | Meaning | Supports |
|---|---|---|
| `mean_speed_um_s` | Magnitude, µm/frame × fps | Is anything moving at all? |
| `radial_outflow_um_s` | Mean of the flow projected on the unit vector from the spike-frame centroid (+ out / − in) | Spreading vs. retreating |
| `divergence_mean` | Mean ∂u/∂x + ∂v/∂y | The bilateral divergent pattern at spike−1→spike |
| `coherence` | Resultant length of unit vectors, 0–1 | Uniform drift vs. chaotic |
| `n_px` | Mask size | Reliability of the numbers |

- Stored in a new `flow_pairs` table in `results.db` (one row per recording × pair). Summarized in the ana-list stats block.
- The Session 58 rose map is **not** revived here. Radial outflow + divergence cover the same question without the bipolar blind spot.
- **Open for after seeing numbers:** whether to add a shuffled/rotated-mask null control.

---

## Phase F (inside each phase): Neatness pattern

Applied to `ach_domain_analysis.py`, `classes/spike_reliability.py`, `classes/region_analyzer.py`, `classes/abf_clip.py` (light), and all new files:

- Module docstring = a 4–6 line step list (like `sp_ach_zones.py` lines 1–13).
- Banner blocks: `# ===== CONFIG`, `# ===== STEP 1 -- ...`, sub-blocks `# --- 1a. ...`.
- `run()` in `ach_domain_analysis.py` split into `analyze_entry()`, with visible steps: **1 Clip → 2 Reliability → 3 Median → 4 Categorize → 5 Region + Flow → 6 Export**.
- Docstrings trimmed to 1–3 lines; history-style commentary ("used to…", "moved here because…") removed.
- Each refactor is a separate step **before** that phase's feature change. It is verified as behavior-identical by re-running `ana_20260915_000.txt` and diffing `results.db` rows plus MED/CAT TIFFs.

---

## Critical files

- New: `classes/spontaneous_zones.py`, `functions/zone_kernels.py`, `functions/ap_threshold.py`, `functions/hotspot_flow.py`, `spontaneous_analysis.py`
- Modified: `ach_domain_analysis.py`, `classes/region_analyzer.py`, `classes/spike_reliability.py`, `classes/results_exporter.py`, `functions/plot_results.py`, `functions/database_ops.py`, `classes/__init__.py`, `functions/__init__.py` (lazy registry), `controllers/ctrl_align_spike.py`
- Reused: `functions/fit_bg_hist.py` (GPU histogram), `functions/check_cuda.py` (GPU availability), `list_parser`, `lookup_rec_from_db`, `PIXEL_SCALE`, `ResultsExporter.build_export_stem`
- Retired after merge: `prototype_reliability_group_analysis.py`, `prototype_flow_analysis.py`

## Verification (every phase)

1. `ruff check <touched files>` is clean.
2. CLI run: `.venv/Scripts/python.exe ach_domain_analysis.py --ana_list data/ana_20260915_000.txt` (Phases 0, 2–4) and `.venv/Scripts/python.exe spontaneous_analysis.py --list <10X list>` (Phase 1).
3. Compare outputs against the reference outputs named in each phase. Full export paths are stated after each run.
4. GUI smoke test of Align Spike (the pipeline still runs from the GUI).
5. Optional later: a GUI entry for the spontaneous analysis (a new button or tab). Not in scope unless requested.
