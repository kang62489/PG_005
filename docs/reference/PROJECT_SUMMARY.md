# ACID - Analyzer for Cholinergic Influence Domain

**Project code**: PG_005. **Updated**: 2026-10-22 (rewritten from the code; the June version described `im_dynamics.py` /
`batch_process.py`, now in `archive/`).

Keep this file and `DEPENDENCY_DIAGRAM.md` current: update both whenever a `.py` file is added, removed, renamed or
moved, or a pipeline's flow changes (`CLAUDE.md`, Documentation).

---

## 1. Overview

ACh imaging (GRAB-ACh / GACh3.0, iAChSnFR) of striatal slices with a patched cholinergic interneuron (ChI).

| Part | Question | Entry point |
|---|---|---|
| Preprocessing | clean ΔF/F₀ z-stacks from raw TIFFs | `img_proc.py` → `als_correct.py` |
| Spike-aligned (MED) | what one ChI spike releases | `ach_domain_analysis.py` |
| Spontaneous (SRE) | where ACh is released again and again (recur_zones) | `permutation_test.py` → `SRE_analysis.py` |
| Side analyses | flash sizes per objective, MED vs recur_zone size | `flash_size_analysis.py`, `area_comparison.py` |
| GUI | pick recordings, run preprocessing / spike-aligned analysis, draw striatum boundaries | `main.py` |

Terms (flash, recur_zone, th_small_obj, sur_flash, ...): `docs/glossary.md`.

---

## 2. Pipelines

Every pipeline reads its recordings through `classes/recording_list.RecordingList` (a `proc_*.txt` / `ana_*.txt` list
from `data/`, joined with `data/rec_data.db` / `data/exp_info.db`).

### 2.1 `img_proc.py` — preprocessing (BIEXP)

Raw TIFF → `sample_tau` (bi-exp τ from 500 random pixels) → `biexp_detrend` (per-pixel baseline, ΔF/F₀) →
`fit_zscore_background` + `img_zscore_convert` (stack-wide z-score) → `gaussian_blur_run` (σ = 4 px) →
`proc_tiffs/{stem}_BIEXP_GAUSS.tif` (float16).

### 2.2 `als_correct.py` — slow-fluctuation removal

`*_BIEXP_GAUSS.tif` → `als_run` (per-pixel asymmetric least squares baseline, CPU / CUDA) → subtract →
`proc_tiffs/{stem}_BIEXP_ALS.tif`. The ALS files are the input of every analysis below.

### 2.3 `ach_domain_analysis.py` — spike-aligned analysis

| Step | Code | Output |
|---|---|---|
| Clip | `AbfClip`: spikes in the ABF → one image / Vm segment per spike | `spikes/*_spike_analysis.png` |
| Reliability | `SpikeReliabilityChecker`: does each segment show its own flash? | `reliability/*_RELIABILITY.png`, `*_VM_SUCCESS_FAIL.png` |
| Median | `spike_centered_median` of the detected segments (MED) | `median/*_MED.tif` |
| Categorize | `SpatialCategorizer`: bright / background per frame (baseline mean + 2.0 σ) | `categorized/*_CAT.tif` |
| Region + Flow | `RegionAnalyzer`: clusters, flash area decay, TV-L1 flow, flow pattern (DV / ML from the striatum boundary) | `spatial/*_SPATIAL.png`, `flow/*_FLOW.png`, `*_STREAMLINES.png` |
| Export | `ResultsExporter` → `results.db` (experiments + flow_pairs); stats sheets | `{ana_list}_cells.xlsx` |

### 2.4 `permutation_test.py` — th_small_obj per recording

Is a th_flash bigger than chance? Time-shifted surrogates (per-pixel random lag before blur + ALS) → largest
sur_flash per surrogate → smallest size with p < α = **th_small_obj** →
`results/permutation/{OBJ}/removal_suggestion.csv`. Method: `docs/permutation_test.md`. Runs on Saion
(`run_permutation_test_on_saion.slm`).

### 2.5 `SRE_analysis.py` — spontaneous release events → recur_zones

| Step | Code | Output (`{results_dir}/spontaneous/`) |
|---|---|---|
| Input | 10X recordings with a `*_BIEXP_ALS.tif`; own th_small_obj from `removal_suggestion.csv` | — |
| Detection | `SREAnalyzer.detect`: background threshold (peak + 2 σ), flash mask (`functions/flash_mask`), every 8-connected blob of a frame = one flash (`JOIN_SAME_FRAME = False`) | `footprints/{stem}_FLASH_FOOTPRINTS.npz`, `mask/{stem}_FLASH_MASK.tif` |
| Clustering | `SREAnalyzer.group_trace`: footprint traces (`functions/footprint_traces`) → 1 − r (Pearson) → `HierarchicalClustering` (average linkage, cut at the best mean silhouette) | `clustering/{stem}_TRACE_CORR_DENDROGRAM.png`, `{stem}_SILHOUETTE.png`, `{stem}_CLUSTERS.png` (z peaks + correlation map) |
| Zones | `SREAnalyzer.map("trace_corr")`: each cluster = one recur_zone (union of footprints, event stats) | `{stem}_TRACE_CORR_ZONES.xlsx`, `footprints/{stem}_TRACE_CORR_ZONES.npz`, `{stem}_TRACE_CORR_ZONE_MAPS.tif` |
| Characterization | striatum outline + axis labels from `bd_*.json` (coverage commented out) | on the zone maps |

The z_peak tree (`group_z_peak`) is kept in the class but not run (`CLUSTER_METHODS`). Method and decisions:
`docs/hierarchical_clustering.md` (§12 for the current setup).

### 2.6 Side analyses

- `flash_size_analysis.py`: `SREAnalyzer.detect` only, for 40X / 60X → `results/flash_size/{stem}.csv` (one row per
  flash, `functions/flash_table`).
- `area_comparison.py`: MED flash area (results.db) vs spontaneous recur_zone sizes of the same field →
  `area_comparison.xlsx`. Reads the old `{stem}_ZONES.xlsx` layout of `results/spontaneous/`.
- `prototype_*.py`: exploratory (full-stack waves, MED kymographs, peak tracking); not part of the pipelines.

---

## 3. File structure

```
PG_005/
├── main.py                     GUI entry point (ACID)
├── img_proc.py                 preprocessing: biexp detrend + z-score + Gaussian blur -> *_BIEXP_GAUSS.tif
├── als_correct.py              ALS slow-fluctuation removal -> *_BIEXP_ALS.tif
├── ach_domain_analysis.py      spike-aligned analysis (MED, CAT, regions, flow) -> results.db + figures
├── permutation_test.py         surrogate test -> th_small_obj per recording
├── SRE_analysis.py             spontaneous flashes -> trace_corr recur_zones
├── flash_size_analysis.py      flash sizes per recording (40X / 60X)
├── area_comparison.py          MED flash vs recur_zone sizes
├── prototype_*.py              exploratory scripts (3)
├── run_*_on_saion.slm / run_ach_domain_analysis_on_deigo.slm   HPC jobs (OIST Saion / Deigo)
│
├── classes/
│   ├── recording_list.py       RecordingList: list file + databases -> recordings, th_small_obj, connect_radius
│   ├── abf_clip.py             AbfClip: spike detection + per-spike image / Vm segments
│   ├── spike_reliability.py    SpikeReliabilityChecker: per-segment flash check
│   ├── spatial_categorization.py  SpatialCategorizer: bright / background (connected / watershed / morphological)
│   ├── region_analyzer.py      RegionAnalyzer: clusters, area, decay, TV-L1 flow, flow pattern; PIXEL_SCALE
│   ├── results_exporter.py     ResultsExporter: results.db + TIFF / PNG layout (keep its filename rules)
│   ├── sre_analyzer.py         SREAnalyzer: detect -> group_trace (/ group_z_peak) -> map -> save_*
│   ├── hierarchical_clustering.py  HierarchicalClustering: average-linkage tree, silhouette / acceleration cut
│   ├── wave_profiles.py        strip profiles + persistence test (prototype_full_als_waves.py)
│   ├── mpl_canvas.py           MplCanvas (GUI plots)
│   ├── archived_methods.py     archived categorization methods
│   └── GUI helpers             bk_worker (BackgroundWorker), dialog_* , helper_* dropdowns, model_from_dataframe
│
├── functions/                  lazy-imported via functions/__init__.py (_LAZY_IMPORTS)
│   ├── preprocessing           detrend, tau_estimate, gaussian_blur, als, fit_hist (histogram fits, z-score)
│   ├── spontaneous             flash_mask, footprint_traces, permutation, flash_table, zone_groups
│   ├── spike-aligned           spike_alignment, load_img_segs, ap_threshold, cluster_kernels, flash_flow,
│   │                           tvl1_flow, flow_pattern
│   ├── plots                   plot_results (headless Figures for every pipeline)
│   ├── data / files            filelist_parser, database_ops, file_status, xlsx_writer, st_boundary
│   └── hardware                check_cuda (call before anything imports numba), get_memory_use
│
├── controllers/ + views/       GUI (one ctrl_* / view_* pair per window, see section 4)
├── utils/params.py             paths (data/, results/, proc_tiffs/, ...), UISizes, ColumnSorter
├── styles/styles.qss           Qt stylesheet (currently not applied)
├── archive/                    retired scripts (im_dynamics, batch_process, test_batch, kmeans, xlsx_reader, ...)
│
├── data/                       pick_/proc_/ana_*.txt lists, rec_data.db, exp_info.db, bd_*.json (striatum boundaries)
├── raw_tiffs/, raw_abfs/       inputs
├── proc_tiffs/                 *_BIEXP_GAUSS.tif, *_BIEXP_ALS.tif
├── results/                    formal outputs (spontaneous/, permutation/{OBJ}/, flash_size/, median/, ... results.db)
├── output/                     test runs only (output/test_*)
└── docs/                       see section 6
```

---

## 4. GUI (`main.py`)

A main window (`view_main` / `ctrl_main`) with the DOR list and one button per pop-out window:

| Window | Controller / view | Does |
|---|---|---|
| Query by DOR (+ Log View) | `ctrl_dor_query` / `view_dor_query` | experiment info per date of recording |
| Data Selector | `ctrl_data_selector` / `view_data_selector` | pick recordings → pick list (`DialogPickList`) |
| Image Processing | `ctrl_img_proc` / `view_img_proc` | proc list, file status, runs `img_proc.run` on a `BackgroundWorker` |
| ALS Correction | `ctrl_als_correct` / `view_als_correct` | ALS test on an ROI + runs `als_correct.run` |
| Spike-aligned Analysis | `ctrl_align_spike` / `view_align_spike` | runs `ach_domain_analysis.run` |
| Striatum Boundary | `ctrl_st_boundary` / `view_st_boundary` | slice orientation + 10X striatum outline → `bd_*.json` |

Signals: `dor_changed` (DOR query → data selector), `pick_confirmed` (data selector → image processing).

---

## 5. Key methods (short)

- **Gaussian blur σ**: noise correlation length (1–2 px) < σ < signal correlation length (cellular scale); σ = 4 px.
- **Spike-aligned median**: per pixel, median over segments, e.g. [2.0, 2.1, 15.0, 1.9, 2.0] → 2.0 (mean 4.6).
- **Flash threshold** (SRE, permutation): Gaussian fit to the left side of the pixel-value histogram, threshold =
  peak + 2 σ (`functions/fit_hist.CROSSOVER_RATIO`).
- **th_small_obj**: blobs smaller than this are removed from the flash mask; from the permutation test.
- **trace_corr clustering**: 1 − r of footprint traces, average linkage, h at the best mean silhouette
  (`docs/hierarchical_clustering.md`).

| Objective | px / µm | µm / px |
|---|---|---|
| 10X | 0.75 | 1.333 |
| 40X | 3.0 | 0.333 |
| 60X | 4.5 | 0.222 |

---

## 6. Docs

| Path | Content |
|---|---|
| `docs/continue_from_here.md` | session log + TODOs (project-tracker) |
| `docs/glossary.md` | terms |
| `docs/hierarchical_clustering.md`, `docs/permutation_test.md` | method notes |
| `docs/reference/` | this file, `DEPENDENCY_DIAGRAM.md`, `DATABASE_STRUCTURE.md`, notes |
| `docs/knowledgebase/` | concepts, techniques |
| `docs/resolved_problems/` | solved bugs |
| `docs/paper/` | paper drafts (`plain_0X/` with figures / scripts / tables), discussion |
| `PIPELINE_FLOW.md`, `RESULTS_GUIDE.md` (root) | pipeline walk-through; guide to the results in `results/` |
