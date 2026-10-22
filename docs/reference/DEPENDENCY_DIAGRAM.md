# DEPENDENCY_DIAGRAM

Who calls what, per pipeline (local imports only). **Updated**: 2026-10-22, rewritten from the imports in the code
(the June version showed `im_dynamics.py` / `batch_process.py`, now in `archive/`). Update together with
`PROJECT_SUMMARY.md` whenever a `.py` file is added, removed, renamed or moved.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                  ACID                                        ║
║          Analyzer for Cholinergic Influence Domain (PG_005)                  ║
║                       PROJECT DEPENDENCY DIAGRAM                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

DATA FLOW
━━━━━━━━━
raw_tiffs/*.tif ──► img_proc.py ──► *_BIEXP_GAUSS.tif ──► als_correct.py ──► *_BIEXP_ALS.tif
                                                                                  │
               ┌──────────────────────────────┬───────────────────────────────────┼───────────────────────┐
               ▼                              ▼                                   ▼                       ▼
   ach_domain_analysis.py          permutation_test.py ──► removal_suggestion.csv ──► SRE_analysis.py   flash_size_analysis.py
   (+ raw_abfs/*.abf)              (Saion)                  (th_small_obj)            │                 (40X / 60X)
               │                                                                      │
               ▼                                                                      ▼
   results.db + median/ categorized/                                     results/spontaneous/
   spatial/ flow/ reliability/ spikes/                                   (TRACE_CORR zones, maps, clustering/)
               │                                                                      │
               └──────────────────────────► area_comparison.py ◄──────────────────────┘

List files (data/pick_*, proc_*, ana_*.txt) + data/rec_data.db + data/exp_info.db
  └─► classes/recording_list.RecordingList ── read by every pipeline


┌─────────────────────────────────────────────────────────────────────────────┐
│                         GUI (main.py)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

main.py
  ├─► views:       ViewMain, ViewDorQuery, ViewDataSelector, ViewImgProc, ViewAlsCorrect, ViewAlignSpike, ViewStBoundary
  ├─► controllers: CtrlMain (pop-out buttons), CtrlDorQuery, CtrlDataSelector, CtrlImgProc, CtrlAlsCorrect,
  │                CtrlAlignSpike, CtrlStBoundary
  └─► utils:       APP_STATUS_MESSAGE, UISizes

  CtrlDorQuery.dor_changed ──► CtrlDataSelector.on_dor_changed
  CtrlDataSelector.pick_confirmed ──► CtrlImgProc.load_pick_list (+ opens the Image Processing window)

  CtrlImgProc     ─► BackgroundWorker ─► img_proc.run            (deferred import)
                  ─► functions.file_status (als_exists, gauss_exists, raw_tiff_ready, build_*_index), check_cuda
  CtrlAlsCorrect  ─► BackgroundWorker ─► als_correct.run         (deferred import)
                  ─► img_proc.parse_proc_list, update_proc_list_gauss_exists
  CtrlAlignSpike  ─► BackgroundWorker ─► ach_domain_analysis.run (deferred import)
                  ─► functions.file_status (abf_ready, als_ready, gauss_ready, build_*_index)
  CtrlStBoundary  ─► BackgroundWorker, functions.st_boundary     → data/bd_*.json, data/st_bd_draft.json
  CtrlDataSelector ─► DialogPickList;  views ─► CheckableDropdown, UISizes


┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING                                       │
└─────────────────────────────────────────────────────────────────────────────┘

img_proc.py
  ├─► classes.RecordingList
  └─► functions: sample_tau (tau_estimate), biexp_detrend (detrend), fit_zscore_background + img_zscore_convert
                 (fit_hist), gaussian_blur_run (gaussian_blur), check_cuda, get_memory_usage

als_correct.py
  ├─► classes.RecordingList
  ├─► functions: als_run (als), check_cuda, get_memory_usage
  └─► img_proc.update_proc_list_gauss_exists

  every CPU/GPU module: @jit / @cuda.jit kernels + a dispatch function (check_cuda() first, before numba is imported)


┌─────────────────────────────────────────────────────────────────────────────┐
│                         SPIKE-ALIGNED (ach_domain_analysis.py)              │
└─────────────────────────────────────────────────────────────────────────────┘

ach_domain_analysis.py
  ├─► classes.RecordingList
  ├─► classes.AbfClip ─────────────────► functions.plot_results (plot_spike_detection_summary)
  ├─► classes.SpikeReliabilityChecker ─► classes.region_analyzer, classes.results_exporter,
  │                                       classes.spatial_categorization, functions.plot_results
  ├─► classes.SpatialCategorizer ──────► functions.cluster_kernels
  ├─► classes.RegionAnalyzer ──────────► classes.spatial_categorization (CATEGORY_BRIGHT),
  │                                       functions.cluster_kernels, functions.flash_flow ─► functions.tvl1_flow,
  │                                       functions.flow_pattern
  ├─► classes.ResultsExporter                                  → results.db, median/, categorized/, figures
  └─► functions: load_img_segs, spike_centered_median (spike_alignment), lookup_rec_from_db +
                 compute_region_stats + compute_flow_pattern_stats + count_unique_cells +
                 get_cell_recording_status (database_ops), load_st_bd + bd_export_path + outline_mask +
                 dv_ml_direction (st_boundary), plot_spatiotemporal_summary + plot_flow_panels +
                 plot_flow_streamlines (plot_results), write_stats_xlsx + write_cell_summary_xlsx (xlsx_writer)

functions.plot_results ─► classes.region_analyzer, classes.spatial_categorization, functions.ap_threshold,
                          functions.flow_pattern


┌─────────────────────────────────────────────────────────────────────────────┐
│                         SPONTANEOUS (permutation_test.py, SRE_analysis.py)  │
└─────────────────────────────────────────────────────────────────────────────┘

permutation_test.py
  ├─► functions.check_cuda (first)
  ├─► classes.RecordingList
  ├─► functions: biexp_detrend, sample_tau, fit_zscore_background, fit_flash_background (fit_hist),
  │              flash_mask (_threshold_open_close, mask_flash_sizes), pixel_lags + sur_flash_masks (permutation)
  └─► img_proc.SIGMA
  functions.permutation ─► functions.als, functions.flash_mask, functions.gaussian_blur (kernels)

SRE_analysis.py
  ├─► classes.RecordingList (select, set_removal_suggestion, set_connect_radius)
  ├─► classes.SREAnalyzer (+ TREE_UNITS, cut_h_name, timed)
  │     ├─► functions.fit_hist (fit_flash_background, CROSSOVER_RATIO)       detect
  │     ├─► functions.flash_mask                                             detect
  │     ├─► functions.footprint_traces ─► functions.flash_mask (f16_lut)     group_trace
  │     ├─► classes.HierarchicalClustering                                   group_trace (/ group_z_peak, not run)
  │     └─► classes.region_analyzer (PIXEL_SCALE)
  └─► functions: check_cuda, img_zscore_convert (fit_hist), load_st_bd + bd_export_path + direction_labels +
                 outline_mask (st_boundary), plot_cluster_dendrogram + plot_silhouette_sweep +
                 plot_cluster_scatter (+ correlation_map_xy) + plot_zone_overview + frame_zone_figures +
                 zone_colors (plot_results)

flash_size_analysis.py
  ├─► classes.RecordingList, classes.SREAnalyzer (detect only)
  └─► functions: check_cuda, CROSSOVER_RATIO (fit_hist), flash_table

area_comparison.py
  ├─► classes.RegionAnalyzer (PIXEL_SCALE)
  └─► functions: write_stats_xlsx, database_ops._read_experiments
  reads results.db + results/spontaneous/{stem}_ZONES.xlsx (old layout)


┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARED                                              │
└─────────────────────────────────────────────────────────────────────────────┘

classes.RecordingList ─► functions.filelist_parser, functions.database_ops (lookup_rec_from_db)
functions.database_ops ─► utils.ColumnSorter
functions/__init__.py   lazy imports (_LAZY_IMPORTS + __getattr__); classes/__init__.py the same
utils/params.py         paths (BASE_DIR, MODELS_DIR = data/, RESULTS_DIR, RAW_/PROC_TIFFS_DIR, ...), UISizes

Prototypes (not pipelines)
  prototype_full_als_waves.py ─► classes.wave_profiles, functions.plot_results
  prototype_med_kymographs.py ─► functions.plot_results
  prototype_peak_tracking.py

docs/paper/**/scripts/*.py  paper figures; several import names that no longer exist
                            (SREAnalyzer.run z_peak flow, spatiotemporally_connect_flashes, NR zones)

External: numpy, scipy, numba (+ CUDA), tifffile, polars, pandas, matplotlib, scikit-image, pyabf, PySide6, rich,
          openpyxl
```
