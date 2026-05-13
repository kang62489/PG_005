# DEPENDENCY_DIAGRAM

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                  ACID                                        ║
║          Analyzer for Cholinergic Influence Domain (PG_005)                  ║
║                       PROJECT DEPENDENCY DIAGRAM                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW 1: IMAGE PREPROCESSING                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────┐
│  img_proc.py                         │  ← Main Script (CLI: --brief _checked.txt)
│  ──────────────────────────────────  │
│  - Parse proc_brief_*_checked.txt    │
│  - Route by MODE: MOV/BIEXP/BOTH     │
│  - sample_tau → biexp_detrend        │
│  - mov_detrend → gaussian_blur_run   │
│  - Save results                      │
└───────────┬──────────────────────────┘
            │
            ├──────────────────────────┐
            │                          │
            ▼                          ▼
┌───────────────────────┐  ┌──────────────────────────┐
│  Hardware Checks      │  │  Image Processing        │
│  ───────────────      │  │  ────────────────        │
│  functions/           │  │  functions/              │
│  ├─ check_cuda()      │  │  ├─ mov_detrend()        │
│  ├─ test_cuda()       │  │  ├─ biexp_detrend()      │
│  └─ get_memory_usage()│  │  ├─ sample_tau()         │
└───────────────────────┘  │  ├─ align_to_min()       │
                            │  └─ gaussian_blur_run()  │
                            └───────────┬──────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
             ┌─────────────────────┐      ┌─────────────────────┐
             │  GPU (@cuda.jit)    │      │  CPU (@jit)         │
             │  detrend.py:        │      │  detrend.py:        │
             │  ├─ _gpu_mov        │      │  ├─ _cpu_mov        │
             │  └─ _gpu_biexp      │      │  └─ _cpu_biexp      │
             │  gaussian_blur.py:  │      │  gaussian_blur.py:  │
             │  └─ _gpu_gaussian   │      │  └─ _cpu_gaussian   │
             │     _blur           │      │     _blur           │
             └─────────────────────┘      └─────────────────────┘

Output: *_MOV_CAL.tif, *_MOV_GAUSS.tif, *_BIEXP_CAL.tif, *_BIEXP_GAUSS.tif


┌─────────────────────────────────────────────────────────────────────────────┐
│                     WORKFLOW 2: SPIKE-ALIGNED ANALYSIS                      │
│                           (Interactive - im_dynamics.py)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────┐
│  im_dynamics.py                │  ← Main Script
│  ────────────────────────────  │
│  - Load processed TIFF + ABF   │
│  - Detect spikes (AbfClip)     │
│  - Extract image segments      │
│  - Z-score normalization       │
│  - Spike-aligned median        │
│  - Spatial categorization      │
│  - Region analysis             │
│  - Export results (SQLite+TIFF) │
│  - Visualization (Qt)          │
└──────────────┬─────────────────┘
               │
   ┌───────────┼───────────┬───────────────────┐
   │           │           │                   │
   ▼           ▼           ▼                   ▼
┌────────────┐ ┌──────────┐ ┌─────────────────┐ ┌─────────────────────┐
│  classes/  │ │ Spike-   │ │ Z-Score         │ │ Spatial             │
│ PlotResults│ │ Triggered│ │ Normalization   │ │ Categorization      │
│ ─────────  │ │ Average  │ │ ─────────────── │ │ ─────────────────── │
│ Interactive│ │ ──────── │ │ functions/      │ │ functions/          │
│ Qt viewer  │ │ functions│ │ imaging_        │ │ spatial_            │
│            │ │ spike_   │ │ segments_       │ │ categorization.py   │
│            │ │ centered_│ │ zscore_         │ │                     │
│            │ │ processes│ │ normalization.py│ │ SpatialCategorizer  │
└────────────┘ │ .py      │ └─────────────────┘ │ ├─ 3 methods        │
               │          │                     │ ├─ 4 threshold      │
               │ Functions│                     │ │   methods         │
               │ ├─ mean  │                     │ ├─ fit()            │
               │ └─ median│                     │ ├─ plot()           │
               └──────────┘                     │ └─ show()           │
                                                └─────────────────────┘

Output: Figures + Statistics for ACh release analysis


┌─────────────────────────────────────────────────────────────────────────────┐
│                     WORKFLOW 3: BATCH PROCESSING                            │
│                     (Automated - batch_process.py / test_batch.py)          │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────┐
│  batch_process.py              │  ← Main Batch Processor (ALL pairs)
│  test_batch.py                 │  ← Test Batch Processor (specific date)
│  ────────────────────────────  │
│  - Read Excel metadata         │
│  - Filter by date (test only)  │
│  - Check existing results      │
│  - Preprocess images           │
│  - Analyze all pairs           │
│  - Export to SQLite + files    │
│  - Generate PNG plots          │
│  - Performance timing          │
└──────────────┬─────────────────┘
               │
   ┌───────────┼───────────┬───────────────────┐
   │           │           │                   │
   ▼           ▼           ▼                   ▼
┌────────────┐ ┌──────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ functions/ │ │ Workflow │ │ Same Analysis   │ │ ResultsExporter     │
│ xlsx_reader│ │ Control  │ │ Pipeline as     │ │ ─────────────────── │
│ ─────────  │ │ ──────── │ │ im_dynamics.py  │ │ SQLite + TIFF + PNG │
│ Read Excel │ │ - Skip   │ │ ─────────────── │ │                     │
│ Get SLICE, │ │   logic  │ │ - AbfClip       │ │ New Columns:        │
│ AT, pairs  │ │ - Timing │ │ - Z-score norm  │ │ - SLICE, AT         │
│            │ │ - Error  │ │ - Categorizer   │ │ - centroid_x/y      │
│            │ │   handling│ │ - RegionAnalyzer│ │ - x/y_span_*        │
└────────────┘ └──────────┘ └─────────────────┘ └─────────────────────┘

Input:  rec_summary/REC_*.xlsx (Excel metadata with PICK column)
Output: results/results.db + results/files/ (same as im_dynamics.py)

Timing:  ⏱️  categorization: 0.234s | region analysis: 0.156s | total: 0.390s


┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLASSES MODULE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

classes/
├── __init__.py
├── plot_results.py
│    ├── MplCanvas         ← Matplotlib canvas for Qt
│    ├── PlotResults       ← Interactive spike detection viewer
│    ├── PlotSegs          ← Segment viewer
│    ├── PlotSpatialDist   ← Spatial categorization overview
│    └── PlotRegion        ← Detailed region viewer with contours
│
├── abf_clip.py
│    └── AbfClip           ← ABF data clipping + spike detection
│
├── spatial_categorization.py  ← STEP 1: Pixel Classification
│    └── SpatialCategorizer
│         ├── INPUT:  Image segments (float32 intensity values)
│         ├── OUTPUT: Categorized frames (uint8: 0/1/2)
│         ├── METHODS: connected, watershed, morphological
│         ├── THRESHOLD_METHODS: manual, multiotsu, li_double, otsu_double
│         ├── fit()            → Categorize image segment
│         ├── get_results()    → Return categorized data
│         └── get_export_data()→ categorized_frames + threshold_method
│                │
│                └─────────────┐
│                              │
├── region_analyzer.py  ← STEP 2: Region Measurement
│    └── RegionAnalyzer       ▼
│         ├── INPUT:  Categorized frames from SpatialCategorizer (0/1/2)
│         ├── OUTPUT: Region properties (area, centroid, contours, bbox)
│         ├── fit()               → Analyze categorized frames
│         ├── get_results()       → Per-frame region data + contours
│         ├── get_summary()       → Aggregate statistics
│         ├── get_frame_results() → Single frame results
│         ├── get_export_data()   → region_summary + region_data
│         └── Supports 10X, 40X, 60X pixel scaling (um² conversion)
│
└── results_exporter.py
     └── ResultsExporter
          ├── export_all()     → Save TIFF (zscore+categorized), update SQLite
          ├── export_figure()  → Save window screenshot
          └── Output: results/results.db + results/files/{prefix}_*.tif/png


┌─────────────────────────────────────────────────────────────────────────────┐
│                             FUNCTIONS MODULE                                │
└─────────────────────────────────────────────────────────────────────────────┘

functions/
│
├── __init__.py  ← Exports all functions with conditional CUDA imports
├── xlsx_reader.py  ← Read experiment metadata from REC_*.xlsx
│    └── get_picked_pairs() → Returns list of {exp_date, img_serial, abf_serial, objective, SLICE, AT}
│
├── [Hardware/CUDA]
│   ├── check_cuda.py          → Verify CUDA availability
│   ├── test_cuda.py           → Test CUDA with sample kernel
│   └── get_memory_use.py      → Memory usage monitoring
│
├── [Preprocessing — unified CPU+GPU]
│   ├── detrend.py             → MOV (@jit _cpu_mov, @cuda.jit _gpu_mov)
│   │                             BIEXP (@jit _cpu_biexp, @cuda.jit _gpu_biexp)
│   │                             + align_to_min()
│   ├── gaussian_blur.py       → Separable 2D Gaussian (@jit CPU + @cuda.jit GPU)
│   └── tau_estimate.py        → sample_tau() — scipy curve_fit on 500 random pixels
│
└── [Analysis]
    ├── kmeans.py              → K-means clustering suite (archived)
    │    ├── prepare_frame_for_kmeans()
    │    ├── apply_kmeans_to_frame()
    │    ├── calculate_cluster_areas()
    │    ├── visualize_clustering_results()
    │    ├── process_segment_kmeans()
    │    ├── process_segment_kmeans_concatenated()
    │    ├── concatenate_frames_horizontally()
    │    └── split_concatenated_result()
    │
    ├── spike_centered_processes.py  → Spike-aligned processing
    │    ├── spike_centered_median()  (robust to outliers)
    │    └── spike_centered_avg()     (mean, sensitive to outliers)
    │
    └── imaging_segments_zscore_normalization.py
         └── Z-score normalization using pre-spike baseline


┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                                 │
└─────────────────────────────────────────────────────────────────────────────┘

[RAW DATA]
│
├─→ Raw TIFF Stack (raw_images/*.tif)
│
├─→ ABF File (raw_abfs/*.abf) - Electrophysiology data
│
└─→ Recording summaries (rec_summary/*.xlsx) - Experiment metadata
    │
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Image Preprocessing (im_preprocess.py)                              │
└─────────────────────────────────────────────────────────────────────────────┘
│
├─→ Load: imageio.volread() → uint16 array (n_frames, height, width)
│
├─→ Detrend: Remove baseline drift (moving average subtraction)
│
├─→ Gaussian Blur (σ=6): Extract spatially coherent ACh signals
│    • Noise (spatially uncorrelated) → cancels out via weighted averaging
│    • Signal (spatially coherent) → preserved across neighboring pixels
│    • Selection: noise_corr_length < σ < signal_corr_length
│    •   Example@60X: 0.4 µm < 1.3 µm < 10 µm ✓
│
└─→ Save: *_Cal.tif, *_Gauss.tif
    │
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Spike-Aligned Analysis (im_dynamics.py)                             │
└─────────────────────────────────────────────────────────────────────────────┘
│
├─→ Load processed TIFF + ABF
│
├─→ Detect Spikes: scipy.signal.find_peaks()
│
├─→ Segment Images: Extract frames around each spike
│
├─→ Z-Score Normalization: (frame - baseline_mean) / baseline_std
│    └─→ Baseline = all frames before spike
│
├─→ Spike-Aligned Median: Align segments by center, compute median
│    └─→ Robust to outliers (random bright spots removed)
│
├─→ STEP 1: Spatial Categorization (SpatialCategorizer)
│    │   Purpose: Classify pixels into categories (0/1/2)
│    ├─→ Input:  Image segments (float32 z-score values)
│    ├─→ Process: Automatic thresholding (li_double, otsu_double, etc.)
│    │            + Spatial connectivity (connected/watershed/morphological)
│    │            + Min region size filtering
│    └─→ Output: Categorized frames (uint8: 0=bg, 1=dim, 2=bright)
│                │
│                ▼
├─→ STEP 2: Region Analysis (RegionAnalyzer)
│    │   Purpose: Extract quantitative measurements from categorized regions
│    ├─→ Input:  Categorized frames from SpatialCategorizer (0/1/2)
│    ├─→ Process: Connected component labeling (skimage.measure.label)
│    │            + Region property calculation (skimage.measure.regionprops)
│    │            + Contour extraction (skimage.measure.find_contours)
│    │            + Pixel scaling (pixels → µm²)
│    └─→ Output: Region properties (area, centroid, bbox, contours)
│                 + Summary statistics (counts, means, stds)
│
├─→ Export Results: Save to disk (ResultsExporter)
│    │
│    ├─→ collect export data using get_export_data():
│    │    • abf_clip.get_export_data() → experiment IDs + ABF segments
│    │    • categorizer.get_export_data() → categorized frames + threshold method
│    │    • analyzer.get_export_data() → region analysis + summary stats
│    │
│    ├─→ exporter.export_all() saves:
│    │    • SQLite: metadata + summaries (results/results.db)
│    │    • TIFF: {prefix}_zscore.tif (float32), {prefix}_categorized.tif (uint8)
│    │
│    └─→ exporter.export_figure() saves:
│         • PNG: {prefix}_spatial_plot.png, {prefix}_region_plot.png
│
└─→ Visualization: Interactive Qt windows
    ├─→ PlotPeaks: View spike detection on voltage trace
    ├─→ PlotSegs: Browse individual spike segments
    ├─→ PlotSpatialDist: Overview of categorization with overlay traces
    └─→ PlotRegion: Frame-by-frame detail with region contours

[OUTPUTS]
│
├─→ results/results.db           (SQLite database - experiment metadata)
├─→ results/files/               (flat directory, all experiments)
│    ├─→ {prefix}_zscore.tif        (spike-centered median, float32)
│    ├─→ {prefix}_categorized.tif   (0=bg, 1=dim, 2=bright, uint8 — ImageJ overlay)
│    ├─→ {prefix}_spatial_plot.png  (PlotSpatialDist screenshot)
│    └─→ {prefix}_region_plot.png   (PlotRegion screenshot)
├─→ processed_images/            (*_Cal.tif, *_Gauss.tif from preprocessing)
└─→ Interactive figures (PlotPeaks, PlotSegs, PlotSpatialDist, PlotRegion)


┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPORT RELATIONSHIPS                                │
└─────────────────────────────────────────────────────────────────────────────┘

External Dependencies:
━━━━━━━━━━━━━━━━━━━━━━
• numpy, scipy          → Array operations, signal processing
• imageio               → TIFF I/O
• matplotlib            → Plotting
• polars                → Data tables (replaced pandas 2026-04)
• scikit-image          → Image processing, thresholding, morphology
• scikit-learn          → K-means algorithm
• numba, numba.cuda     → JIT compilation, GPU kernels
• PySide6               → Qt GUI for interactive plots
• pyabf                 → ABF file reader
• rich                  → Console formatting, progress bars

Internal Dependencies:
━━━━━━━━━━━━━━━━━━━━━━
img_proc.py
  └─→ functions (check_cuda, get_memory_usage,
                 mov_detrend, biexp_detrend, align_to_min,
                 sample_tau, gaussian_blur_run)

im_dynamics.py
  ├─→ classes
  │    ├─→ AbfClip
  │    ├─→ PlotPeaks, PlotSegs, PlotSpatialDist, PlotRegion
  │    ├─→ SpatialCategorizer
  │    ├─→ RegionAnalyzer
  │    └─→ ResultsExporter
  └─→ functions
       ├─→ spike_centered_processes (spike_centered_median)
       └─→ imaging_segments_zscore_normalization (img_seg_zscore_norm)


┌─────────────────────────────────────────────────────────────────────────────┐
│                    KEY ALGORITHMS                                           │
└─────────────────────────────────────────────────────────────────────────────┘

1. SPIKE-ALIGNED MEDIAN
   ─────────────────────
   Purpose: Find consistent signal, filter outliers

   Example (one pixel across 5 segments):
     Values: [2.0, 2.1, 15.0, 1.9, 2.0]
     Mean:   4.6  ← contaminated by outlier
     Median: 2.0  ← robust, true signal

2. CENTER ALIGNMENT
   ─────────────────
   Segments of different lengths aligned by spike frame:

     target_center = target_frames // 2
     seg_center = n_frames // 2
     start = target_center - seg_center

3. TWO-PASS THRESHOLDING (li_double, otsu_double)
   ────────────────────────────────────────────────
   Pass 1: Li/Otsu on all pixels → separates background
   Pass 2: Li/Otsu on signal pixels → separates dim/bright


┌─────────────────────────────────────────────────────────────────────────────┐
│              ARCHIVED SCRIPTS (archive/ — reference only)                   │
└─────────────────────────────────────────────────────────────────────────────┘

archive/run_als_1d.py
  └─→ scipy.sparse / scipy.sparse.linalg → ALS baseline on 1-D Excel trace

archive/run_biexp_detrend.py
  └─→ Original standalone biexp script (superseded by img_proc.py + functions/detrend.py)

archive/im_preprocess.py
  └─→ Original MOV-only pipeline (superseded by img_proc.py)

archive/cpu_binning.py, archive/kmeans.py
  └─→ Unused analysis helpers


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated: 2026-01-15
Updated: 2026-05-14 (refactored preprocessing pipeline: unified detrend.py,
         gaussian_blur.py, tau_estimate.py; img_proc.py replaces im_preprocess.py;
         old CPU/GPU split files archived)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
