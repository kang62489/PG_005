     ╔══════════════════════════════════════════════════════════════════════════════╗
     ║                       PROJECT DEPENDENCY DIAGRAM                             ║
     ╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW 1: IMAGE PREPROCESSING                     │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────┐
    │  im_preprocess.py       │  ← Main Script
    │  ─────────────────────  │
    │  - Load raw TIFF        │
    │  - Check CUDA           │
    │  - Process images       │
    │  - Save results         │
    └───────────┬─────────────┘
                │
                ├──────────────────────────┐
                │                          │
                ▼                          ▼
    ┌───────────────────────┐  ┌──────────────────────┐
    │  Hardware Checks      │  │  Image Processing    │
    │  ───────────────      │  │  ────────────────    │
    │  functions/           │  │  functions/          │
    │  ├─ check_cuda()      │  │  ├─ process_on_gpu() │
    │  ├─ test_cuda()       │  │  └─ process_on_cpu() │
    │  └─ get_memory_usage()│  └───────────┬──────────┘
    └───────────────────────┘              │
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
        ┌──────────────────────┐                    ┌──────────────────────┐
        │  GPU Pipeline        │                    │  CPU Pipeline        │
        │  ─────────────       │                    │  ─────────────       │
        │  functions/          │                    │  functions/          │
        │  gpu_process.py      │                    │  cpu_process.py      │
        └───────────┬──────────┘                    └─────────┬────────────┘
                    │                                         │
         ┌──────────┴───────────┐                  ┌──────────┴──────────┐
         ▼                      ▼                  ▼                     ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ gpu_detrend │    │ gpu_gauss   │    │ cpu_detrend │    │ cpu_gauss   │
    │  (CUDA)     │    │  (CUDA)     │    │  (Numba)    │    │  (Numba)    │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

    Output: *_Cal.tif (detrended), *_Gauss.tif (Gaussian filtered)


┌─────────────────────────────────────────────────────────────────────────────┐
│                    WORKFLOW 2: SPIKE-TRIGGERED ANALYSIS                     │
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
    │  - Export results (SQLite+NPZ) │
    │  - Visualization (Qt)          │
    └──────────────┬─────────────────┘
                   │
       ┌───────────┼───────────┬───────────────────┐
       │           │           │                   │
       ▼           ▼           ▼                   ▼
┌────────────┐ ┌────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│  classes/  │ │ Spike-Trig │ │ Z-Score         │ │ Spatial             │
│ PlotResults│ │ Average    │ │ Normalization   │ │ Categorization      │
│ ─────────  │ │ ────────── │ │ ─────────────── │ │ ─────────────────── │
│ Interactive│ │ functions/ │ │ functions/      │ │ functions/          │
│ Qt viewer  │ │ spike_     │ │ imaging_        │ │ spatial_            │
│            │ │ triggered_ │ │ segments_       │ │ categorization.py   │
│            │ │ average.py │ │ zscore_         │ │                     │
│            │ │            │ │ normalization.py│ │ SpatialCategorizer  │
└────────────┘ │            │ └─────────────────┘ │ ├─ 5 methods        │
               │ spike_trig │                     │ ├─ 7 threshold      │
               │ _avg()     │                     │ │   methods          │
               │ ├─ mean    │                     │ ├─ fit()            │
               │ └─ median  │                     │ ├─ plot()           │
               └────────────┘                     │ └─ show()           │
                                                  └─────────────────────┘

    Output: Figures + Statistics for ACh release analysis


┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASSES MODULE                                      │
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
    ├── spatial_categorization.py
    │    └── SpatialCategorizer
    │         ├── METHODS: connected, watershed, morphological
    │         ├── THRESHOLD_METHODS: manual, multiotsu,
    │         │                      li_double, otsu_double
    │         ├── fit()         → Process image segment
    │         ├── plot()        → Create matplotlib figure
    │         ├── show()        → Display figure
    │         └── get_results() → Return categorized data
    │
    ├── region_analyzer.py
    │    └── RegionAnalyzer
    │         ├── fit()              → Analyze categorized frames
    │         ├── get_results()      → Per-frame region data + contours
    │         ├── get_summary()      → Aggregate statistics
    │         ├── get_frame_results()→ Single frame results
    │         └── Supports 10X, 40X, 60X pixel scaling
    │
    └── results_exporter.py
         └── ResultsExporter
              ├── export_all()     → Save TIFF, NPZ, update SQLite
              ├── export_figure()  → Save window screenshot
              └── Output: results/results.db + data/{date}/abf{}_img{}/


┌─────────────────────────────────────────────────────────────────────────────┐
│                        FUNCTIONS MODULE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    functions/
    │
    ├── __init__.py  ← Exports all functions with conditional CUDA imports
    │
    ├── [Hardware/CUDA]
    │   ├── check_cuda.py          → Verify CUDA availability
    │   ├── test_cuda.py           → Test CUDA with sample kernel
    │   └── get_memory_use.py      → Memory usage monitoring
    │
    ├── [CPU Processing]
    │   ├── cpu_detrend.py         → @numba.jit detrending
    │   ├── cpu_gauss.py           → @numba.jit Gaussian blur
    │   ├── cpu_binning.py         → Spatial binning
    │   └── cpu_process.py         → Orchestrator (warm up → detrend → blur)
    │
    ├── [GPU Processing]
    │   ├── gpu_detrend.py         → @cuda.jit detrending kernel
    │   ├── gpu_gauss.py           → CUDA Gaussian blur
    │   └── gpu_process.py         → Orchestrator (transfer → process → copy back)
    │
    └── [Analysis]
        ├── kmeans.py              → K-means clustering suite
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
│                      DATA FLOW DIAGRAM                                      │
└─────────────────────────────────────────────────────────────────────────────┘

[RAW DATA]
    │
    ├─→ Raw TIFF Stack (raw_images/*.tif)
    │
    └─→ ABF File (raw_abfs/*.abf) - Electrophysiology data
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
│ STEP 2: Spike-Triggered Analysis (im_dynamics.py)                           │
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
    ├─→ Spatial Categorization: Identify ACh release regions
    │    ├─→ Automatic thresholding (li_double, otsu_double, etc.)
    │    ├─→ Connected component analysis
    │    └─→ Min region size filtering
    │
    ├─→ Region Analysis: Quantify properties (RegionAnalyzer)
    │    ├─→ Area calculation (pixels → µm²)
    │    ├─→ Centroid extraction
    │    ├─→ Contour detection
    │    └─→ Summary statistics
    │
    ├─→ Export Results: Save to disk (ResultsExporter)
    │    ├─→ SQLite: metadata + summaries (results/results.db)
    │    ├─→ TIFF: zscore_stack.tif, categorized_stack.tif
    │    ├─→ NPZ: img_segments.npz, abf_segments.npz
    │    └─→ PNG: region_plot.png (Qt window screenshot)
    │
    └─→ Visualization: Interactive Qt windows
         ├─→ PlotSegs: Browse individual spike segments
         ├─→ PlotSpatialDist: Overview of categorization
         └─→ PlotRegion: Frame-by-frame detail with contours

[OUTPUTS]
    │
    ├─→ results/results.db           (SQLite database)
    ├─→ results/data/{date}/abf{}_img{}/
    │    ├─→ zscore_stack.tif        (spike-centered median)
    │    ├─→ categorized_stack.tif   (0=bg, 1=dim, 2=bright)
    │    ├─→ img_segments.npz        (individual spike segments)
    │    ├─→ abf_segments.npz        (time + voltage traces)
    │    └─→ region_plot.png         (PlotRegion screenshot)
    └─→ Interactive figures (PlotSegs, PlotSpatialDist, PlotRegion)


┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPORT RELATIONSHIPS                                │
└─────────────────────────────────────────────────────────────────────────────┘

External Dependencies:
━━━━━━━━━━━━━━━━━━━━━━
  • numpy, scipy          → Array operations, signal processing
  • imageio               → TIFF I/O
  • matplotlib            → Plotting
  • pandas                → Data tables
  • scikit-image          → Image processing, thresholding, morphology
  • scikit-learn          → K-means algorithm
  • numba, numba.cuda     → JIT compilation, GPU kernels
  • PySide6               → Qt GUI for interactive plots
  • pyabf                 → ABF file reader
  • rich                  → Console formatting, progress bars

Internal Dependencies:
━━━━━━━━━━━━━━━━━━━━━━
  im_preprocess.py
    └─→ functions (check_cuda, test_cuda, get_memory_usage,
                   process_on_gpu, process_on_cpu)

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


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated: 2026-01-15
Updated: 2026-01-20 (Added Gaussian blur explanation, RegionAnalyzer, ResultsExporter)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
