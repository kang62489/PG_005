# PG_005 Project Summary

## Project Overview

This project performs **acetylcholine (ACh) imaging analysis** with two main workflows:
1. **Image Preprocessing**: Raw TIFF stack preprocessing (detrending, Gaussian filtering)
2. **Spike-Triggered Analysis**: Spike-aligned imaging analysis with spatial categorization to identify ACh release patterns

---

## File Structure

```
PG_005/
├── im_preprocess.py                 # Main script: Image preprocessing (detrend + Gaussian)
├── im_dynamics.py                   # Main script: Spike-triggered ACh analysis
│
├── classes/
│   ├── __init__.py
│   ├── plot_results.py              # Interactive Qt viewers (PlotPeaks, PlotSegs, PlotSpatialDist, PlotRegion)
│   ├── abf_clip.py                  # ABF data clipping + spike detection utilities
│   ├── spatial_categorization.py    # SpatialCategorizer class
│   ├── region_analyzer.py           # RegionAnalyzer class (area, centroid, contours)
│   ├── results_exporter.py          # ResultsExporter (SQLite + NPZ + TIFF exporter)
│   └── archived_methods.py          # Archived: dbscan, region_growing
│
├── functions/
│   ├── __init__.py
│   │
│   ├── [Hardware/CUDA]
│   │   ├── check_cuda.py            # Verify CUDA availability
│   │   ├── test_cuda.py             # Test CUDA functionality
│   │   └── get_memory_use.py        # Memory usage monitoring
│   │
│   ├── [CPU Processing]
│   │   ├── cpu_detrend.py           # Numba JIT detrending
│   │   ├── cpu_gauss.py             # Numba JIT Gaussian blur
│   │   ├── cpu_binning.py           # Spatial binning
│   │   └── cpu_process.py           # CPU processing orchestrator
│   │
│   ├── [GPU Processing]
│   │   ├── gpu_detrend.py           # CUDA kernel detrending
│   │   ├── gpu_gauss.py             # CUDA Gaussian blur
│   │   └── gpu_process.py           # GPU processing orchestrator
│   │
│   └── [Analysis]
│       ├── kmeans.py                            # K-means clustering
│       ├── spike_centered_processes.py          # Spike-aligned median/mean
│       └── imaging_segments_zscore_normalization.py  # Z-score normalization
│
├── doc/
│   ├── PROJECT_SUMMARY.md           # This file
│   ├── DEPENDENCY_DIAGRAM.md        # Architecture diagram
│   └── examples/                    # Usage examples
│
├── raw_images/                      # Raw TIFF stacks (input)
├── raw_abfs/                        # ABF files (electrophysiology input)
├── processed_images/                # Preprocessed TIFFs (*_Cal.tif, *_Gauss.tif)
├── results/                         # Analysis results
│   ├── results.db                   # SQLite database
│   └── {exp_date}/                  # Date-organized results
│       └── abf{}_img{}/             # Experiment-specific data
├── rec_summary/                     # Recording summary Excel files
├── logs/                            # Processing logs
└── utils/                           # Utility scripts (currently unused)
```

---

## Workflow 1: Image Preprocessing

**Script**: `im_preprocess.py`

```
Raw TIFF Stack
      ↓
  Load Image (imageio.volread)
      ↓
  [CUDA Detection]
      ↓
  ┌─────┴─────┐
  ↓           ↓
CPU Path   GPU Path
  ↓           ↓
  └─────┬─────┘
      ↓
  Detrend → Gaussian Blur
      ↓
  Save: *_Cal.tif, *_Gauss.tif
```

### Processing Steps

1. **Detrending**: Remove baseline drift using moving average subtraction
2. **Gaussian Filtering**: Spatial smoothing for signal extraction (see below)

### Gaussian Filtering for ACh Signal Extraction

Fluorescence images of acetylcholine (ACh) release exhibit substantial high-frequency noise arising from photon shot noise and camera readout noise, which obscures the underlying ACh release signals. To extract spatially coherent ACh signals, we apply Gaussian filtering.

**Why Gaussian filtering works:**

Gaussian filtering performs a spatially weighted average centered on each pixel, with weights following a 2D Gaussian distribution. This operation exploits the fundamental difference between signal and noise:

- **Noise**: Spatially uncorrelated (random per pixel). When weighted-averaged, random values cancel out.
- **Signal**: Spatially coherent—neighboring pixels detecting the same ACh release event show correlated intensity changes because ACh diffuses from physical release sites, creating spatial concentration gradients.

**Parameter selection (σ):**

The critical parameter σ determines the spatial scale of averaging:
- Features smaller than σ are attenuated
- Features larger than σ are preserved

Selection rule based on **spatial autocorrelation**:
```
noise_correlation_length < σ < signal_correlation_length
```

| Component | Correlation Length | Basis |
|-----------|-------------------|-------|
| Noise | ~1-2 pixels | Camera/shot noise is uncorrelated |
| Signal | >50 pixels | ACh release sites are cellular-scale |

**Example at 60X (0.22 µm/pixel):**
```
σ = 6 pixels = 1.3 µm

Noise correlation:  ~0.2-0.4 µm (1-2 pixels)
Signal correlation: >10 µm (cellular-scale ACh release)

0.4 µm < 1.3 µm < 10 µm  ✓
```

This ensures effective noise suppression without distorting the spatial extent or morphology of ACh release regions.

### Data Transformations

| Stage | Shape | Data Type |
|-------|-------|-----------|
| Load | (n_frames, height, width) | uint16 |
| Processing | (n_frames, height, width) | float32 |
| Save (preprocessed) | (n_frames, height, width) | uint16 (clipped to 0-65535) |
| Save (analysis results) | (n_frames, height, width) | float32 (zscore), uint8 (categorized) |

---

## Workflow 2: Spike-Triggered Analysis

**Script**: `im_dynamics.py`

```
Processed TIFF + ABF (electrophysiology)
              ↓
        Load & Align Data (AbfClip)
              ↓
        Detect Spikes (scipy.signal.find_peaks)
              ↓
        Extract Image Segments Around Spikes
              ↓
        Z-Score Normalization (baseline = pre-spike frames)
              ↓
        Spike-Aligned Median (robust to outliers)
              ↓
        Spatial Categorization (SpatialCategorizer)
              ↓
        Region Analysis (RegionAnalyzer)
              ↓
        Export Results (ResultsExporter)
              ↓
        Output: SQLite DB + TIFF + NPZ + PNG figures
```

### Key Analysis Functions

#### Spike-Centered Processing
- Aligns image segments by spike frame (center alignment)
- Two functions: `spike_centered_median()` and `spike_centered_avg()`
- Median removes outliers → cleaner ACh signal

```python
from functions.spike_centered_processes import spike_centered_median, spike_centered_avg

# Use median for robust averaging (recommended)
avg_segment = spike_centered_median(lst_segments)

# Or use mean (sensitive to outliers)
avg_segment = spike_centered_avg(lst_segments)
```

#### `SpatialCategorizer` - Spatial Analysis
- 3 methods: connected, watershed, morphological
- 4 threshold methods: manual, multiotsu, li_double, otsu_double
- Global thresholding across all frames

```python
from classes.spatial_categorization import SpatialCategorizer

# Use factory methods - each shows only relevant parameters
categorizer = SpatialCategorizer.connected(min_region_size=30)
categorizer = SpatialCategorizer.watershed(min_region_size=20, min_distance=5)
categorizer = SpatialCategorizer.morphological(kernel_size=5)

# Fit the categorizer
categorizer.fit(avg_segment)

# Get results
results = categorizer.get_results()
# Returns: source_frames, categorized_frames, frame_regions, thresholds_used, method, threshold_method

# With manual thresholds
categorizer = SpatialCategorizer.connected(
    threshold_method="manual",
    threshold_dim=0.5,
    threshold_bright=1.5,
)
```

#### `RegionAnalyzer` - Post-Categorization Analysis
- Calculates region properties: area, centroid, bounding box
- Extracts contours for visualization
- Supports pixel-to-micrometer conversion for objectives (10X, 40X, 60X)

```python
from classes.spatial_categorization import SpatialCategorizer
from classes.region_analyzer import RegionAnalyzer

# Step 1: Categorize
categorizer = SpatialCategorizer.morphological()
categorizer.fit(image_segment)

# Step 2: Analyze regions
analyzer = RegionAnalyzer(obj="10X", min_area=20)
analyzer.fit(categorizer.categorized_frames)

# Get results
results = analyzer.get_results()
# results["bright_regions"][frame_idx] = [{area_pixels, area_um2, centroid, bbox, label}, ...]
# results["bright_contours"][frame_idx] = [contour_array, ...]

summary = analyzer.get_summary()
# Returns: n_frames, total_dim_regions, total_bright_regions, area stats

# Draw contours on plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(categorizer.categorized_frames[0], cmap="gray")
for contour in results["bright_contours"][0]:
    ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
plt.show()
```

### Pixel Scaling Factors

| Objective | pixel/um | um/pixel |
|-----------|----------|----------|
| 10X | 0.75 | 1.333 |
| 40X | 3.0 | 0.333 |
| 60X | 4.5 | 0.222 |

---

## Key Algorithms

### Spike-Aligned Median

**Purpose**: Find consistent signal across spike events, filter outliers

```
Pixel (y,x) at frame 5 across N segments:
  Seg1: 2.0, Seg2: 2.1, Seg3: 15.0 (outlier), Seg4: 1.9, Seg5: 2.0

  Mean:   (2.0 + 2.1 + 15.0 + 1.9 + 2.0) / 5 = 4.6  ← contaminated
  Median: [1.9, 2.0, 2.0, 2.1, 15.0] → 2.0           ← robust
```

### Center Alignment Formula

Segments of different lengths aligned by their center (spike frame):

```python
target_center = target_frames // 2
seg_center = n_frames // 2
start = target_center - seg_center
```

### Two-Pass Thresholding (li_double, otsu_double)

1. **Pass 1**: Separate background from signal (Li/Otsu on all pixels)
2. **Pass 2**: Separate dim from bright (Li/Otsu on signal pixels only)

---

## External Dependencies

| Package | Purpose |
|---------|---------|
| numpy, scipy | Array operations, signal processing |
| imageio | TIFF I/O |
| matplotlib | Plotting |
| pandas | Data tables |
| scikit-image | Image processing, thresholding, morphology |
| scikit-learn | K-means clustering |
| numba, numba.cuda | JIT compilation, GPU kernels |
| PySide6 | Qt GUI for interactive plots |
| pyabf | ABF file reader |
| rich | Console formatting, progress bars |

---

## Post-Processing with RegionAnalyzer

After spatial categorization, use `RegionAnalyzer` for region analysis:

```python
from classes.spatial_categorization import SpatialCategorizer
from classes.region_analyzer import RegionAnalyzer

# Categorize
categorizer = SpatialCategorizer.morphological()
categorizer.fit(image_segment)

# Analyze regions with pixel scaling
analyzer = RegionAnalyzer(obj="10X", min_area=50)
analyzer.fit(categorizer.categorized_frames)

# Get per-frame results
frame_result = analyzer.get_frame_results(0)
for region in frame_result["bright_regions"]:
    print(f"Area: {region['area_um2']:.2f} um², Centroid: {region['centroid']}")

# Get summary across all frames
summary = analyzer.get_summary()
print(f"Total bright regions: {summary['total_bright_regions']}")
print(f"Mean bright area: {summary['bright_area_um2_mean']:.2f} um²")
```

### Categorical vs Labeled Images

| Type | Values | Example |
|------|--------|---------|
| Categorical | 0=background, 1=dim, 2=bright | All dim pixels = 1 |
| Labeled | Unique ID per connected region | Two dim regions = 1, 2 |

Use `skimage.measure.label()` to convert categorical → labeled for regionprops.

---

## Quick Reference

### Run Preprocessing
```bash
python im_preprocess.py
```

### Run Spike-Triggered Analysis
```bash
python im_dynamics.py
```

### SpatialCategorizer Factory Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `.connected()` | `min_region_size` | Connected component labeling |
| `.watershed()` | `min_region_size`, `min_distance` | Watershed segmentation |
| `.morphological()` | `kernel_size` | Morphological operations |

All methods also accept: `threshold_method`, `global_threshold`, `threshold_dim`, `threshold_bright`

### RegionAnalyzer Methods

| Method | Returns |
|--------|---------|
| `.fit(categorized_frames)` | self (for chaining) |
| `.get_frame_results(idx)` | dict with dim/bright regions and contours |
| `.get_results()` | dict with all per-frame data |
| `.get_summary()` | dict with aggregated statistics |
| `.pixel_to_um(pixels)` | distance in micrometers |
| `.area_pixel_to_um2(area)` | area in um² |

### Threshold Methods

| Method | Description |
|--------|-------------|
| manual | User-specified thresholds |
| multiotsu | Multi-level Otsu |
| li_double | Two-pass Li thresholding |
| otsu_double | Two-pass Otsu thresholding |

---

## Export Workflow

### Data Export Structure

Analysis results are organized by date and experiment:

```
results/
├── results.db                              # SQLite database with metadata
└── {exp_date}/                             # e.g., "2025_12_15"
    └── abf{abf_serial}_img{img_serial}/    # e.g., "abf0034_img0042"
        ├── zscore_stack.tif                # Spike-centered median (float32)
        ├── categorized_stack.tif           # Categorized frames (uint8: 0=bg, 1=dim, 2=bright)
        ├── img_segments.npz                # Individual z-score segments
        ├── abf_segments.npz                # Time + voltage traces
        ├── spatial_plot.png                # Spatial distribution figure
        └── region_plot.png                 # Region detail figure
```

### Using get_export_data()

Each analysis class provides a `get_export_data()` method for easy export:

```python
from classes import AbfClip, SpatialCategorizer, RegionAnalyzer, ResultsExporter

# Step 1: Load and process data
abf_clip = AbfClip(exp_date="2025_12_15", abf_serial="0034", img_serial="0042")
categorizer = SpatialCategorizer.morphological(threshold_method="otsu_double")
categorizer.fit(spike_centered_median_segment)
analyzer = RegionAnalyzer(obj="10X")
analyzer.fit(categorizer.categorized_frames)

# Step 2: Export all results
exporter = ResultsExporter()
exp_dir = exporter.export_all(
    **abf_clip.get_export_data(),        # Experiment metadata + ABF segments
    **categorizer.get_export_data(),     # Categorized frames + threshold method
    **analyzer.get_export_data(),        # Region analysis + summary statistics
    zscore_stack=med_img_segment_zscore,
    img_segments_zscore=lst_img_segments_zscore,
)

# Step 3: Export figures
exporter.export_figure(exp_dir, plt_spatial.grab(), filename="spatial_plot.png")
exporter.export_figure(exp_dir, plt_region.grab(), filename="region_plot.png")
```

### SQLite Database Schema

The `results.db` file contains experiment metadata:

| Column | Type | Description |
|--------|------|-------------|
| exp_date | TEXT | Experiment date (YYYY_MM_DD) |
| abf_serial | TEXT | ABF file serial number |
| img_serial | TEXT | Image file serial number |
| timestamp | TEXT | Analysis timestamp (ISO format) |
| objective | TEXT | Microscope objective (10X, 40X, 60X) |
| um_per_pixel | REAL | Pixel scale factor |
| threshold_method | TEXT | Threshold method used |
| n_spikes_detected | INTEGER | Total spikes found |
| n_spikes_analyzed | INTEGER | Spikes included in analysis |
| n_frames | INTEGER | Number of frames in segment |
| total_dim_regions | INTEGER | Total dim regions found |
| total_bright_regions | INTEGER | Total bright regions found |
| dim_area_um2_mean | REAL | Mean area of dim regions (µm²) |
| dim_area_um2_std | REAL | Std of dim region areas |
| bright_area_um2_mean | REAL | Mean area of bright regions (µm²) |
| bright_area_um2_std | REAL | Std of bright region areas |
| region_analysis | TEXT | JSON with detailed region data |
| data_dir | TEXT | Relative path to data directory |

---

## Understanding Thresholding Methods

All automatic threshold methods find the best value to separate background from signal.

### Otsu: Maximize Between-Class Variance

```
pixels = [1, 2, 2, 3, 3, 7, 8, 8, 9, 9]

For threshold t=3:
  c0 = [1,2,2,3,3]  (5 pixels, mean=2.2)
  c1 = [7,8,8,9,9]  (5 pixels, mean=8.2)

  w0 = 5/10 = 0.5,  w1 = 5/10 = 0.5

  variance = w0 * w1 * (mu0 - mu1)²
           = 0.5 * 0.5 * (2.2 - 8.2)²
           = 9.0  ← maximum! → best threshold
```

### Li: Minimize Cross-Entropy

```
Cross-entropy = information loss when replacing pixels with class mean
Lower = better fit

For threshold t=3:
  ce = -sum(pixels * log(class_mean))

Pick threshold with MINIMUM cross-entropy
```

### Yen: Maximize Entropic Correlation

```
Finds threshold where both classes have good "spread" (entropy)
Prefers balanced class sizes (50/50 split)
```

---

## Understanding Spatial Categorization Methods

All three methods share the same thresholds (thresh_dim, thresh_bright).
They differ in HOW they use those thresholds.

### Example Image (5x5, z-scored)

```
    0.1  0.2  0.3  0.2  0.1
    0.2  0.8  1.2  0.8  0.2
    0.3  1.2  2.0  1.2  0.3
    0.2  0.8  1.2  0.8  0.2
    0.1  0.2  0.3  0.2  0.1

thresh_dim = 0.5,  thresh_bright = 1.5
```

---

### Method 1: MORPHOLOGICAL

**Order: Threshold pixels → Clean up shapes**

```
Step 1: Apply thresholds directly to each pixel
        pixel > 1.5 → bright (2)
        pixel > 0.5 → dim (1)
        else → background (0)

    0    0    0    0    0
    0    1    1    1    0       (0.8, 1.2, 0.8 > 0.5 → dim)
    0    1    2    1    0       (2.0 > 1.5 → bright)
    0    1    1    1    0
    0    0    0    0    0

Step 2: Cleanup with erosion/dilation
        - erosion: shrink regions (removes small noise)
        - dilation: expand back (fills small holes)

        Tunable parameter: kernel_size (default=3)
        Larger kernel = more aggressive cleanup
```

**Key feature:** Each pixel categorized individually, then shapes cleaned.

---

### Method 2: CONNECTED

**Order: Threshold → Find connected regions → Categorize each region by mean**

```
Step 1: Create binary mask (pixels > thresh_dim)
    0    0    0    0    0
    0    1    1    1    0
    0    1    1    1    0
    0    1    1    1    0
    0    0    0    0    0

Step 2: Find connected regions (scipy.ndimage.label)
    0    0    0    0    0
    0    A    A    A    0       All connected = one region "A"
    0    A    A    A    0
    0    A    A    A    0
    0    0    0    0    0

Step 3: Calculate mean intensity of region A
    pixels = [0.8, 1.2, 0.8, 1.2, 2.0, 1.2, 0.8, 1.2, 0.8]
    mean = 1.1

Step 4: Categorize ENTIRE region by mean
    1.1 > 1.5 (thresh_bright)?  NO
    1.1 > 0.5 (thresh_dim)?     YES → dim (1)

Result:
    0    0    0    0    0
    0    1    1    1    0
    0    1    1    1    0       ← ALL dim (even 2.0 pixel!)
    0    1    1    1    0
    0    0    0    0    0
```

**Key feature:** Whole region gets same category based on mean.
**Limitation:** Bright pixels "eaten" if connected to many dim pixels.

---

### Method 3: WATERSHED

**Order: Threshold → Distance transform → Find peaks → Flood fill → Categorize**

#### Example: Two blobs separated by a gap

```
Original image (z-scores):
col:  0    1    2    3    4    5    6    7    8

      0    0    0    0    0    0    0    0    0
      0  1.5  1.2  0.8  0    0.7  1.1  1.6  0
      0  1.8  1.5  1.0  0    0.9  1.4  1.8  0
      0  1.4  1.1  0.7  0    0.6  1.0  1.5  0
      0    0    0    0    0    0    0    0    0

         [  Blob A  ]       [  Blob B  ]

thresh_dim = 0.5
```

#### Step 1: Create mask (pixels > 0.5)

```
      0    0    0    0    0    0    0    0    0
      0    1    1    1    0    1    1    1    0
      0    1    1    1    0    1    1    1    0
      0    1    1    1    0    1    1    1    0
      0    0    0    0    0    0    0    0    0
```

#### Step 2: Distance transform

For each `1`, count minimum steps to reach nearest `0`.

- Edge pixels (directly touching 0): distance = 1
- Center pixel (2 steps from nearest 0): distance = 2

```
      0    0    0    0    0    0    0    0    0
      0    1    1    1    0    1    1    1    0
      0    1    2    1    0    1    2    1    0
      0    1    1    1    0    1    1    1    0
      0    0    0    0    0    0    0    0    0
              ↑                   ↑
           peak A              peak B
           col 2               col 6
```

#### Step 3: Find peaks and `min_distance` parameter

Peaks = local maxima. Here: peak A (col 2) and peak B (col 6).

**Distance between peaks = 6 - 2 = 4 pixels**

`min_distance` = tunable parameter = "minimum distance required to be separate"

```
min_distance=3:  Is 4 ≥ 3?  YES → Keep BOTH peaks → 2 regions
min_distance=5:  Is 4 ≥ 5?  NO  → Keep ONE peak  → 1 region (merged)
```

The algorithm:
1. Find ALL local maxima (peaks)
2. Filter out peaks that are closer than `min_distance`
3. Remaining peaks become seeds

#### Step 4: Invert distance (multiply by -1)

Watershed fills from LOW to HIGH. We want to start from centers, so invert:

```
Distance:                      -Distance:
0    0    0    0    0          0    0    0    0    0
0    1    1    1    0          0   -1   -1   -1    0
0    1    2    1    0    →     0   -1   -2   -1    0
0    1    1    1    0          0   -1   -1   -1    0
0    0    0    0    0          0    0    0    0    0

Now center (-2) is LOWEST = "valley" where water starts
```

#### Step 5: Flood from seeds (water rising)

```
Level -2: Only seeds are wet
      .    .    .    .    .    .    .    .    .
      .    .    .    .    .    .    .    .    .
      .    .    A    .    .    .    B    .    .
      .    .    .    .    .    .    .    .    .

Level -1: Water spreads to neighbors with value -1
      .    .    .    .    .    .    .    .    .
      .    A    A    A    .    B    B    B    .
      .    A    A    A    .    B    B    B    .
      .    A    A    A    .    B    B    B    .

Level 0: Stops at mask boundary
      Final: Two separate regions A and B
```

**MEETING RULE:** When water from two different seeds meet,
a "watershed line" (boundary) is created. Each side = separate region.

#### Step 6: Categorize each region by mean

Same as connected method - calculate mean of original pixels in each region.

#### Tuning `min_distance`

| min_distance | Effect |
|--------------|--------|
| Small (3-5) | Sensitive, finds small blobs, might over-segment |
| Medium (10) | Default, balanced |
| Large (15-20) | Only finds big blobs, merges small ones |

```python
# Tune in code:
categorizer = SpatialCategorizer(method="watershed", min_distance=5)
```

---

## Summary: When to Use Each Method

| Method | Best For | Limitation |
|--------|----------|------------|
| **Morphological** | Simple cleanup, preserving bright spots | No region analysis |
| **Connected** | Fast, simple blobs | Merges touching regions |
| **Watershed** | Separating touching blobs | More complex, slower |

---

*Last updated: 2026-01-30*
