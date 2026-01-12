# Code Structure Analysis & Organization Guide

## Project Overview
This project performs **acetylcholine (ACh) imaging analysis** with two main workflows:
1. **Image Processing**: Raw image preprocessing (detrending, Gaussian filtering)
2. **Cluster Analysis**: Spike-triggered analysis with k-means clustering to identify ACh release patterns

---

## 📁 Current File Structure

```
PG_005/
├── im_process_main.py          # Main script for image preprocessing
├── cluster_analysis.py         # Main script for spike-triggered ACh analysis
├── classes/
│   ├── __init__.py
│   └── plot_results.py         # GUI plotting class (PySide6 + matplotlib)
└── functions/
    ├── __init__.py
    ├── check_cuda.py           # CUDA availability checker
    ├── test_cuda.py            # CUDA functionality tester
    ├── get_memory_use.py       # Memory usage reporter
    ├── CPU_detrend.py          # CPU-based detrending (Numba JIT)
    ├── CPU_gauss.py            # CPU-based Gaussian blur
    ├── CPU_process.py          # CPU processing orchestrator
    ├── GPU_detrend.py          # GPU-based detrending (CUDA)
    ├── GPU_gauss.py            # GPU-based Gaussian blur (CUDA)
    ├── GPU_process.py          # GPU processing orchestrator
    ├── kmeans.py               # K-means clustering for ACh analysis
    └── spatial_processing.py   # (Currently unused?)
```

---

## 🔗 Dependency Relationships

### **Script 1: `im_process_main.py`**
```
im_process_main.py
├── External: imageio, numpy, numba.cuda, rich
└── Internal (from functions/):
    ├── check_cuda()          → Check if CUDA is available
    ├── test_cuda()           → Test CUDA functionality
    ├── get_memory_usage()    → Get memory consumption
    ├── process_on_gpu()      → GPU processing pipeline
    │   ├── gpu_detrend_jitted()
    │   └── gpu_gaussian_blur()
    └── process_on_cpu()      → CPU processing pipeline (fallback)
        ├── cpu_detrend_jitted()
        └── cpu_gaussian_blur()
```

**Purpose**: Load raw TIFF stacks → Detrend → Gaussian blur → Save results
**Output**: `*_Cal.tif` (detrended), `*_Gauss.tif` (Gaussian-filtered)

---

### **Script 2: `cluster_analysis.py`**
```
cluster_analysis.py
├── External: imageio, numpy, pandas, matplotlib, scipy, pyabf, PySide6, rich
├── Internal (from classes/):
│   └── PlotResults            → Interactive Qt-based peak detection plotter
└── Internal (from functions.kmeans/):
    ├── process_segment_kmeans()                    → Frame-by-frame k-means
    ├── process_segment_kmeans_concatenated()       → Concatenated k-means
    └── visualize_clustering_results()              → Multi-panel visualization
        ├── prepare_frame_for_kmeans()
        ├── apply_kmeans_to_frame()
        ├── calculate_cluster_areas()
        └── split_concatenated_result()
```

**Purpose**:
1. Load ABF (electrophysiology) + TIFF (imaging) → Detect spikes → Segment images
2. Perform Z-score normalization using pre-spike baseline
3. Frequency-based seed pixel analysis (pixels active in ≥X% of segments)
4. ACh clearance analysis (spatial area changes over time)
5. K-means clustering to identify ACh release zones

**Output**: Multiple PNG figures + Excel tables

---

## 📊 Module Descriptions

### **Classes** (`classes/`)

| Module | Class | Purpose |
|--------|-------|---------|
| `plot_results.py` | `PlotResults` | Qt-based interactive window for viewing voltage traces with detected spikes (used only in cluster_analysis.py) |
| | `MplCanvas` | Helper class for matplotlib canvas in Qt |

---

### **Functions** (`functions/`)

#### **1. CUDA/GPU Management**
| File | Function | Purpose |
|------|----------|---------|
| `check_cuda.py` | `check_cuda()` | Verify CUDA availability with diagnostics |
| `test_cuda.py` | `test_cuda()` | Test GPU functionality with sample kernel |
| `get_memory_use.py` | `get_memory_usage()` | Return current process memory usage (GB) |

#### **2. Image Processing - CPU**
| File | Function | Purpose |
|------|----------|---------|
| `CPU_detrend.py` | `cpu_detrend_jitted()` | Numba JIT detrending (moving average subtraction) |
| `CPU_gauss.py` | `cpu_gaussian_blur()` | Numba JIT Gaussian blur |
| `CPU_process.py` | `process_on_cpu()` | **Orchestrator**: Warm up → Detrend → Gaussian blur |

#### **3. Image Processing - GPU**
| File | Function | Purpose |
|------|----------|---------|
| `GPU_detrend.py` | `gpu_detrend_jitted()` | CUDA kernel for parallel detrending |
| `GPU_gauss.py` | `gpu_gaussian_blur()` | CUDA-accelerated Gaussian blur |
| `GPU_process.py` | `process_on_gpu()` | **Orchestrator**: Transfer to GPU → Detrend → Gaussian blur → Transfer back |

#### **4. Clustering & Analysis**
| File | Function | Purpose |
|------|----------|---------|
| `kmeans.py` | `prepare_frame_for_kmeans()` | Reshape 2D image to 1D array |
| | `apply_kmeans_to_frame()` | Run k-means on single frame, sort clusters by intensity |
| | `calculate_cluster_areas()` | Convert pixel counts to µm² based on magnification |
| | `visualize_clustering_results()` | Create multi-panel figure (original, clustered, spike trace) |
| | `process_segment_kmeans()` | Apply k-means frame-by-frame |
| | `process_segment_kmeans_concatenated()` | Apply k-means to horizontally concatenated frames |
| | `concatenate_frames_horizontally()` | Stack frames side-by-side |
| | `split_concatenated_result()` | Split concatenated result back to frames |

---

## 🔍 Key Observations & Issues

### ✅ **Strengths**
1. **Clear separation**: GPU/CPU implementations are modular
2. **Graceful fallback**: CPU processing if CUDA unavailable
3. **Good documentation**: Functions have clear docstrings with "WHY" and "GOAL"
4. **Type hints**: Many functions have proper type annotations

### ⚠️ **Issues to Address**

#### **1. Code Duplication**
- `cluster_analysis.py` has **1500 lines** with multiple analysis sections
- Repeated code patterns (creating figures, scalebars, legends)
- Hardcoded parameters scattered throughout

#### **2. Unclear Dependencies**
- `spatial_processing.py` exists but is never imported (dead code?)
- Both scripts have duplicate logging/console setup

#### **3. Mixed Responsibilities**
- `cluster_analysis.py` does:
  - Data loading
  - Spike detection
  - Segmentation
  - Z-score normalization
  - Frequency analysis
  - Clearance analysis
  - K-means clustering
  - Visualization (5+ different figure types)
  - File I/O

#### **4. Hard-to-Maintain Configuration**
```python
# Scattered throughout cluster_analysis.py:
exp_date = "2025_12_15"
magnification: str = "10X"
z_threshold = 0.25
minimal_required_frames: int = 3
maximum_allowed_frames: int = 4
TTL_5V_HIGH: float = 2.0
```

---

## 💡 Recommended Refactoring Plan

### **Phase 1: Extract Configuration**
Create `config.py`:
```python
class AnalysisConfig:
    # Experiment parameters
    EXP_DATE = "2025_12_15"
    MAGNIFICATION = "10X"

    # Detection parameters
    TTL_HIGH_THRESHOLD = 2.0
    TTL_LOW_THRESHOLD = 0.8
    SPIKE_MIN_DISTANCE = 1500
    SPIKE_MIN_PROMINENCE = 10

    # Segmentation parameters
    MIN_REQUIRED_FRAMES = 3
    MAX_ALLOWED_FRAMES = 4

    # Analysis parameters
    Z_SCORE_THRESHOLD = 0.25
    FREQUENCY_PERCENTAGES = [50, 60, 70, 80, 90, 99, 100]
    KMEANS_CLUSTERS = 3
```

### **Phase 2: Reorganize Functions Module**
```
functions/
├── __init__.py
├── config.py                 # NEW: Configuration constants
├── hardware/                 # NEW: Group CUDA-related
│   ├── __init__.py
│   ├── check_cuda.py
│   ├── test_cuda.py
│   └── get_memory_use.py
├── preprocessing/            # NEW: Group image processing
│   ├── __init__.py
│   ├── cpu_ops.py           # Merge CPU_detrend + CPU_gauss
│   ├── gpu_ops.py           # Merge GPU_detrend + GPU_gauss
│   ├── cpu_process.py
│   └── gpu_process.py
├── spike_detection/          # NEW: Extract from cluster_analysis
│   ├── __init__.py
│   ├── peak_finder.py       # Extract spike detection logic
│   └── segmentation.py      # Extract segmentation logic
├── analysis/                 # NEW: Analysis functions
│   ├── __init__.py
│   ├── normalization.py     # Z-score functions
│   ├── frequency_analysis.py
│   ├── clearance_analysis.py
│   └── kmeans.py            # Keep as is
└── visualization/            # NEW: Extract plotting
    ├── __init__.py
    ├── frequency_plots.py
    ├── clearance_plots.py
    └── cluster_plots.py
```

### **Phase 3: Simplify Main Scripts**

**New `im_process_main.py`** (reduce to ~50 lines):
```python
from pathlib import Path
from functions.hardware import check_cuda, test_cuda
from functions.preprocessing import process_on_gpu, process_on_cpu
from functions.config import ProcessingConfig

def main():
    config = ProcessingConfig.load()

    # Check hardware
    use_gpu = check_cuda() and test_cuda()

    # Process files
    for filename in config.file_list:
        img_raw = load_image(filename)

        if use_gpu:
            detrended, gaussian = process_on_gpu(img_raw)
        else:
            detrended, gaussian = process_on_cpu(img_raw)

        save_results(filename, detrended, gaussian)

if __name__ == "__main__":
    main()
```

**New `cluster_analysis.py`** (reduce to ~200 lines):
```python
from functions.config import AnalysisConfig
from functions.spike_detection import detect_peaks, create_segments
from functions.analysis import (
    zscore_normalize,
    frequency_analysis,
    clearance_analysis,
    kmeans_clustering
)
from functions.visualization import (
    plot_frequency_results,
    plot_clearance_results,
    plot_clustering_results
)

def main():
    config = AnalysisConfig.load()

    # Load data
    abf_data, img_data = load_data(config)

    # Detect spikes
    peaks = detect_peaks(abf_data, config)

    # Create segments
    segments = create_segments(img_data, peaks, config)

    # Normalize
    normalized = zscore_normalize(segments)

    # Analyze
    freq_results = frequency_analysis(normalized, config)
    clearance_results = clearance_analysis(normalized, config)
    cluster_results = kmeans_clustering(normalized, config)

    # Visualize
    plot_frequency_results(freq_results)
    plot_clearance_results(clearance_results)
    plot_clustering_results(cluster_results)

    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🎯 Immediate Quick Wins (No Refactoring Required)

### **1. Add Module Docstrings**
Add to top of each file:
```python
"""
Module: im_process_main.py
Purpose: Preprocess raw TIFF image stacks (detrending + Gaussian filtering)
Input: Raw TIFF files from raw_images/
Output: Processed TIFF files (*_Cal.tif, *_Gauss.tif)
"""
```

### **2. Extract Magic Numbers**
At top of `cluster_analysis.py`:
```python
# Configuration (move to config.py later)
EXP_CONFIG = {
    "date": "2025_12_15",
    "magnification": "10X",
    "z_threshold": 0.25,
    # ...
}
```

### **3. Add Section Comments**
```python
# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

# ============================================================================
# SECTION 2: SPIKE DETECTION
# ============================================================================

# etc...
```

### **4. Extract Long Functions**
Current line 753-1092 in `cluster_analysis.py` (ACh clearance analysis) → extract to function:
```python
def perform_ach_clearance_analysis(
    segments: list,
    z_threshold: float,
    display_positions: list
) -> tuple[pd.DataFrame, dict]:
    """Analyze ACh clearance by measuring active area over time."""
    # Move 340 lines here
    return area_stats, avg_frames_by_position
```

---

## 📈 Benefits After Refactoring

| Aspect | Before | After |
|--------|--------|-------|
| **Readability** | 1500-line monolith | ~200-line orchestrator + small modules |
| **Testability** | Hard to test | Each function testable independently |
| **Reusability** | Plotting code duplicated | Reusable visualization functions |
| **Maintainability** | Change requires editing multiple places | Change config once |
| **Collaboration** | Merge conflicts likely | Clear module boundaries |

---

## 🚀 Implementation Priorities

### **Priority 1 (Week 1)**: Documentation & Organization
- [ ] Add module docstrings to all files
- [ ] Extract magic numbers to constants at file top
- [ ] Add clear section separators in cluster_analysis.py

### **Priority 2 (Week 2)**: Extract Configuration
- [ ] Create `config.py` with all parameters
- [ ] Update both main scripts to use config

### **Priority 3 (Week 3-4)**: Modularize Functions
- [ ] Extract spike detection → `functions/spike_detection/`
- [ ] Extract visualization → `functions/visualization/`
- [ ] Extract analysis → `functions/analysis/`

### **Priority 4 (Week 5)**: Reorganize Function Folders
- [ ] Group CUDA functions → `functions/hardware/`
- [ ] Merge CPU/GPU ops → `functions/preprocessing/`

---

## 📞 Questions for You

1. **Is `spatial_processing.py` still needed?** It's not imported anywhere.
2. **Do you want to keep averaged output?** `process_on_cpu` returns 3 values but `process_on_gpu` returns 2.
3. **Should I create the refactored version?** Or just document the current structure?
4. **Testing framework**: Do you want unit tests for the refactored modules?

---

*Generated: 2026-01-12*
