# PG_005 - Acetylcholine Imaging Analysis

Two-stage pipeline for analyzing acetylcholine (ACh) release from fluorescence imaging:
1. **Image Preprocessing** (`im_preprocess.py`): Raw TIFF detrending + Gaussian filtering (CPU/GPU)
2. **Spike-Triggered Analysis** (`im_dynamics.py`): Spike-aligned imaging with spatial categorization

## Documentation

**IMPORTANT**: Before implementing algorithms or modifying analysis pipelines, always consult:
- `doc/PROJECT_SUMMARY.md` - Detailed workflows, algorithms, API examples, parameter explanations
- `doc/DEPENDENCY_DIAGRAM.md` - Architecture overview, module dependencies, data flow

These docs contain critical context about why algorithms work the way they do.

## Project Structure

```
PG_005/
├── im_preprocess.py          # Main: Image preprocessing
├── im_dynamics.py            # Main: Spike-triggered analysis
├── classes/                  # Core analysis classes
│   ├── spatial_categorization.py  # SpatialCategorizer (3 methods, 4 threshold types)
│   ├── region_analyzer.py         # RegionAnalyzer (area, centroid, contours)
│   ├── abf_clip.py                # ABF clipping + spike detection
│   ├── plot_results.py            # Qt interactive viewers
│   └── results_exporter.py        # SQLite + NPZ + TIFF exporter
└── functions/                # Processing functions
    ├── cpu_*.py              # Numba JIT implementations
    ├── gpu_*.py              # CUDA kernel implementations
    └── spike_centered_processes.py, imaging_segments_zscore_normalization.py
```

## Coding Conventions

### General Style
- Type hints required for all public functions
- Docstrings: NumPy style with Parameters/Returns sections
- Use `rich` for console output (progress bars, formatted tables)
- Prefer descriptive variable names (e.g., `categorized_frames` not `cat_frm`)

### Algorithm Preferences
- **Spike-aligned averaging**: Use `spike_centered_median()` over `spike_centered_avg()` (robust to outliers)
- **Spatial categorization**: Default to morphological method unless user specifies otherwise
- **Threshold selection**: Prefer automatic methods (li_double, otsu_double) over manual
- **Processing path**: Prefer GPU when CUDA available, gracefully fall back to CPU

### Data Conventions
- ABF/image pair naming: `abf{N}_img{M}` (e.g., abf1_img2)
- Output directory: `results/data/{date}/abf{N}_img{M}/`
- TIFF saving: float16 to save space (preprocessing handles float32 internally)
- Pixel scaling factors:
  - 10X: 1.333 µm/pixel
  - 40X: 0.333 µm/pixel
  - 60X: 0.222 µm/pixel

### Class Design Patterns
- **SpatialCategorizer**: Use factory methods (`.connected()`, `.watershed()`, `.morphological()`)
- **RegionAnalyzer**: Always specify objective (`obj="10X"`) for proper µm² conversion
- **Fit-transform pattern**: `.fit(data)` returns self for chaining, `.get_results()` for output

## Important Implementation Notes

### Gaussian Filtering (σ parameter)
The σ parameter is critical and scientifically justified:
```
noise_correlation_length < σ < signal_correlation_length
```
- Noise: ~1-2 pixels (spatially uncorrelated)
- ACh signal: >50 pixels (cellular-scale release sites)
- Default σ=6 pixels @ 60X = 1.3 µm (validated range)

Don't change this without consulting docs.

### Spatial Categorization Methods
Each method has specific use cases (see doc/PROJECT_SUMMARY.md for detailed comparison):
- **Morphological**: Best for simple cleanup, preserves bright spots
- **Connected**: Fast, good for isolated blobs
- **Watershed**: Separates touching blobs (tunable with `min_distance`)

### Z-Score Normalization
Baseline = ALL frames before spike (not a fixed window). This is intentional to capture true pre-spike state.

## Development Workflow

### Before Modifying Analysis Code
1. Check `doc/PROJECT_SUMMARY.md` for algorithm explanation
2. Check `doc/DEPENDENCY_DIAGRAM.md` for where the function is used
3. Consider impact on both CPU and GPU pipelines if changing preprocessing

### Testing Changes
- Test both CPU and GPU paths when modifying preprocessing
- Verify Qt plots still work after changing analysis outputs
- Check SQLite schema compatibility if modifying ResultsExporter

### Adding New Features
- New analysis methods → add to `classes/`
- New processing functions → add to `functions/` with both CPU/GPU if applicable
- New plots → extend `classes/plot_results.py`

## Common Tasks

**Adding a new threshold method:**
- Update `SpatialCategorizer._threshold_methods` dict
- Add method to `SpatialCategorizer` class
- Update doc/PROJECT_SUMMARY.md threshold table

**Changing pixel scaling:**
- Update `RegionAnalyzer.PIXEL_SCALING` dict
- Update doc/PROJECT_SUMMARY.md pixel scaling table

**Modifying output format:**
- Update `ResultsExporter` methods
- Consider SQLite schema changes
- Update doc/DEPENDENCY_DIAGRAM.md data flow section

---

*When in doubt, check the docs. They explain the "why" behind implementation choices.*
