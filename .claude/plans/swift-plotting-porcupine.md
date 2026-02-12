# Plan: Unified GUI Application for RADIO (PG_005)

## Project Context

**RADIO** (Response Associated Distribution Imaging Observer) is an acetylcholine imaging analysis platform with:
- **2 main workflows**: Image preprocessing + Spike-aligned analysis
- **Existing components**: 4 Qt interactive viewers (PlotPeaks, PlotSegs, PlotSpatialDist, PlotRegion)
- **Current limitation**: Workflows run via separate CLI scripts (`im_preprocess.py`, `im_dynamics.py`)

---

## Goal

Create a **unified GUI application** that:
1. Integrates both preprocessing and spike-aligned analysis workflows
2. Provides intuitive parameter configuration
3. Displays real-time processing progress
4. Embeds existing Qt viewers for interactive visualization
5. Simplifies the end-to-end analysis pipeline

---

## Current State Analysis

### Existing GUI Components ✓
- **PlotPeaks**: Spike detection visualization on voltage trace
- **PlotSegs**: Individual spike segment browser
- **PlotSpatialDist**: Spatial categorization overview with traces
- **PlotRegion**: Frame-by-frame region viewer with contours
- All built with **PySide6** (Qt)

### Existing Workflows ✓
1. **Preprocessing** (`im_preprocess.py`):
   - Input: Raw TIFF stacks
   - Processing: Detrend → Gaussian blur (CPU/GPU)
   - Output: *_Cal.tif, *_Gauss.tif

2. **Spike-Aligned Analysis** (`im_dynamics.py`):
   - Input: Processed TIFF + ABF files
   - Processing: Spike detection → Segmentation → Z-score → Median → Categorization → Region analysis
   - Output: SQLite DB + TIFF stacks + NPZ + PNG figures

### Key Classes for Integration ✓
- `AbfClip`: ABF data handling + spike detection
- `SpatialCategorizer`: 3 methods (connected, watershed, morphological) + 4 threshold methods
- `RegionAnalyzer`: Region properties (area, centroid, contours)
- `ResultsExporter`: Save to SQLite + files

---

## Questions for User

1. **GUI Scope**: Should the GUI be:
   - A) Full pipeline tool (preprocessing → analysis → export) ✓ (Recommended)
   - B) Analysis-only tool (assumes preprocessing is done)
   - C) Parameter configuration + batch processing tool

2. **User Workflow**: What's the typical usage pattern?
   - A) Process one experiment at a time with immediate visualization
   - B) Batch process multiple experiments, then review results
   - C) Both workflows needed

3. **Parameter Presets**: Should the GUI support:
   - Saving/loading analysis parameter presets?
   - Different presets for different microscope objectives (10X, 40X, 60X)?

4. **Additional Features** (optional):
   - Results comparison viewer (compare multiple experiments)?
   - Integration with recording summary Excel files?
   - Export reports (PDF/HTML summary of analysis)?

---

## Proposed Implementation Plan

### Phase 1: Main Window & Project Structure ✓

**Goal**: Create main application window with workflow navigation

**Tasks**:
1. Create `gui_main.py` - Main application entry point
2. Design main window layout:
   - Workflow tabs: "Preprocessing" | "Spike Analysis" | "Results"
   - Menu bar: File (Open/Save project) | Tools (Settings) | Help
   - Status bar: Progress indicator + log messages
3. Implement project file system:
   - Load/save GUI state (last used directories, parameters)
   - Session management

**Files to create**:
- `gui_main.py` - Main window (QMainWindow)
- `gui/` (new directory)
  - `__init__.py`
  - `main_window.py` - Main window class
  - `config.py` - Configuration management

---

### Phase 2: Preprocessing Tab ✓

**Goal**: GUI for image preprocessing workflow

**Tasks**:
1. File selection widget:
   - Browse button for raw TIFF directory
   - File list view with preview
2. Parameter configuration:
   - CPU/GPU selection (auto-detect CUDA)
   - Gaussian sigma (default: 6, range: 3-10)
   - Detrending parameters
3. Processing controls:
   - "Run Preprocessing" button
   - Progress bar (per-file)
   - Real-time log output
4. Output preview:
   - Side-by-side comparison: Raw | Detrended | Gaussian
   - Frame slider to browse stack

**Files to create**:
- `gui/preprocess_tab.py` - Preprocessing interface
- `gui/widgets/image_preview.py` - TIFF stack preview widget
- `gui/workers/preprocess_worker.py` - Background processing thread

**Integration points**:
- Use existing `functions/check_cuda.py`, `process_on_cpu()`, `process_on_gpu()`

---

### Phase 3: Spike Analysis Tab ✓

**Goal**: GUI for spike-aligned analysis workflow

**Tasks**:
1. File selection widget:
   - Browse for processed TIFF (*_Gauss.tif)
   - Browse for corresponding ABF file
   - Validate file pairing
2. Spike detection parameters:
   - Height, prominence, distance (scipy.find_peaks)
   - Preview detected spikes (embed PlotPeaks)
3. Categorization parameters:
   - Method dropdown: Connected | Watershed | Morphological
   - Threshold method: manual | multiotsu | li_double | otsu_double
   - Method-specific parameters (min_region_size, min_distance, kernel_size)
   - Objective selector: 10X | 40X | 60X
4. Processing controls:
   - "Run Analysis" button
   - Multi-stage progress (spike detection → segmentation → categorization → region analysis)
5. Results preview:
   - Embed PlotSpatialDist for overview
   - Embed PlotRegion for detailed inspection
   - Export button → trigger ResultsExporter

**Files to create**:
- `gui/analysis_tab.py` - Spike analysis interface
- `gui/widgets/parameter_panel.py` - Dynamic parameter configuration widget
- `gui/workers/analysis_worker.py` - Background analysis thread

**Integration points**:
- Use existing: AbfClip, SpatialCategorizer, RegionAnalyzer, ResultsExporter
- Embed existing: PlotPeaks, PlotSegs, PlotSpatialDist, PlotRegion

---

### Phase 4: Results Browser Tab ✓

**Goal**: Browse and compare previous analysis results

**Tasks**:
1. Results database viewer:
   - Query SQLite (`results/results.db`) for experiments
   - Table view: Date | ABF | Image | Threshold Method | # Spikes | # Regions
   - Search/filter by date, method, etc.
2. Result loader:
   - Click experiment → load saved TIFF stacks + NPZ
   - Display in PlotSpatialDist/PlotRegion
3. Comparison mode (optional):
   - Select multiple experiments
   - Side-by-side or overlay comparison

**Files to create**:
- `gui/results_tab.py` - Results browser interface
- `gui/widgets/results_table.py` - SQLite results table viewer

**Integration points**:
- Query `results/results.db`
- Load TIFF/NPZ from `results/{exp_date}/abf{}_img{}/`

---

### Phase 5: Settings & Polish ✓

**Goal**: Configuration management + user experience improvements

**Tasks**:
1. Settings dialog:
   - Default directories (raw_images, raw_abfs, processed_images, results)
   - Default parameters (Gaussian sigma, threshold methods, etc.)
   - GPU preferences
2. Parameter presets:
   - Save current parameters as named preset
   - Load preset from dropdown
   - Objective-specific presets (10X, 40X, 60X)
3. Batch processing mode:
   - Select multiple TIFF+ABF pairs
   - Queue processing with progress
4. Help/documentation:
   - Link to PROJECT_SUMMARY.md
   - Tooltips for all parameters (explain what they do)

**Files to create**:
- `gui/settings_dialog.py` - Settings window
- `gui/preset_manager.py` - Parameter preset system
- `gui/batch_processor.py` - Batch processing queue

---

## Technical Design Decisions

### UI Framework
- **PySide6** (Qt for Python) - Already used in existing viewers
- Modern Qt widgets: QTabWidget, QDockWidget for flexible layouts

### Threading Strategy
- **QThread** for background processing (prevent GUI freeze)
- **Signals/Slots** for progress updates and completion notifications
- **Queue** for batch processing

### State Management
- **JSON config file** (`~/.radio/config.json`) for user preferences
- **SQLite** (reuse `results.db`) for analysis history
- **Parameter presets** stored as JSON in `~/.radio/presets/`

### Integration with Existing Code
- **Minimal modifications** to existing classes
- GUI wraps existing functions, doesn't reimplement
- Existing viewers (PlotPeaks, etc.) embedded using QDockWidget or QTabWidget

---

## File Structure (New)

```
PG_005/
├── gui_main.py                    # Main application entry point
├── gui/
│   ├── __init__.py
│   ├── main_window.py             # Main window (QMainWindow)
│   ├── config.py                  # Configuration management
│   ├── preprocess_tab.py          # Preprocessing interface
│   ├── analysis_tab.py            # Spike analysis interface
│   ├── results_tab.py             # Results browser
│   ├── settings_dialog.py         # Settings window
│   ├── preset_manager.py          # Parameter presets
│   ├── batch_processor.py         # Batch processing
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── image_preview.py       # TIFF stack viewer
│   │   ├── parameter_panel.py     # Dynamic parameter widget
│   │   └── results_table.py       # SQLite results viewer
│   └── workers/
│       ├── __init__.py
│       ├── preprocess_worker.py   # Background preprocessing
│       └── analysis_worker.py     # Background analysis
├── classes/                        # (existing, unchanged)
├── functions/                      # (existing, unchanged)
└── ...
```

---

## Success Criteria

✓ Users can run full pipeline (preprocessing → analysis) without touching CLI
✓ All existing Qt viewers integrated and functional
✓ Real-time progress feedback during processing
✓ Parameter presets speed up repeated analyses
✓ Results browser allows reviewing past experiments
✓ Clear, intuitive interface for non-programmers

---

## Next Steps

1. **Get user feedback** on Questions section above
2. **Create Phase 1** skeleton (main window + tabs)
3. **Iterate** - build one tab at a time, test with real data
4. **Document** - add GUI usage guide to `doc/`

---

*Plan created: 2026-02-07*
*Status: Ready for user review and approval*
