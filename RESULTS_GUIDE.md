# Results Guide — ACh imaging in striatum (PG_005)

What the data show for each claim, then where to find it.
Results copied from the OIST clusters on 2026-09-28 into `results/`
(spike-aligned run with `BASELINE_SIGMA_MULT = 2.0`, `MIN_OBJECT_UM2 = 900`).

Terms in *italics* are explained in [Key terms](#2-key-terms).

---

## 1. Claims

### Claim 1 — Spontaneous ACh forms local zones that cover the striatum, but not in phase

**What we see:**
- 70 / 105 recordings (10X) have spontaneous *zones* — 1,330 zones in total.
- Median per recording: 15 zones, zone area ~102,700 µm², one event every ~9 s (0.11 Hz).
- Zones cover a median 56 % of the striatum (*coverage* 0.56).
- Good example: `2025_06_11-0003` — 26 zones, coverage 0.64.
- 35 recordings have no zones at all. They are whole days: 2025_12_14 (16), 2025_11_27 (10), 2025_12_18 (7), 2024_10_11 (2, iAChSnFR).

**Where to look:**
- `spontaneous/spontaneous_summary.xlsx`
  - sheet `recordings` → `n_zones`, `median_area_um2`, `median_freq_hz`, `striatum_coverage`
  - sheet `zones` → every zone: area, number of events, frequency
- `spontaneous/{rec}_ZONE_MAPS.tif` (multi-page)
  - page 1: max projection + all zones in colour, numbered; dashed line = striatum boundary; axes = ventral–dorsal, medial–lateral
  - page 2 onward: one page per detection frame; title = frame / time, *threshold*, max *z*, active zone; blue = zone outline, white = the hotspot in that frame
- `spontaneous/{rec}_ZONES.xlsx` → sheets `trace_corr_groups` / `proximity_groups` / `isolated_tracks` list the **active frames** of each zone

**Still open:**
- **"Not in phase" is not measured yet.** The active frames of each zone are available; a timing comparison between zones is still to do.
- `2025_11_27-0033`: one very large hotspot in mid-recording (frame 1078) was removed by the 80 % filter — worth a look.

---

### Claim 2 — One spike of one neuron releases a hotspot ≥ natural spontaneous hotspots

**What we see:**

Spike-triggered hotspots (*MED*), detected recordings only:

| OBJ | Detected / total | *Reliability* (median) | Spike-frame hotspot (median) | *Lasting time* (median) |
|---|---|---|---|---|
| 10X | 44 / 92 | 66 % | ~80,900 µm² | 75 ms |
| 40X | 8 / 8 | 13 % | ~41,000 µm² | 56 ms |
| 60X | 22 / 66 | 16 % | ~10,500 µm² | 66 ms |

MED hotspot vs. spontaneous zones of the same field of view (10X, `area_comparison.xlsx`):

| Group | n | Median | Q1 – Q3 |
|---|---|---|---|
| A — *estim_induced* MED | 26 | 74,383 µm² | 14,643 – 166,960 |
| B — *spontaneous* MED | 18 | 132,694 µm² | 38,758 – 179,346 |
| C — spontaneous zones (not at the MED spot) | 836 | 45,144 µm² | 30,649 – 73,482 |

| Test (Mann-Whitney U, two-sided) | p |
|---|---|
| A vs C | 0.44 |
| B vs C | 0.006 |
| A vs B | 0.23 |

- Per recording, the MED is larger than a median 84 % of that recording's zones; in 29 / 43 recordings it is larger than at least half of them (`2025_11_27-0010` has no zones).
- Group A is split: many estim MEDs are ~200,000 µm² (e.g. the 2025_12_15 and 2025_06_11 cells), others are ~10,000 µm² (e.g. 2025_11_08-0034 / -0035, 2025_11_13, 2026_01_08).
- Good example: `2025_12_15-0012` (estim_induced) — 8 / 8 segments detected, 209,170 µm² on the spike frame, τ = 84 ms.
- Skipped recordings: 92 × "no significant ACh detection", 35 × "no valid segments" (the neuron fired too densely for a clean baseline).

**Where to look:**
- `area_comparison.xlsx`
  - `MED` → one row per recording: MED area, matched zone, `pct_zones_smaller` (% of that recording's zones smaller than the MED)
  - `Zones` / `Zone events` → zone sizes (per zone / per event)
  - `Groups`, `Group tests` → the two tables above
- `ana_20260922_000_deigo_cells.xlsx`
  - `Summary` → neurons with a detected hotspot (19 / 37)
  - `Spatial`, `Temporal` → hotspot size and lasting time across neurons
  - `Neurons` → which recording of which neuron was detected
  - `Skipped` → why a recording has no result
- `spatial/{rec}_SPATIAL.png`
  - top: hotspot area per frame (star = *critical frame*, green dashed = decay fit with τ)
  - middle: *CAT* masks, frame −4 to +4 (pink = hotspot counted, × = centroid)
  - bottom: all spike waveforms overlaid
- `reliability/{rec}_RELIABILITY.png` → one tile per *segment*; green frame = hotspot found; title = reliability
- `reliability/{rec}_VM_SUCCESS_FAIL.png` → Vm of successful vs. failed segments; black dots = AP threshold
- `spikes/ABF_*_spike_analysis.png` → whole Vm trace; green dots = spikes used
- `median/*_MED.tif`, `categorized/*_CAT.tif` → the movies behind the figures (open in ImageJ)

**Still open:**
- Why group A splits into large and small hotspots (cell? slice? stimulation?).
- The per-objective table above was computed from `results.db`; it is not in a sheet yet.

---

### Claim 3 — Released ACh stays local instead of spreading as a wave

**What we see:**

*Flow pattern* per frame pair, in the spike → spike+4 window:

| OBJ | Recordings | Anisotropic | Source / sink |
|---|---|---|---|
| 10X | 44 | 63 % | 37 % |
| 40X | 8 | 88 % | 12 % |
| 60X | 22 | 84 % | 16 % |

- Example `2025_12_15-0012`: source → sink → sink → sink → anisotropic `L 24° D`. The hotspot appears, shrinks back in place and fades; it does not travel across the field.

**Where to look:**
- `ana_20260922_000_deigo_cells.xlsx`, sheet `Flow pattern` → the table above
- `results.db`, table `flow_pairs` → one row per frame pair: pattern, drift / spread (µm per frame), DV / ML direction
- `flow/{rec}_FLOW.png`
  - row 1: flow arrows on the MED (gray = z), inside the striatum
  - row 2: flow arrows only inside the CAT hotspot
  - row 3: flow speed (µm/s)
- `flow/{rec}_STREAMLINES.png`
  - row 1: streamlines on the MED
  - row 2: streamlines inside the hotspot; title ends with the pattern; last panel has a D–V–M–L crosshair

**Still open:**
- No single "stays local" number yet (e.g. how far the hotspot centre moves vs. its size).

---

## 2. Key terms

| Term | Meaning |
|---|---|
| **Hotspot** | a connected patch of pixels above threshold in one frame |
| **Threshold** | from the intensity histogram: `thr = peak + 2 × σ` (figure titles: `thr = c + 2.0 × σ = value`) |
| **z** (gray colour bar) | intensity in units of baseline σ; 1 = noise level, > 5 = strong signal |
| **Segment** | a short clip around one spike (10 frames before, 10 after) |
| **MED** | pixel-wise median of all segments of one recording — the "typical" response to one spike |
| **CAT** | the MED turned into a binary hotspot mask (window 201 px, density ≥ 0.1, threshold 2σ, objects ≥ 900 µm²) |
| **Reliability** | % of single segments that show a hotspot on their own |
| **Critical frame** | the spike frame or the spike+1 frame, whichever has the larger hotspot |
| **Lasting time** | decay time constant τ of the hotspot area after the peak (ms) |
| **estim_induced / spontaneous** | origin of the spike: evoked by a current pulse (> 200 pA in ABF CH2) vs. fired on its own |
| **Zone** | a place that lights up repeatedly; the union of all hotspot footprints that belong together |
| **trace-corr / proximity / isolated** | how hotspots were grouped into a zone: correlated time traces (r ≥ 0.95) / close by / alone |
| **Coverage** | zone area inside the striatum ÷ striatum area |
| **Optical flow (TV-L1)** | for every pixel, how far the image content moved between two frames (px per frame) |
| **Flow pattern** | fit of the flow inside the CAT hotspot: **source** (spreads out), **sink** (shrinks in), **anisotropic** (drifts one way, e.g. `L 25° D` = lateral, tilted 25° toward dorsal) |

---

## 3. How the results were produced

**Spontaneous** (`spontaneous_analysis.py`, saion job 4734460, 105 × 10X):
1. Threshold each frame (`peak + 2σ`); keep hotspots between 1 % and 80 % of the frame.
2. Chain hotspots across frames into tracks (centroids < 115 px apart).
3. Group tracks into zones (trace correlation, then proximity).
4. Per zone: area, number of events, frequency. Per recording: coverage of the striatum.

**Spike-aligned** (`ach_domain_analysis.py`, deigo job 45504859, 166 recordings):
1. Detect spikes in the patch-clamp Vm (ABF); keep spikes with enough quiet time around them.
2. Cut a ±10-frame segment around each spike → reliability (single segments) and MED (median).
3. MED → CAT hotspot mask → hotspot size on the spike / spike+1 frame, lasting time.
4. TV-L1 optical flow between consecutive MED frames → flow pattern per frame pair.

**Area comparison** (`area_comparison.py`, run locally on the copied `results.db`):
1. MED hotspot size = the larger of the spike / spike+1 frame.
2. Zone size = median per-event area of each spontaneous zone.
3. The zone nearest the MED hotspot (< 50 px) is set aside; the others form group C.

---

## 4. Quick facts

| Item | Value |
|---|---|
| Imaging | 1200 frames at 20 Hz (1 frame = 50 ms), 1024 × 1024 px |
| Sensors | GACh3.0 (most), iAChSnFR (2024_10_11) |
| Objectives | 10X (≈ 1.33 µm/px), 40X, 60X |
| File name | `2025_12_15-0012_A1S3RC1_BIEXP_ALS_...` = date - image serial _ animal 1, slice 3, right hemisphere, cell 1 _ preprocessing |
| `BIEXP_ALS` | movie preprocessed with a bi-exponential detrend (bleaching) + ALS baseline removal (slow drift) |
| Striatum boundary | hand-drawn outlines in `data/bd_20260922_000.json` (used for coverage and DV / ML directions) |
| Logs | `results/logs/` — both `.err` files empty, no crashes |

---

## 5. Folder map

```
results/
├── results.db                              spike-aligned numbers, one row per recording (SQLite)
├── ana_20260922_000_deigo_cells.xlsx       spike-aligned summary sheets
├── area_comparison.xlsx                    MED hotspot vs. spontaneous zones (claim 2)
├── spikes/        ABF_*_spike_analysis.png           Vm trace + which spikes were used
├── reliability/   *_RELIABILITY.png, *_VM_SUCCESS_FAIL.png
├── median/        *_MED.tif                          spike-aligned median movie (21 frames)
├── categorized/   *_CAT.tif                          binary hotspot movie of the MED
├── spatial/       *_SPATIAL.png                      hotspot size over time
├── flow/          *_FLOW.png, *_STREAMLINES.png      how the hotspot moves / spreads
└── spontaneous/
    ├── spontaneous_summary.xlsx            one row per recording + one row per zone
    ├── *_ZONES.xlsx                        per-recording zone details
    ├── *_ZONE_MAPS.tif                     multi-page zone figure
    ├── footprints/  *_ZONES.npz            zone masks (for re-analysis)
    └── mask/        *_ZONE_MASK.tif        thresholded movie (for re-analysis)
```

---

## 6. Known issues

- 9 recordings have `has_region = 1` in `results.db` but 0 % reliability and no MED / CAT files. Treat them as *not detected* (`area_comparison.py` already skips them):
  - 10X: 2025_12_14-0020, -0021, -0022, -0023, 2025_12_18-0023, -0024
  - 60X: 2024_12_19-0010, 2025_04_03-0035, -0038
- Neurons whose recordings were all "no valid segments" never appear in `Neurons`, so `Summary` counts only 37 neurons.
