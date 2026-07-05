# DBSCAN Cluster Detection — Notes & Open Questions

> Reflects the current implementation in `classes/region_analyzer.py`. The old
> demo file (`_demo_dbscan_tmp.py`) has been archived — DBSCAN has been fully
> integrated into `RegionAnalyzer` since Session 43.

## What is DBSCAN?

**Density-Based Spatial Clustering of Applications with Noise**

- Finds clusters automatically — no need to specify number of clusters upfront
- Returns a label per input point: `0, 1, 2, ...` for clusters, `-1` for noise
- Two key parameters: `eps` (neighbourhood radius) and `min_samples` (minimum neighbours to be a core point)

---

## How DBSCAN works — step by step

1. For each pixel, count neighbours within `eps` distance (Euclidean)
2. If a pixel has ≥ `min_samples` neighbours → it's a **core point**
3. Start a cluster from a core point
4. **Expand**: add all pixels within `eps` of this core point
5. For each newly added pixel that is also a core point → expand again from it
6. Repeat until no new pixels are found within `eps` of any cluster member
7. Move on to the next unvisited core point → start a new cluster
8. Any pixel never reached by any cluster → **noise** (label = -1)

### Key insight — "reachable" means chaining

A cluster can span 1000px even with `eps=20` because it expands in 20px hops continuously.
There is no gap > 20px anywhere along the path — like a flood fill that spreads 20px at a time.

### Distance metric

Plain Euclidean (Pythagoras):
```
distance = sqrt((row_A - row_B)² + (col_A - col_B)²)
```
Diagonal neighbours are slightly farther (~1.41px) than straight neighbours (1px).

---

## How the CAT.tif is used

- Pixel values: `0` = background, `1` = dim, `2` = bright
- DBSCAN runs on **all non-background pixels** (values 1 and 2 combined)
- Input to DBSCAN: array of `(row, col)` coordinates of every non-bg pixel

---

## B+D% (`area_pct`) and picking the critical frame

**Definition** (`compute_area_pct`): fraction of non-background pixels in a frame

```
area_pct = count(pixels > 0) / total_pixels * 100
```

**Purpose**: detect whether a real ACh release event occurred, and pick which
frame to run DBSCAN on.

### Critical-frame pick (`pick_critical_frame`) — baseline+σ threshold, not "higher wins"

Comparing spike vs spike+1 by "whichever is higher" was noise-prone, so the
current logic instead compares each candidate against a **baseline-derived
significance threshold**:

```
threshold = mean(area_pct[:spike_idx]) + AREA_PCT_SIGMA_MULT * std(area_pct[:spike_idx])
```

- Defaults to the **spike frame** unless it doesn't clear the threshold and
  spike+1 does (delayed-signal case)
- `AREA_PCT_SIGMA_MULT = 5.0` (tuned down from an initial `10.0` after real-data
  testing showed 10σ silently defaulted to the spike frame even when spike+1
  was clearly dominant)

---

## eps / min_samples — OBJ-consistent physical units

`eps_and_min_samples(obj)` converts physical units to pixels per objective:

- `EPS_UM = 10.0` µm — inter-varicosity gap distance (tunable), **not** varicosity
  size (sub-resolution) or axon span (cluster-size upper bound)
- `MIN_DENSITY_FRAC = 0.1` — `min_samples = 0.1 * π * eps_px²` (10% of the eps-circle area)
- Per `PIXEL_SCALE`: 10X→0.75 px/µm, 40X→3.0 px/µm, 60X→4.5 px/µm
  (e.g. 10µm → 7px at 10X, 45px at 60X)

---

## Cluster size filter — `MIN_CLUSTER_FRACTION`

After DBSCAN, many small clusters remain (dim pixels scattered by Gaussian
noise exceeding the categorization threshold). `_run_cluster_seeker` drops
undersized ones:

```
keep cluster  if  cluster_px / total_non_bg_px  >=  MIN_CLUSTER_FRACTION (0.05)
```

- `total_non_bg_px` = all non-bg pixels in the **same frame being clustered**
- Self-calibrating: threshold scales automatically with signal level per recording
- Kept clusters are remapped to `0..N-1`, **largest first**; everything else
  becomes noise (`-1`)
- Still a guessed number — worked well on test cases, no principled biological
  basis yet

### Rejected alternatives

| Method | Problem |
|--------|---------|
| SNR: `cluster_pct / baseline_pct >= 2x` | baseline can be 0% → denominator collapses; multi-cluster recordings get each cluster rejected individually |
| Fixed µm² threshold | Absolute — does not adapt to OBJ or signal level; 10X and 60X recordings need very different pixel counts |

---

## Saturation guard — `SATURATION_AREA_PCT`

A real 60X recording once hit **82.5% non-background coverage** and made
sklearn's DBSCAN neighbor-graph construction balloon to 73GB+ memory (thrashing,
not just slow).

`_detect_clusters` now skips DBSCAN entirely when the frame's `area_pct >= 15.0`
(`SATURATION_AREA_PCT`) and instead treats every non-background pixel as **one
big cluster** — at that density it's one region, not discrete release sites,
so it still gets ring-split like any other single-cluster case (not reported
as zero clusters). `RegionAnalyzer.saturated` flags when this happened.

This is the single shared implementation used by both the critical frame
(`__init__`) and the independent max-area frame (`_compute_max_area`) — a
duplication bug (guard implemented twice, slightly differently) was fixed by
consolidating into this one function.

---

## Enclosing-circle radius R — `_resolve_R`

R is the **farthest cluster-pixel distance from the centroid** (not RMS —
an earlier RMS-based R was inflated by scattered outlier pixels, e.g. a
visually ~50px cluster measuring R=109px), capped at the centroid's distance
to the nearest frame edge:

```
R = min(farthest_pixel_distance, distance_from_centroid_to_nearest_edge)
```

The cap fixes a real overshoot bug: an uncapped R could draw a circle that
extends past the frame boundary whenever the centroid isn't near the image
center — capping guarantees the drawn circle stays inside the frame while
still containing every cluster pixel (unless the true farthest pixel is
closer than the edge, in which case that distance wins).

---

## Ring-based spreading detection (1 cluster) vs cross-cluster asynchrony (>1 cluster)

**Goal**: detect whether ACh signal propagates outward from the release
centroid, or whether multiple release sites fire out of sync.

These are genuinely different computations, not one shared formula — an
earlier attempt to unify them collapsed two different concepts into one and
was corrected mid-session:

### Exactly 1 kept cluster → `compute_ring_traces`

- Split point at **R/√2 ≈ 0.71R** → equal area in both zones for a circular cluster
- Inner zone: pixels at distance `0` to `R/√2` (tight core)
- Outer zone: pixels at distance `R/√2` to `R` (near boundary)
- Both zones stay entirely within the cluster's own pixels — no background noise captured
- Signal per zone: mean z-score from the MED stack (continuous, no threshold artifact)
- Latency (`get_peak_latency_ms`) = outer-ring peak time − inner-ring peak time
  (spread *within* the one release site)

### More than 1 kept cluster → `compute_cluster_trace`

- One whole-cluster z-score trace per cluster, **no ring split** — lets
  cluster-to-cluster peak timing be compared directly instead of measuring
  spread within each individually
- Latency (`get_peak_latency_ms`) = latest-peaking cluster's time − earliest-peaking
  cluster's time (largest asynchrony *between* separate release sites)

**Why not bright/dim comparison?**
- Bright and dim areas are determined by intensity thresholds → completely arbitrary relative sizes
- Pixels change category across frames (dim→bright as signal grows) → fixed-mask traces are unreliable
- Distance rings avoid both problems

---

## Max-area frame — independent second stat

`_compute_max_area` picks whichever of spike/spike+1 has the larger **raw**
`area_pct`, independent of `pick_critical_frame`'s significance-threshold pick
(decoupled — it's a headline area stat, not a latency input).

- Runs the same DBSCAN cluster-detection path (`_detect_clusters`) on that frame
- `max_area_um2` sums pixels across **all** kept clusters on that frame (not
  just the largest) — an earlier version used the raw non-background pixel
  count with no clustering/filtering at all, which was wrong and got fixed
  after spot-checking real DB rows where it disagreed with the
  cluster-filtered `critical_frame_area_um2` on the same physical frame
- `max_area_eq_radius_um = sqrt(max_area_um2 / π)` — circle-equivalent radius
  of that same area, directly comparable to `R_lat_um`. Two earlier attempts at
  a second spatial-extent stat for this frame (a second DBSCAN-clustered R,
  then an x/y bounding-box span with a cross+label visualization) were both
  discarded per user review before this simpler formula was added.

---

## Open TODOs

### Visual tweak for demo figures

Cluster shading colors/alpha could still use tuning — carried over, low priority.

### `MIN_CLUSTER_FRACTION` / eps µm-basis

Still guessed constants (`MIN_CLUSTER_FRACTION=0.05`, `EPS_UM=10.0`). No
principled biological basis yet — validated only against real test recordings
seen so far.

### Detection gate (`has_event` boolean)

Still not implemented: a formal B+D%-vs-baseline gate that decides *whether*
an event occurred at all (currently `pick_critical_frame`/`saturated` only
decide *which* frame and *how* to cluster, not a pass/fail event flag).
