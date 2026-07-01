# DBSCAN Cluster Detection — Notes & Open Questions

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

## B+D% (frame coverage %) — Idea 1

**Definition**: fraction of non-background pixels in a frame

```
B+D% = count(pixels > 0) / total_pixels * 100
```

**Purpose**: detect whether a real ACh release event occurred.

A real event shows a clear elevation at spike frame (or spike+1) vs spike-1 (baseline).

### Delayed signal fix

Run DBSCAN on whichever of spike / spike+1 has the higher B+D% — handles cases where the signal peaks one frame after the electrophysiology spike.

---

## Cluster size filter — Idea 2

After DBSCAN, many small clusters remain (dim pixels scattered by Gaussian noise exceeding the 2σ threshold). A size filter removes these.

### Current implementation: MIN_CLUSTER_FRACTION = 0.05

```
keep cluster  if  cluster_px / total_non_bg_px  >=  5%
```

- `total_non_bg_px` = all non-bg pixels in the **same analysis frame**
- Self-calibrating: threshold scales automatically with signal level per recording

### Why 5%?

Honestly it is a guessed number. It has worked well on the test cases so far but has no principled biological basis yet.

### Rejected alternatives

| Method | Problem |
|--------|---------|
| SNR: `cluster_pct / baseline_pct >= 2x` | baseline can be 0% → denominator collapses; multi-cluster recordings get each cluster rejected individually |
| Fixed µm² threshold | Absolute — does not adapt to OBJ or signal level; 10X and 60X recordings need very different pixel counts |

---

## Spreading detection design (settled)

**Goal**: detect whether ACh signal propagates outward from the release centroid.

**Approach**: concentric equal-area rings centered on the DBSCAN centroid, using raw z-score traces (not CAT categories).

- R = RMS distance of all cluster pixels from centroid (cluster radius, scale-adaptive)
- Split point at **R/√2 ≈ 0.71R** → equal area in both zones for a circular cluster
- Inner zone: pixels at distance 0 to R/√2 (tight core)
- Outer zone: pixels at distance R/√2 to R (near boundary)
- Both zones stay entirely within the cluster — no background noise pixels captured
- Signal per zone: mean z-score from MED stack (continuous, no threshold artifact)

**Expected spreading signature**: inner zone peaks before outer zone.

**Why not bright/dim comparison?**
- Bright and dim areas are determined by intensity thresholds → completely arbitrary relative sizes
- Pixels change category across frames (dim→bright as signal grows) → fixed-mask traces are unreliable
- Distance rings avoid both problems

---

## Open TODOs

### 1. ✅ Make eps and min_samples OBJ-consistent — DONE

EPS_UM=10µm (inter-varicosity gap distance) + MIN_DENSITY_FRAC=0.1 (10% of eps-circle area).
`lookup_obj()` queries `rec_data.db` per recording; `eps_and_min_samples(obj)` converts to pixels.
Currently: 10µm → 45px at 60X, 7px at 10X.
**Still needs**: visual validation of EPS_UM=10µm against output PNGs.

### 2. Principled cluster size filter

`MIN_CLUSTER_FRACTION = 0.05` is a guess. Once EPS_UM and MIN_AREA_UM2 are set in physical units, the cluster size filter could also be expressed in µm² for consistency.

### 3. Integrate into RegionAnalyzer

Once the demo approach is validated, replace the current largest-CC method in `classes/region_analyzer.py` with the DBSCAN-based approach.

### 4. Implement ring-based spreading analysis

Per the settled design above (partially in demo):
- ⚠️ **R computation is buggy**: currently RMS distance — inflated by scattered outlier pixels far from the dense core (e.g. 0029 visual cluster ~50px but R=109px). Fix: use 90th-percentile distance instead of RMS
- Split pixels into inner (0–R/√2) and outer (R/√2–R) zones
- Extract z-score traces from MED stack for each zone
- Compare peak timings → time lag = spreading signal

### 5. Visual tweak for demo figure

Cluster shading colors/alpha in `_demo_dbscan_tmp.py` need further tuning (deferred).
