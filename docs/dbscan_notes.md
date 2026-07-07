# Cluster Detection — Notes & Design Decisions

> Reflects the current implementation in `classes/region_analyzer.py`.
> DBSCAN was replaced by distance-transform + connected components in the session
> where 60X recordings caused memory explosions. See `distance_transform_clustering.md`
> for a full explanation of the replacement algorithm.

---

## How clustering works now

`_run_cluster_seeker` groups **bright pixels only** (category = 2) into clusters:

1. Compute Euclidean distance from every pixel to the nearest bright pixel
   (`distance_transform_edt`)
2. Mark every pixel within `eps_px` of any bright pixel as reachable
   (`within_eps_of_bright`)
3. Find connected components in that reachable mask (`component_map`)
4. Intersect each component back with `bright_mask` → only original bright pixels
   survive
5. Drop components below `MIN_CLUSTER_FRACTION`
6. Sort survivors largest first → cluster index `0` is always the biggest

Dim pixels (category = 1) are excluded entirely — they never enter the
clustering, size filter, centroid, or label_frame.

---

## B% (`area_pct`) and picking the critical frame

**Definition** (`compute_area_pct`): fraction of **bright** pixels in a frame
(not bright+dim — dim pixels are excluded here too):

```
area_pct = count(pixels == 2) / total_pixels * 100
```

**`critical_frame_area_pct`** — the raw B% at the critical frame (exported to
the DB as a threshold-check-friendly percentage).

**`critical_frame_area_um2`** — the cluster-filtered area at the critical frame:
sums only kept-cluster bright pixels (undersized-cluster pixels excluded).

### Critical-frame pick (`pick_critical_frame`) — baseline+σ threshold

Defaults to the **spike frame** unless it doesn't clear the threshold and
spike+1 does (delayed-signal case). Comparing by "whichever is higher" was
noise-prone, so instead each candidate is compared against:

```
threshold = mean(area_pct[:spike_idx]) + max(AREA_PCT_SIGMA_MULT * std(...), AREA_PCT_MIN_ELEVATION)
```

- `AREA_PCT_SIGMA_MULT = 10.0`
- `AREA_PCT_MIN_ELEVATION = 1` — floor prevents near-zero baseline std from
  trivially passing noise

---

## eps — physical units to pixels

`compute_eps_px(obj)` converts `EPS_UM` to pixels per objective:

- `EPS_UM = 30.0` µm — maximum gap between two bright pixels that still belong
  to the same release site
- Per `PIXEL_SCALE`: 10X→0.75 px/µm, 40X→3.0 px/µm, 60X→4.5 px/µm
  - 10X: 30µm → 22px
  - 40X: 30µm → 90px
  - 60X: 30µm → 135px

`min_samples` (the old DBSCAN core-point threshold) no longer exists — the
`MIN_CLUSTER_FRACTION` size filter handles minimum cluster size instead.

---

## Cluster size filter — `MIN_CLUSTER_FRACTION`

After clustering, `_run_cluster_seeker` drops undersized components:

```
keep component  if  bright_px_count / total_bright_px  >=  MIN_CLUSTER_FRACTION (0.05)
```

- `total_bright_px` = all bright pixels in the **same frame being clustered**
- Self-calibrating: threshold scales automatically with signal level per recording
- Accepted components are assigned cluster indices `0..N-1`, **largest first**;
  everything else stays at `-1` (noise) in `label_frame`
- Still a guessed number — worked well on test cases, no principled biological
  basis yet

### Rejected alternatives

| Method | Problem |
|--------|---------|
| SNR: `cluster_pct / baseline_pct >= 2x` | baseline can be 0% → denominator collapses |
| Fixed µm² threshold | Does not adapt to objective or signal level |

---

## Saturation guard — removed

The old `SATURATION_AREA_PCT` guard skipped DBSCAN above 10% B% because dense
point clouds caused sklearn's neighbor-graph to balloon in memory (a real 60X
recording once hit 82.5% B% and consumed 73GB+).

The distance-transform approach has no such limitation — cost is O(H×W)
regardless of pixel density — so the saturation guard and its fallback
(`skimage_label` without dilation) were removed entirely. `RegionAnalyzer.saturated`
no longer exists.

---

## Enclosing-circle radius R — `_resolve_R`

R is the **farthest cluster-pixel distance from the centroid** (not RMS),
capped at the centroid's distance to the nearest frame edge:

```
R = min(farthest_pixel_distance, distance_from_centroid_to_nearest_edge)
```

The cap fixes a real overshoot bug: an uncapped R could draw a circle extending
past the frame boundary whenever the centroid isn't near the image center.

---

## Ring-based spreading detection (1 cluster) vs cross-cluster asynchrony (>1 cluster)

### Exactly 1 kept cluster → `compute_ring_traces`

- Split point at **R/√2 ≈ 0.71R** → equal area in both zones for a circular cluster
- Inner zone: pixels at distance `0` to `R/√2`
- Outer zone: pixels at distance `R/√2` to `R`
- Latency = outer-ring peak time − inner-ring peak time (spread *within* the
  one release site)

### More than 1 kept cluster → `compute_cluster_trace`

- One whole-cluster z-score trace per cluster, no ring split
- Latency = latest-peaking cluster's time − earliest-peaking cluster's time
  (asynchrony *between* separate release sites)

---

## Max-area frame — independent second stat

`_compute_max_area` picks whichever of spike/spike+1 has the larger raw B%,
independent of `pick_critical_frame`'s significance-threshold pick.

- Runs `_run_cluster_seeker` on that frame
- `max_area_um2` sums pixels across **all** kept clusters on that frame
- `max_area_eq_radius_um = sqrt(max_area_um2 / π)` — circle-equivalent radius,
  directly comparable to `R_lat_um`

---

## Open TODOs

### `MIN_CLUSTER_FRACTION` / `EPS_UM` biological basis

Still guessed constants (`MIN_CLUSTER_FRACTION=0.05`, `EPS_UM=30.0`). No
principled biological basis yet — validated only against real test recordings.

### Detection gate (`has_event` boolean)

A formal B%-vs-baseline gate deciding *whether* an event occurred at all is
not yet implemented — `pick_critical_frame` only decides *which* frame and
*how* to cluster, not a pass/fail event flag.
