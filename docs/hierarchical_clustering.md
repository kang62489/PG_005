# Hierarchical clustering of spontaneous flashes (test)

Created 2026-10-16. Scratch scripts in `output/test_centroid_cluster/` (no pipeline code changed):
`centroid_cluster.py`, `trace_corr_cluster.py`, `silhouette_cut.py` (each takes an optional recording stem).

Goal: replace the current Step 2 grouping of `classes/sp_zone_analyzer.py` (circle-rule units → best-r trace-corr
groups, complete linkage cut at the hand-picked `MIN_GROUP_CORR = 0.95`) with one hierarchical clustering of all
flashes of a recording, and a cut height h chosen from the data.

---

## 1. Input: one row per flash

Flashes = per-frame flashes of the pipeline (Step 1 detect → cleaned mask, `TH_SMALL_OBJ = 5000 px`; Step 2a
same-frame pieces within `CONNECT_RADIUS` merged; flashes > `MAX_FLASH_FRAC` of the frame dropped).

| flash_id | frame_id | centroid_y | centroid_x |
|---|---|---|---|
| 0 | 11 | 160.4 | 597.1 |
| 1 | 12 | 164.7 | 583.4 |
| 2 | 13 | 171.2 | 581.7 |

Centroid = mean (y, x) of the flash's footprint pixels. Exported as `{stem}_flash_centroids.csv`.

Pipeline now (2026-10-19): each recording's own th_small_obj (permutation test) and same-frame blobs joined
within r = √(th_small_obj / π) (section 5), instead of 5,000 px and 75 px.

Mathematica: `Import["..._flash_centroids.csv", "Dataset", "HeaderLines" -> 1]`.

---

## 2. Two distances between flashes

| tree | distance between flash i and j | file |
|---|---|---|
| centroid | Euclidean distance of the centroids, √(Δy² + Δx²), px | `{stem}_flash_centroids.csv` |
| trace | 1 − r, r = Pearson r of the two footprint traces | `{stem}_flash_r.csv` (n × n, no header, order = flash_id) |

Footprint trace = mean ΔF/F over the flash's own footprint pixels, in every frame (pipeline step 2c,
`functions/zone_kernels.footprint_traces`). Time (frame_id) is not used by either distance.

**Why Euclidean** (example: flash 0 ↔ 3, Δy = 15.4, Δx = 33.6 px):

| distance | value | equal-distance shape | note |
|---|---|---|---|
| Euclidean | 37.0 px | circle | straight-line distance in the slice; rotation-invariant |
| Manhattan | 49.0 px | diamond | diagonal pairs look up to √2 farther |
| Maximum | 33.6 px | square | diagonal pairs look up to 1/√2 closer |

The image x/y axes have no biological meaning, so only Euclidean is a physical distance.

---

## 3. How the tree is built (agglomerative, average linkage)

1. Start: every flash is its own cluster (0009: 465 clusters).
2. Merge the two closest clusters. The distance between them is the **merge height** (bar height in the dendrogram).
3. Repeat until one cluster is left (465 flashes → 464 merges).

Distance between two clusters (**linkage**):

| linkage | distance between clusters | effect |
|---|---|---|
| single | nearest pair | chaining: a trail of flashes joins far-apart sites |
| complete | farthest pair | compact; every pair in a group within h; one off-centre flash can split a site |
| **average** (used) | mean of all cross pairs | in between |

Toy example, flashes on a line at A = 0, B = 1, C = 5, D = 12 px:

| merge | candidates (average linkage) | merged | height |
|---|---|---|---|
| 1 | all 6 pairs; A–B = 1 smallest | {A, B} | 1 |
| 2 | {A,B}–C = (5+4)/2 = 4.5; {A,B}–D = (12+11)/2 = 11.5; C–D = 7 | {A, B, C} | 4.5 |
| 3 | {A,B,C}–D = (12+11+7)/3 = 10 | {A, B, C, D} | 10 |

Real checks in 0009 (centroid tree):
- merge 1: flash 239 (frame 647) + 322 (frame 840), 0.14 px apart → height 0.14.
- merge 11: flash 147 joins {32, 236}: (0.608 + 0.412) / 2 = 0.510 px = scipy height.
- merge 464 (top): 182 + 283 flashes, mean of 51,506 cross distances = 525.78 px = scipy height.

The tree always ends in one cluster; it is a record of merges. **Groups come from cutting the tree at h**: all
merges above h are undone.

---

## 4. Choosing h: mean silhouette

For each flash i at a given cut:
- a = mean distance to the other flashes of its own cluster
- b = mean distance to the flashes of the nearest other cluster
- s = (b − a) / max(a, b), from −1 (wrong cluster) to +1 (tight and well separated); a flash alone scores 0

Mean silhouette = mean of s over all flashes. Sweep h, take the h with the highest mean.

Toy example (A = 0, B = 1, C = 5, D = 12 px):

| cut | clusters | s(A), s(B), s(C), s(D) | mean |
|---|---|---|---|
| h = 2 | {A,B} {C} {D} | 0.80, 0.75, 0, 0 | 0.39 |
| h = 6 | {A,B,C} {D} | 0.75, 0.77, 0.36, 0 | **0.47** |
| h = 15 | {A,B,C,D} | not defined | — |

Sweep: centroid h = 5–500 px (step 5), trace 1 − r = 0.01–0.90 (step 0.01). The function matches
`sklearn.metrics.silhouette_score` (0009, h = 110 px: 0.675593 both).

The silhouette is a separation score, **not a p-value**. Other measures, for reference:

| question | measure | 0009 |
|---|---|---|
| is the whole tree a good summary of the distances? | cophenetic correlation | centroid 0.809, trace 0.901 |
| is this branch reliable? | bootstrap AU / BP (`pvclust`; resamples variables → frames for traces, not usable for 2-D centroids) | not run |
| is this merge a big jump? | inconsistency coefficient (scipy `inconsistent`) | not run |

---

## 5. Why the connect radius r = √(th_small_obj / π)

Before clustering, same-frame blobs at most r apart are joined into one flash (`SRE_analysis.py` STEP 2,
`set_connect_radius`).

1. The permutation test gives th_small_obj: the largest blob size that also appears in the time-shuffled
   surrogates. Blobs up to this size can arise by chance. (0009: th_small_obj = 2,100 px)
2. A blob of that size, shaped as a circle, has radius r = √(th_small_obj / π).
   (0009: √(2,100 / π) = 25.9 px -> 25 px, rounded to 5 px)
3. If two blobs in the same frame are at most r apart, the gap between them is no wider than a chance-size
   blob. A missing chance-size piece is enough to explain the gap, so the two blobs are treated as one flash.
4. If the gap is wider than r, a chance-size piece cannot explain it, so the blobs stay separate flashes.

Example, 0009 frame 916 (r = 25 px): gaps 10, 31 and 38.9 px -> only the 10 px pair joins -> 3 flashes.

---

## 6. How the cut height h is chosen, and why the silhouette

How:
1. Build the centroid tree of all flashes (Euclidean centroid distance, average linkage).
2. Cut it at every h from 5 to 500 px in 5 px steps (100 cuts).
3. For each cut, score every flash: a = mean distance to the flashes of its own cluster,
   b = mean distance to the flashes of the nearest other cluster, s = (b − a) / max(a, b),
   from −1 (wrong cluster) to +1 (tight and well separated); a flash alone in its cluster scores 0.
4. Mean silhouette of the cut = mean of s over all flashes.
5. Take the h with the highest mean silhouette (the first one, if several tie). The clusters at that h are
   the zones. (0009: best h = 110 px -> 16 zones, mean silhouette 0.646)

Why:
- h sets how many zones there are; choosing h by hand would choose the zone count.
- The mean silhouette asks directly what a zone should be: flashes close to their own zone and far from the
  others. The highest score is the cut with the clearest separation.
- It is computed per recording, so each recording gets the h that fits its own flashes.
- Limits: it is a separation score, not a p-value; the curve can be flat over a range of h.

Pipeline results (own th_small_obj, connect radius, zones = centroid-tree clusters):

| recording | th_small_obj | r | flashes | best h | zones | mean silhouette |
|---|---|---|---|---|---|---|
| 2025_06_11-0009 | 2,100 px | 25 px | 609 | 110 px | 16 | 0.646 |
| 2025_12_15-0010 | 1,400 px | 20 px | 208 | 145 px | 15 | 0.547 |
| 2026_01_08-0013 | 1,300 px | 20 px | 213 | 100 px | 25 | 0.620 |

---

## 7. Results of the first test (10X GACh3.0, average linkage, th_small_obj 5,000 px, connect radius 75 px)

| recording | flashes | centroid: best h | clusters | silhouette | trace: best 1 − r (r) | clusters | silhouette |
|---|---|---|---|---|---|---|---|
| 2025_06_11-0009 | 465 | 110 px | 13 | 0.676 | 0.32 (0.68) | 9 | 0.812 |
| 2025_12_15-0010 | 93 | 200 px | 5 | 0.697 | 0.12 (0.88) | 12 | 0.797 |
| 2026_01_08-0013 | 109 | 140 px | 12 | 0.700 | 0.33 (0.67) | 12 | 0.790 |
| 2025_11_27-0021 | 0 | — | — | — | — | — | — |

0021 has no flash ≥ 5,000 px (permutation test: 1 qf_flash at th_small_obj 1,200 px), so nothing to cluster.

Observations:
- The trace tree, which uses no position, still gives spatially compact clusters.
- The trace tree scores higher (0.79–0.81) than the centroid tree (0.68–0.70) in all three recordings.
- Both silhouette curves have plateaus; the reported best h is the start of the plateau (0009 centroid: ~0.67 from
  110 to ~175 px; trace: ~0.81 from 0.32 to ~0.44).
- The best h differs per recording (centroid 110–200 px, trace r 0.67–0.88).
- 0013: both trees give 12 clusters with almost the same sites.
- 0010: the centroid best cut (200 px, 5 clusters) joins the whole left side into one cluster; the trace cut
  splits it into ~8 compact sites.
- 0009: the trace tree joins the long column (y ≈ 600–870) into one cluster; the centroid tree splits it into 3.

---

## 8. Open question

How to justify h beyond "best silhouette per recording" (e.g. a fixed h across recordings, a physical size, or a
surrogate comparison). Not decided.

---

## 9. Files (`output/test_centroid_cluster/`)

| file | content |
|---|---|
| `{stem}_flash_centroids.csv` | flash_id, frame_id, centroid_y, centroid_x |
| `{stem}_flash_r.csv` | flash × flash r of the footprint traces |
| `{stem}_dendrogram.png`, `{stem}_cluster_scatter.png` | centroid tree, cuts 25 / 50 / 100 / 200 px |
| `{stem}_trace_corr_dendrogram.png`, `{stem}_trace_corr_scatter.png` | trace tree, cuts r 0.95 / 0.9 / 0.8 / 0.5 |
| `{stem}_silhouette_{centroid,trace_corr}.csv` | cut_height, n_clusters, mean_silhouette |
| `{stem}_silhouette_vs_h.png`, `{stem}_silhouette_best_scatter.png` | silhouette sweep; clusters at the best h |
