# Distance Transform Clustering — How and Why

## Context

`_run_cluster_seeker` in `classes/region_analyzer.py` groups bright pixels into
clusters. Two pixels belong to the same cluster if they are within `EPS_UM`
micrometres of each other (even with a dark gap between them).

---

## Why Not `binary_dilation + disk`?

The original approach expanded each bright pixel outward by `eps_px` using a
circular kernel (`disk(eps_px)`), then ran connected components on the result.

**Cost: O(H × W × kernel_size²)**

For 60X with `EPS_UM = 30`:
- `eps_px = 30 × 4.5 = 135 px`
- `disk(135)` is a 271×271 kernel
- On a 1024×1024 frame: 1024 × 1024 × 271 × 271 ≈ **80 billion ops** → hangs

---

## The Fix — `distance_transform_edt`

```python
dilated = distance_transform_edt(~bright_mask) <= eps_px
cc_map   = skimage_label(dilated, connectivity=2)
```

Three steps:

### Step 1 — Flip the mask (`~bright_mask`)

```
bright_mask:        ~bright_mask:
0 0 0 0 0           1 1 1 1 1
0 0 B 0 0    →      1 1 0 1 1
0 0 0 0 0           1 1 1 1 1
0 0 0 B 0           1 1 1 0 1
```

`False` (0) = bright pixel, `True` (1) = everything else.

### Step 2 — `distance_transform_edt(~bright_mask)`

For every pixel, compute its **Euclidean distance to the nearest bright pixel**:

```
~bright_mask:       distance to nearest bright pixel:
1 1 1 1 1           2.2  1.4  1.0  1.4  2.2
1 1 0 1 1    →      1.4  1.0  0.0  1.0  2.0
1 1 1 1 1           2.2  1.4  1.0  1.4  2.2
1 1 1 0 1           2.0  1.0  1.0  0.0  1.0
```

Cost: **O(H × W)** — a single linear-time scan, independent of `eps_px`.

### Step 3 — `<= eps_px` (threshold)

Keep every pixel within `eps_px` of any bright pixel. With `eps_px = 2`:

```
distances:              <= 2 → dilated:
2.2  1.4  1.0  1.4      F  T  T  T
1.4  1.0  0.0  1.0  →   T  T  T  T
2.2  1.4  1.0  1.4      F  T  T  T
2.0  1.0  1.0  0.0      T  T  T  T
```

This is **identical** to what `binary_dilation` with `disk(2)` produces — every
pixel within `eps_px` of a bright pixel is marked True.

---

## Why They Are Equivalent

- `binary_dilation` with a disk of radius R: pixel P is True if **any bright
  pixel is within R of P**.
- `distance_transform_edt(~mask) <= R`: pixel P is True if its **distance to
  the nearest bright pixel is ≤ R**.

These are the same condition. ✅

---

## Cost Comparison

| Objective | `eps_px` | `binary_dilation` kernel | `distance_transform_edt` |
|-----------|----------|--------------------------|--------------------------|
| 10X       | 22 px    | 45×45 — fast             | O(H×W) — flat            |
| 40X       | 90 px    | 181×181 — slow           | O(H×W) — flat            |
| 60X       | 135 px   | **271×271 — explosion**  | O(H×W) — flat            |

---

## Map Back to Bright Pixels

After connected components on the dilated mask, each component is intersected
back with the original `bright_mask`:

```python
comp_mask = (cc_map == comp_label) & bright_mask
```

This ensures only the original bright pixels are counted and labeled — the
dilation was only used to decide which bright pixels belong together.

---

## Weighted Centroid Note

`_weighted_centroid` is called with `comp_coords` derived from `comp_mask`
(bright pixels only). Weights are the z-score values at those bright pixel
positions — so both the pixel set and the weighting are restricted to bright
pixels, with higher z-score pixels pulling the centroid toward the brightest
sub-region.
