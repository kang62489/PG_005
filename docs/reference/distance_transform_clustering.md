# Distance Transform Clustering — How and Why

## Context

`_run_cluster_seeker` in `classes/region_analyzer.py` groups bright pixels into
clusters. Two pixels belong to the same cluster if they are within `EPS_UM`
micrometres of each other (even with a dark gap between them).

---

## Why Not `binary_dilation + disk`?

The original approach expanded each bright pixel outward by `eps_px` using a
circular kernel (`disk(eps_px)`), then ran connected components on the result.

For every output pixel, `binary_dilation` slides the kernel across the frame
and asks: "does any True cell in the kernel overlap a True cell in bright_mask?"
This requires one full kernel scan per output pixel.

**Cost per pixel** = kernel_width × kernel_height comparisons
**Total cost** = H × W × kernel_width × kernel_height = **O(H × W × eps_px²)**

For 60X with `EPS_UM = 30`:
- `eps_px = 30 × 4.5 = 135 px`
- `disk(135)` is a 271×271 kernel
- On a 1024×1024 frame: 1024 × 1024 × 271 × 271 ≈ **77 billion comparisons** → hangs

Doubling `eps_px` quadruples the work. The cost grows with the objective.

---

## The Fix — `distance_transform_edt`

```python
within_eps_of_bright = distance_transform_edt(~bright_mask) <= eps_px
component_map        = skimage_label(within_eps_of_bright, connectivity=2)
```

### Step 1 — Flip the mask (`~bright_mask`)

`distance_transform_edt` measures distance to the nearest **False** pixel.
Flipping makes bright pixels False so every pixel's computed distance means
"how far am I from the nearest bright pixel?":

```
bright_mask:        ~bright_mask:
0 0 0 0 0           1 1 1 1 1
0 0 B 0 0    →      1 1 0 1 1
0 0 0 0 0           1 1 1 1 1
0 0 0 B 0           1 1 1 0 1
```

### Step 2 — `distance_transform_edt(~bright_mask)`

For every pixel, compute its exact Euclidean distance to the nearest bright pixel
using two linear passes across the frame (horizontal then vertical). Bright pixels
themselves get distance 0.

```
~bright_mask:       distance to nearest bright pixel:
1 1 1 1 1           2.2  1.4  1.0  1.4  2.2
1 1 0 1 1    →      1.4  1.0  0.0  1.0  2.0
1 1 1 1 1           2.2  1.4  1.0  1.4  2.2
1 1 1 0 1           2.0  1.0  1.0  0.0  1.0
```

**Cost: O(H × W)** — two scans across the whole frame, completely independent
of `eps_px`. Doubling `eps_px` costs nothing here.

### Step 3 — `<= eps_px` (threshold)

Keep every pixel within `eps_px` of any bright pixel. With `eps_px = 2`:

```
distances:              <= 2 → within_eps_of_bright:
2.2  1.4  1.0  1.4      F  T  T  T
1.4  1.0  0.0  1.0  →   T  T  T  T
2.2  1.4  1.0  1.4      F  T  T  T
2.0  1.0  1.0  0.0      T  T  T  T
```

This is identical to what `binary_dilation` with `disk(2)` would produce —
every pixel within `eps_px` of a bright pixel is True — but costs only one
comparison per pixel.

---

## Why They Are Equivalent

- `binary_dilation` with disk radius R: pixel P is True if **any bright pixel
  is within R of P**.
- `distance_transform_edt(~mask) <= R`: pixel P is True if its **distance to
  the nearest bright pixel is ≤ R**.

Same condition, different algorithm. ✅

---

## Cost Comparison

| Objective | `eps_px` | `binary_dilation` comparisons | `distance_transform_edt` |
|-----------|----------|-------------------------------|--------------------------|
| 10X       | 22 px    | 1024×1024×45×45 ≈ 2B          | 2×1024×1024 ≈ 2M         |
| 40X       | 90 px    | 1024×1024×181×181 ≈ 34B       | 2×1024×1024 ≈ 2M         |
| 60X       | 135 px   | **1024×1024×271×271 ≈ 77B💥** | 2×1024×1024 ≈ 2M         |

`eps_px` only appears in the final `<= eps_px` threshold step, which is a
trivial per-pixel comparison on the already-computed distance array.

---

## Map Back to Bright Pixels

After connected components on `within_eps_of_bright`, each component is
intersected back with the original `bright_mask`:

```python
bright_px_in_component = (component_map == component_id) & bright_mask
```

This discards the dilation expansion and keeps only original bright pixels —
the expansion was only used to decide which bright pixels belong together.

---

## Weighted Centroid

`_weighted_centroid` is called with `bright_px_coords` derived from
`bright_px_in_component` (bright pixels only). Weights are the z-score values
at those pixel positions — higher z-score pixels pull the centroid toward the
brightest sub-region of the cluster.

Both the pixel set and the weighting are restricted to bright pixels. Dim
pixels never reach this step.
