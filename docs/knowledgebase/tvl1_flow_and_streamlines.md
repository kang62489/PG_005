---
keywords: optical flow, TV-L1, total variation, L1 data term, pyramid, warp, dual projection, flow pattern, source, sink, anisotropic, drift, spread, divergence, least squares, streamlines, streamplot
files_referenced: functions/tvl1_flow.py, functions/flash_flow.py, functions/flow_pattern.py, functions/plot_results.py
related: paper_claims.md
---

# 2026-09-27

## TV-L1 optical flow, flow-pattern labels and streamlines — step by step

Used for claim 3 (ACh stays local). Figures: `results/flow/*_FLOW.png`, `*_STREAMLINES.png`; table `flow_pairs`.

---

## 1. What optical flow answers

For every pixel: how far did the image content move from frame A to frame B?
Output = two arrays of the image size: `u` (px/frame, + = right), `v` (px/frame, + = down).
Pairs used: spike-1→spike, spike→spike+1, …, spike+3→spike+4 on the MED stack (`FLOW_OFFSETS`).

## 2. What "TV-L1" means

It minimises

```
E = Σ λ · | B(x + u) − A(x) |   +   Σ ( |∇u| + |∇v| )
      └──── L1 data term ────┘      └── TV (total variation) ──┘
```

- L1 data term: moved B should look like A; absolute difference -> robust to noisy pixels.
- TV term: neighbours move alike, but sharp edges are allowed.
- λ = `ATTACHMENT` = 15, θ = `TIGHTNESS` = 0.3.

In one line: **L1 = "follow your own data"; TV = "agree with your neighbours".** They take turns until they balance.

## 3. Worked example on two 3 × 3 frames

All rows identical -> nothing moves vertically (v stays 0); follow one row and the horizontal shift u.

```
        A (frame 1)            B (frame 2)
     x=0  x=1  x=2          x=0  x=1  x=2
   [ 10   20   30 ]       [  0   10   25 ]      (roughly +1 px to the right;
   [ 10   20   30 ]       [  0   10   25 ]       x=2 is "messy": 25, not 20)
   [ 10   20   30 ]       [  0   10   25 ]
```

The algorithm cannot "see" the shift; it starts from the neutral guess **u = 0** everywhere and corrects it.

### 3a. Slope g of B (`_grad_terms`, same as `np.gradient`)

| Pixel | Rule | Calculation | g |
|---|---|---|---|
| x=0 (edge) | right − self | 10 − 0 | 10 |
| x=1 (middle) | (right − left) / 2 | (25 − 0) / 2 | 12.5 |
| x=2 (edge) | self − left | 25 − 10 | 15 |

### 3b. L1 data step — each pixel follows its own data (`_data_step`)

Mismatch ρ = B(x + u) − A(x). Reading B "between pixels" uses the shortcut B(x + u) ≈ B(x) + g·u, so

```
ρ(u) = B(x) − A(x) + g·u        new u = u − ρ / g   (the u where ρ = 0)
```

| Pixel | ρ at u = 0 | New u = −ρ / g |
|---|---|---|
| x=0 | 0 − 10 = −10 | 10 / 10 = **1.00** |
| x=1 | 10 − 20 = −10 | 10 / 12.5 = **0.80** |
| x=2 | 25 − 30 = −5 | 5 / 15 = **0.33** |

u is one value per pixel: every row now reads `[1.00, 0.80, 0.33]`; each arrow is (u, v) = e.g. (1.00, 0).
Check x=2: B ≈ 25 + 15 × 0.333 = 30 = A ✅.
(L1 part: if |ρ| > λ·θ·g² the step is capped at λ·θ·g — one bad pixel cannot push u arbitrarily far.)

### 3c. TV step — neighbours pull toward each other (`_proj_step` + `_div_step`)

Constants: **dt = 0.25** (step size, fixed in skimage for 2-D), **f1 = 0.25 / θ = 0.833** (brake).

1. Difference to the right neighbour: 0→1: 0.80 − 1.00 = −0.20; 1→2: 0.33 − 0.80 = −0.47
2. Amount on each link: `p = −dt × diff / (1 + f1 × |diff|)`
   - p(0→1) = 0.050 / 1.167 = **0.043**
   - p(1→2) = 0.117 / 1.389 = **0.084**
3. **p is like water along a link a → b: the start pixel loses p, the end pixel gains p.**

```
        p(0→1) = 0.043        p(1→2) = 0.084
  x=0  ───────────────▶  x=1  ───────────────▶  x=2
  −0.043                +0.043   −0.084          +0.084
```

| Pixel | Gains from | Loses to | New u |
|---|---|---|---|
| x=0 | — | 0→1 | 1.00 − 0.043 = **0.957** |
| x=1 | 0→1 | 1→2 | 0.80 + 0.043 − 0.084 = **0.759** |
| x=2 | 1→2 | — | 0.33 + 0.084 = **0.417** |

- The total is unchanged (2.13): TV only redistributes.
- A negative p means the water flows right → left; it always flows from high u to low u.
- The brake keeps edges: diff 0.2 passes 21 % of it, diff 2.0 passes only 9 %.
- In 2-D each pixel also has a **down** link (y → y+1): gains from left and above, loses to right and below.

### 3d. Next round — L1 pushes back

Pixel x=2 at u = 0.417: ρ = −5 + 15 × 0.417 = +1.25 (overshoot) -> u = 0.417 − 1.25 / 15 = 0.333 (back to its own data).
TV then pulls it toward its neighbours again (~0.42), and so on.

| Pixel type | Who wins |
|---|---|
| strong edge / texture (clear data) | L1 -> shift follows the data |
| flat / empty (no data, g ≈ 0) | TV -> shift copied from neighbours |
| noisy single pixel | TV smooths it toward neighbours |

## 4. Earlier 1-D toys

- Blob `A = [0, 5, 10, 5, 0, 0]` -> `B = [0, 0, 5, 10, 5, 0]`, pixel x = 2: ρ = 5 − 10 = −5, g = 5, u = 5 / 5 = **+1**.
- Noisy flow `[0, 0, 1, 0, 0]` after one TV step -> `[0, 0.136, 0.727, 0.136, 0]` (the spike is shared, sum stays 1).

## 5. Full solver loop

| Loop | Count | Why |
|---|---|---|
| Pyramid levels | 6 (32 → … → 1024 px) | big moves look small at coarse size |
| Warps per level | 5 | re-sample B with current flow, re-linearise |
| Iterations per warp | 10 × (data step + 2 TV steps) | alternate fit and smoothing |

Flow is upscaled ×2 (values doubled) between levels. The numba port is bit-identical to skimage.
Each warp re-reads B at x + u for real (bilinear interpolation), replacing the slope shortcut of 3b.

## 6. Real code on toy blobs

- Blob moved +2 px: u ≈ 2.00 at the blob, v ≈ 0; **u = 1.61 in the empty far corner** -> TV fills flat background with neighbours' flow, so flow is only trusted inside the CAT mask.
- Blob grows (σ 3 → 4): u from −1.99 (left edge) to +1.99 (right edge) -> spreading out.

## 7. Source / sink / anisotropic (`flow_pattern.py`)

1. Block-average u, v in 8 × 8 px blocks; keep blocks inside the CAT flash.
2. Least-squares fit around the centroid: `u ≈ a11·(x−cx) + a12·(y−cy) + bu`, `v ≈ a21·(x−cx) + a22·(y−cy) + bv`.
3. spread = |a11 + a22| / 2 × r_rms; drift = √(bu² + bv²).
4. drift > spread -> anisotropic; else trace > 0 -> source, trace < 0 -> sink.

Hand example (4 blocks, 8 px from centre):

| Block (x−cx, y−cy) | u | v |
|---|---|---|
| (−8, 0) | −0.7 | 0 |
| (+8, 0) | +1.3 | 0 |
| (0, −8) | +0.3 | −1 |
| (0, +8) | +0.3 | +1 |

a11 = a22 = 0.125 -> trace 0.25; bu = 0.3 -> drift 0.3; r_rms = 8 -> spread 1.0 -> **source**.

Real code on toys (64 × 64): grows -> source (drift 0.20, spread 4.14); shifts +4 px -> anisotropic 0° (3.97, 0.00); shrinks -> sink (0.10, 3.17).
Anisotropic angles are converted to DV / ML (e.g. `L 25° D`) with the dorsal / medial vectors from `data/bd_20260922_000.json`.

## 8. Streamlines (`plot_results.py`, matplotlib `streamplot`)

A quiver = one arrow per sample point. A streamline **connects the arrows into a line**: start at a seed, follow the arrow, look again where you landed, repeat — as if the flow of that one frame pair were frozen.

Streamlines add **no new information**: they only draw the TV-L1 (u, v), **binned** into 8 × 8 px block means.

```
MED frame pair
   └─► TV-L1 (u, v) per pixel ─┬─► quivers      : sample one pixel every 24 px
                               ├─► speed map    : √(u² + v²) per pixel, µm/s
                               └─► 8 × 8 block mean ─┬─► streamlines  (matplotlib streamplot)
                                                     └─► pattern label (fit_flow_pattern)
```

### 8a. Who does what

| Part | Done by |
|---|---|
| TV-L1 flow (u, v) | our code: `functions/tvl1_flow.py` |
| 8 × 8 binning + CAT mask | our code: `block_mean()` in `functions/flow_pattern.py` |
| Seeds, tracing, stopping, spacing, arrowheads | **matplotlib** `Axes.streamplot` (no custom tracing code) |

```python
u_small = block_mean(pair["u"])
v_small = block_mean(pair["v"])
keep_small = block_mean(region.astype(float)) > 0
ax.streamplot(xs, ys, np.ma.masked_where(~keep_small, u_small), np.ma.masked_where(~keep_small, v_small),
              color="red", density=FLOW_STREAM_DENSITY, linewidth=0.8, arrowsize=1.2)
```

### 8b. Tracing one line — same 3 × 3 example (flow after TV, section 3c)

Each "pixel" here stands for one 8 × 8 block.

```
u:  every row [0.96  0.76  0.42]        v: all 0
```

Each step: (1) read the arrow, mixing neighbours between pixels (bilinear); (2) **keep only the direction** (divide by length); (3) step 0.5 px.

| Step | Position (x, y) | u there | Direction | Next |
|---|---|---|---|---|
| 1 | (0.0, 1) | 0.96 | (1, 0) → | (0.5, 1) |
| 2 | (0.5, 1) | (0.96 + 0.76) / 2 = 0.86 | (1, 0) → | (1.0, 1) |
| 3 | (1.0, 1) | 0.76 | (1, 0) → | (1.5, 1) |
| 4 | (1.5, 1) | (0.76 + 0.42) / 2 = 0.59 | (1, 0) → | (2.0, 1) |
| 5 | (2.0, 1) | 0.42 | (1, 0) → | leaves grid, stop |

Result: a straight line with one arrowhead in the middle pointing right.
u fell from 0.96 to 0.42, yet the line looks the same as for constant u -> **a streamline shows direction only, not speed** (use `FLOW.png` row 3 for speed).

### 8c. A 2-D source, so lines can curve

```
u:  [ −1  0  +1 ]        v (+ = down):  [ −1  −1  −1 ]
    [ −1  0  +1 ]                       [  0   0   0 ]
    [ −1  0  +1 ]                       [ +1  +1  +1 ]
```

Seed (x = 1.5, y = 0.5), between pixels (1,0), (2,0), (1,1), (2,1):
- u = (0 + 1 + 0 + 1) / 4 = 0.5; v = (−1 − 1 + 0 + 0) / 4 = −0.5
- length √(0.5² + 0.5²) = 0.707 -> direction (0.71, −0.71), up-right ↗
- step 0.5 px -> (1.85, 0.15) -> next step leaves the top-right corner -> stop
- matplotlib also traces **backwards** from the seed: toward the centre (1, 1), where u = v = 0 -> speed 0 -> stop

Full line: centre -> up-right corner. Other seeds give centre -> right, centre -> down-left, …

```
   ↖   ↑   ↗
   ←   •   →        source = lines radiate from the centre
   ↙   ↓   ↘
```

Sink = all arrows flipped (lines run into the centre); anisotropic = parallel lines going one way.

### 8d. Rules of the real figure (matplotlib defaults, except density)

| Rule | What it does |
|---|---|
| Seeds | panel divided into ≈ 75 × 75 cells (`density = 2.5`); lines start from free cells |
| No crowding | each line marks the cells it passes; a new line stops on entering an occupied cell -> even spacing |
| Mask | blocks outside the CAT flash are masked -> lines stop at the flash edge (row 2) |
| Stop at zero | speed 0 -> stop (e.g. centre of a source) |
| Steps | small adaptive steps (2nd-order Runge–Kutta), same idea as the 0.5 px demo |
| Arrowhead | one per line, at its middle, pointing along the flow (`arrowsize = 1.2`) |

A streamline is the shape of **one** frame pair's flow field, not a molecule's path over time. Colour does not encode speed.
The STREAMLINES titles carry the source / sink / anisotropic label: lines are the visual, the label comes from the 8 × 8 block fit (section 7).

## 9. How 1024 × 1024 vectors become a figure

Every pixel has its own (u, v) (~1 million arrows). Each output reduces them differently:

| Output | Reduction | Code |
|---|---|---|
| Quivers (`FLOW.png` rows 1–2) | **sampling**: the vector of one pixel every 24 px (43 × 43 = 1,849 arrows) | `FLOW_QUIVER_STEP = 24` |
| Streamlines | **binning**: mean of each 8 × 8 px block | `block_mean()`, `FLOW_PATTERN_BLOCK = 8` |
| Pattern label | **binning**: same 8 × 8 block means | `fit_flow_pattern()` |
| Speed map (`FLOW.png` row 3) | none: every pixel, µm/s | `imshow(speed)` |

- The quiver arrow at (x = 24, y = 48) is exactly `u[48, 24], v[48, 24]`, not a 24 × 24 average (TV makes neighbours similar, so one pixel is representative).
- Row 2 draws only grid points within 24 px of the CAT flash, so thin flashes are not missed.
- Arrow **length is auto-scaled per panel** (95th-percentile arrow drawn 0.9 × 24 ≈ 22 px long): compare lengths within a panel only. For absolute speed use row 3 (shared colour scale).
