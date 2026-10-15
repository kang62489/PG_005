# Permutation test for flash sizes (`th_small_obj` from data)

Created 2026-10-15. Script: `permutation_test.py` (root), kernels in `functions/permutation.py`.

Goal: replace the hand-picked `th_small_obj = 5,000 px` with a size we can explain. That size is the one that
chance alone almost never reaches.

---

## 1. Terminology

| term | meaning | example (2025_02_05-0006) |
|---|---|---|
| **th_flashes** | touching above-threshold (background peak + 2σ) pixels in one frame of the **recording**, any size | 774,203 |
| **surrogate** | one shuffled movie (every pixel's trace rotated in time by its own random lag), run through the same blur → ALS → threshold → detection | 1 sur_idx = 1 surrogate |
| **sur_flashes** | touching above-threshold pixels in one frame of a **surrogate**, any size | ~704,000 per surrogate |
| **cutoff** | a size read from the largest sur_flashes, proposed as the new `th_small_obj` | |
| **qf_flashes** | th_flashes with size ≥ cutoff | |

"Noise" is used only for the background σ of the threshold.

```
recording      ──► th_flashes ──(size ≥ cutoff)──► qf_flashes
surrogate      ──► sur_flashes ──► largest sur_flash per surrogate ──► cutoff
```

---

## 2. The hypothesis test

Goal: check that the flashes we detect (threshold + size removal) are **not formed by chance**.

| | H₀ | H₁ |
|---|---|---|
| plain (paper / discussion) | A detected flash is **formed by chance**: its pixels are above threshold at the same moment only by coincidence. | A detected flash is a **coordinated event**: neighbouring pixels are above threshold together more than chance allows. |
| operational (what the shuffle does) | Each pixel keeps its own activity, but its **timing has no relation** to its neighbours' timing. | Neighbouring pixels' timelines are **aligned**: they rise in the same frames. |

The shuffle does **not** reorder frames. Each pixel's timeline is **rotated** by its own lag, so the order within a
pixel is kept and only its timing relative to the neighbours changes:

```
                 f1 f2 f3 f4 f5 f6
recording  A     0  0  5  6  0  0
           B     0  0  5  6  0  0      A, B, C rise in the same frames -> th_flash of 3 px
           C     0  0  5  6  0  0
surrogate  A     0  0  5  6  0  0      lag 0
           B     5  6  0  0  0  0      lag 2: same event, earlier (the end wraps to the start)
           C     0  0  0  0  5  6      lag 4
                                       -> only one pixel active per frame, no 3-px sur_flash
```

| part | in this test |
|---|---|
| **H₀ (null hypothesis)** | Neighbouring pixels are **not** coordinated in time. Any flash comes only from chance overlap of above-threshold pixels (plus the blur). |
| **H₁ (alternative)** | Neighbouring pixels rise **together** (a coordinated release event). |
| **test statistic** | size of a th_flash |
| **reference distribution under H₀** | the largest sur_flash of each of N surrogates |
| **p-value** | probability, if H₀ is true, of getting a flash this big or bigger anywhere in the movie |
| **decision** | p < 0.005 → reject H₀ → **qf_flash** |

### How a surrogate makes H₀ true

1. Detrend + z-score the raw movie (as `img_proc.py`).
2. **Shuffle:** give every pixel its own random lag and rotate its trace by it. With lag 300, frame 0 takes the value
   of frame 300.
   - Each pixel keeps **all its own values** (same fluctuations, same events, same drift shape).
   - Neighbouring pixels are no longer active **at the same time** → spatial coordination is destroyed.
3. Gaussian blur → ALS → threshold → open / close / fill holes → sur_flashes, exactly as for the recording.
   The **recording's own** threshold is used.
   - The shuffle comes **before** the blur, so the blur rebuilds the smooth specks of the recording.

---

## 3. Why only the largest sur_flash

One movie has ~774,000 th_flashes, so that's ~774,000 tests at once.

- Comparing each th_flash with the **largest** sur_flash of each surrogate controls false positives across the
  whole movie at once (the **max-statistic** method). In effect it does the job of a Bonferroni correction, but it is
  less strict.
- If the largest sur_flash is below the cutoff, every other sur_flash is too, so the smaller sizes are not needed.
- Using **all** sur_flash sizes instead (95th percentile of single sur_flashes) would let 5 % × ~704,000 ≈ 35,000
  chance flashes pass per movie.

---

## 4. p-value and cutoff

```
p(s) = (1 + number of surrogates whose largest sur_flash ≥ s) / (N + 1)
cutoff = smallest size s with p(s) < α
th_small_obj = cutoff rounded UP to 500 px      (rounding up only lowers p)
th_flash size ≥ th_small_obj   →   qf_flash
```

- **α (alpha)** is the line chosen beforehand (here 0.005); **p** is computed from the data for each size.
  Rule: p < α → qf_flash.
- Run settings (`permutation_test.py` defaults): N = 1000, α = 0.005 → at most 4 of the 1000 largest sur_flashes
  may reach the cutoff, so cutoff = 5th largest + 1 (p = 5/1001 = 0.004995).

- **"+1"**: counts the recording as one more possible outcome under H₀. This is the standard way to make a Monte
  Carlo p-value valid (p is never exactly 0).
- **Why N = 100**: the smallest possible p is 1 / (N + 1).

| surrogates (N) | smallest possible p |
|---|---|
| 20 | 1/21 = 0.048 (barely < 0.05) |
| 100 | 1/101 = 0.0099 (cannot reach p < 0.005) |
| 1000 | 1/1001 = 0.001 |

### Worked example (2025_02_05-0006, first 20 surrogates only, preliminary)

Largest sur_flash of each surrogate (px), sorted:

```
2703  2807  2886  2902  2911  2934  2978  3016  3047  3078
3111  3253  3272  3280  3395  3446  3493  3681  3751  3784
```

With α = 0.05: cutoff ≈ 3,753 px → 22 th_flashes above it (the first draft used the 95th percentile, which is the
same idea at α = 0.05).

| th_flash size | surrogates with largest ≥ it | p | qf_flash? |
|---|---|---|---|
| 3,000 px | 13 | (1 + 13) / 21 = 0.67 | no |
| 3,700 px | 2 (3751, 3784) | (1 + 2) / 21 = 0.14 | no |
| 5,000 px | 0 | (1 + 0) / 21 = 0.048 | yes |

---

### Choosing α and wording the result

| α | usual reading |
|---|---|
| 0.05 | "significant", the classic default in biology (1 in 20 false alarms) |
| 0.01 | "highly significant" |
| 0.005 | "significant" under the stricter proposal of Benjamin et al. 2018 (*Nat Hum Behav*); 0.005-0.05 = "suggestive" |
| 0.001 | very strong |

α is a convention chosen **before** looking at the data and stated in Methods. This analysis uses α = 0.005.

| result | correct wording | avoid |
|---|---|---|
| p < α | **reject H₀**: the flash is unlikely to come from uncoordinated pixels → evidence for coordinated release | "H₁ is proven" |
| p ≥ α | **fail to reject H₀**: not enough evidence that the flash is more than chance | "H₀ is accepted", "the flash is random" |

A small th_flash with p = 0.3 is not proven to be random. The test cannot tell it apart from chance; it may be a coordinated
but tiny event hidden among the sur_flash sizes.

Example sentence: *"Flashes larger than th_small_obj were unlikely under the null hypothesis of temporally
uncoordinated pixels (p < 0.005, permutation test with 1000 time-lag surrogates per recording)."*

---

## 5. Compared with a t-test

| | t-test | this permutation test |
|---|---|---|
| reference distribution under H₀ | a **formula** (t-distribution) | **built from the data** by shuffling (surrogates) |
| assumptions | normal data, etc. | only: "shifting each trace in time breaks coordination and nothing else" |
| meaning of p | P(result this extreme \| H₀) | the same |

Permutation tests are standard statistics. They are used whenever no formula describes the statistic, like flash
sizes after blur + ALS.

---

## 6. Cautions / open points

- The p-value and cutoff are **per recording**. Combining 103 recordings adds another layer of multiple testing,
  to be handled then.
- The circular lag joins the end of a trace to its start (a "seam"). After biexp detrend + z-score, the two ends
  are at similar levels, so the seam should matter little.
- A per-tile lag (16 × 16 px, the earlier scratch test, shuffled **after** ALS) gave lower cutoffs
  (2,345 / 2,789 / 2,623 px). A likely reason: tile edges cut sur_flashes apart (not verified).
- `permutation_test.py` saves `{stem}_th_flash_sizes.npz`, `{stem}_surrogates.csv` (`sur_idx`,
  `largest_sur_flash_px`) and `summary.csv` (`recording`, `th_small_obj_px`), not a p-value per th_flash.
- Saion: `run_permutation_test_on_saion.slm`, one 10X recording per array task, pixel-wise (`pixel/`) and
  16 × 16 tile backup (`tile16/`), N = 1000 each.
