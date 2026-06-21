# Why one cluster of close spikes collapses the whole recording's segments

## Step 1 — spikes mapped onto image frames, gaps measured between neighbors

```
frame idx:   ...  9   10  11  12  ...  24  25  26  ...  39  40  41  42  ...  54  55  56  ...
spike?              ●   ●                   ●                   ●   ●                   ●
                    └─┬─┘                   │                   └─┬─┘                   │
                  gap = 0               gap = 13               gap = 0               gap = 13
                (collision!)          (plenty of room)        (collision!)          (plenty of room)
```

Two pairs of spikes (`10↔11` and `40↔41`) fired closer together than one
imaging frame can resolve — `gap = 0`, i.e. zero usable baseline frames
between them. The spike at `25` and `55` are fine, with 13–14 clean frames
of room on each side.

## Step 2 — one number summarizes the WHOLE file

```
all gaps measured in this recording:   [ 0, 13, 14, 0, 13, ... ]
                                          ▲       ▲
                                          └───┬───┘
                                    most common value (mode) = 0

        set_interval_frames = min(mode, 20) = 0   ← ONE shared value
```

`set_interval_frames` is computed **once per recording**, from the *mode*
of every spike's gap — not per spike. If `0` happens to be the most
frequent gap (even from just a couple of colliding pairs), the file-wide
value becomes `0`.

## Step 3 — that one value is broadcast to EVERY segment, even the healthy ones

```
                 segment built using shared set_interval_frames = 0
                 ────────────────────────────────────────────────
spike @ 10  (gap=0)   →  [10:10]   1 frame   ✗ no baseline (expected — it really has no room)
spike @ 25  (gap=14)  →  [25:25]   1 frame   ✗ no baseline (BUT it actually had 14 frames free!)
spike @ 40  (gap=0)   →  [40:40]   1 frame   ✗ no baseline (expected)
spike @ 55  (gap=13)  →  [55:55]   1 frame   ✗ no baseline (BUT it actually had 13 frames free!)
```

Every segment in the file gets clipped to a single frame — including spikes
that had plenty of clean baseline available — because they all share the
same file-wide window size.

## Step 4 — what crashed, and the fix

```
1-frame segment ──▶ zscore_img_segs(): baseline = segment[:0] = []  (silently NaN, no crash)
                ──▶ SpatialCategorizer: source_frames[:0] = []
                       np.concatenate([])  ──▶  ValueError  (the actual crash)
```

Fix (`classes/abf_clip.py:184`): when `set_interval_frames < 1`, mark every
spike in the entry as skipped instead of building 1-frame segments —
the entry is logged as "no valid segments" and the pipeline moves on,
instead of reaching the empty-baseline crash.
