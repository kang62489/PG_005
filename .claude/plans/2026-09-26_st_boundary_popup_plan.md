# Striatum boundary popup -- TODO K (2026-09-26)

Scope: new GUI popout "Striatum Boundary" + headless helpers + `data/st_bd.json`.
Executed phase by phase -- user checks after each phase (assembly-line style). No code is touched before approval.
Consumers (coverage in `spontaneous_analysis.py`, anatomical direction in flow analysis) are a LATER, separate plan.

## Decisions (user, 2026-09-26)

| # | Decision |
|---|---|
| Q1 | Preview = mean of the first 50 frames of the RAW tiff (anatomy visible; ALS is detrended); only the first pages are read |
| Q2 | Anchor tool (replaced freehand, user 2026-09-26): left-click = anchor, double-click = finish line, drag = move any anchor; centripetal Catmull-Rom curve through the anchors, ends snapped to the frame edge / an earlier line; lines split the frame into regions; right-click = striatum region, Shift + right-click = cortex region (optional). JSON keeps `anchors_px` (for editing) + `boundaries_px` (sampled curve) |
| Q3 | User sets only `Dorsal is: up / down / left / right`. Medial from `rec_data.db` `SLICE` hemisphere: L -> dorsal turned 90° clockwise, R -> counter-clockwise (dorsal up + L -> medial right). `SLICE` without L / R -> manual medial dropdown |
| Q3b | 10X cross-check (only when a cortex region is clicked): cortex centroid must lie on the lateral side of the striatum centroid, else a warning (non-blocking) |
| Q4 | One file `data/st_bd.json`; every Confirm writes it immediately (no Export button) |
| Q5 | No contrast slider, no ROI preview PNG |
| Q6 | All recordings are coronal -> DV / ML only |
| Q7 | Per recording: every recording (10X / 40X / 60X) gets a direction; boundary only on 10X |
| Q8 | Two lists (Unchecked / Confirmed), single selection, Confirm auto-selects the next unchecked recording |
| Q9 | 10X can be confirmed only with a boundary (>= 2 regions + striatum picked) |
| Q10 | Draft renamed `data/st_bd_draft.json` (autosave per Confirm). Export button (enabled at Unchecked 0) -> `bd_{date}_{serial}.json` next to the proc list (`proc_20260922_000[_saion].txt` -> `bd_20260922_000.json`): per recording `dorsal` / `medial` + `dorsal_vec` / `medial_vec` (x, y), 10X also `striatum_area_px` + `striatum_outline_px` (closed polygon, `outline_mask()` rebuilds the mask, IoU >= 0.9999). Exported recordings are removed from the draft |
| Q11 | "Finish line" button ends a line (double-click no longer finishes; a fast double-click = one anchor) |

---

## `data/st_bd.json` layout (flat, one entry per recording)

```json
{
  "2024_10_11-0009": {
    "obj": "10X", "slice": "2R", "image_shape": [1024, 1024],
    "dorsal": "up", "medial": "left",
    "boundaries_px": [[[610.0, 0.0], [598.2, 21.4], "..."], [["..."]]],
    "striatum_seed_px": [300, 500],
    "cortex_seed_px": [850, 400],
    "saved": "2026-09-26T13:05"
  },
  "2024_10_11-0013": {
    "obj": "60X", "slice": "2R", "image_shape": [1024, 1024],
    "dorsal": "left", "medial": "down",
    "saved": "2026-09-26T13:07"
  }
}
```

- Coordinates `[x, y]` = `[column, row]`, origin top-left (same as the tiff).
- `boundaries_px`: one list per line, already smoothed + snapped (~50 points each). `cortex_seed_px` may be null.
- The striatum mask is rebuilt from lines + seed (`region_masks()`), not stored.

---

## Phase 1 -- Popup skeleton + lists + preview + Confirm (DONE)

## Phase 2 + 3 -- Drawing, regions, 10X Confirm

`functions/st_boundary.py`:
- `smooth_stroke(points) -> np.ndarray` -- drop repeated points, `splprep` smoothing spline
  (`s = n × SMOOTH_PX²`), resample to `N_LINE_POINTS = 50`
- `snap_ends(line, shape, other_lines) -> np.ndarray` -- extend each end along its direction up to `SNAP_PX = 30`
  until it leaves the frame (clipped to the edge) or hits another line
- `label_regions(lines, shape) -> (labels, n_big)` -- lines rasterized (`skimage.draw.line`), 4-connected
  labelling of the rest; `n_big` = regions >= `MIN_REGION_FRAC = 1 %` of the frame
- `lateral_check(striatum_mask, cortex_mask, dorsal, medial) -> str | None` -- warning text or None

Controller (matplotlib `button_press` / `motion_notify` / `button_release`, 10X only):
- left drag = live red stroke; release -> smooth -> snap -> stored line (red dashed)
- right-click = striatum region (green tint), Shift + right-click = cortex region (red tint)
- Undo line / Clear buttons; status + warning in the canvas title
- Confirm enabled for 10X once >= 2 regions and a striatum region are set; reselecting reloads lines + seeds

Verify: draw 1-line and 2-line boundaries on 2 recordings, check snapping / regions / cross-check, confirm, reload.

## Later (separate plan)

- Coverage: zone area inside the striatum mask / striatum area (`spontaneous_analysis.py` summary).
- Flow direction in anatomical axes (DV / ML) for `flow_pairs`, read from `data/st_bd.json`.
