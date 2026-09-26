"""
st_boundary.py  --  Striatum boundary / slice orientation helpers (headless, used by the Striatum Boundary popup).

  Step 1. Orientation : dorsal direction + hemisphere (SLICE "3L" / "2R") -> medial direction;
                        image angle -> DV / ML pole + tilt (flow drift)
  Step 2. Preview     : mean of the first N pages of a raw TIFF (anatomy visible, no full 2.5 GB load)
  Step 3. Boundary    : clicked anchors -> Catmull-Rom curve -> ends snapped -> frame regions -> checks
  Step 4. Storage     : data/st_bd_draft.json (working draft, one entry per recording stem)
  Step 5. Export      : draft -> data/bd_{date}_{serial}.json (orientation vectors + striatum outline) per proc list
  Step 6. Re-edit     : exported entry -> draft entry (anchors as saved, or rebuilt from the outline)

Points are (x, y) = (column, row), origin top-left, same as the TIFF.

Example:
    curve = snap_ends(anchor_curve(anchors), (1024, 1024), earlier_lines)
    labels, n_regions = label_regions([curve], (1024, 1024))
    striatum = outline_mask(bd["recordings"][stem]["striatum_outline_px"], (1024, 1024))  # reading an export
"""

## Modules
# Standard library imports
import json
import re
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt
from skimage.draw import line as draw_line
from skimage.draw import polygon
from skimage.measure import approximate_polygon, find_contours, label

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 1: orientation ---
DIRECTIONS = ("up", "right", "down", "left")  # clockwise order on the image
UNIT_VECTORS = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}  # (x, y), y grows downwards

# --- Step 2: preview ---
PREVIEW_FRAMES = 50  # frames averaged from the start of the raw TIFF (~0.25 s to read)

# --- Step 3: boundary ---
POINTS_PER_SEGMENT = 20  # curve samples between two neighbouring anchors
CATMULL_ROM_ALPHA = 0.5  # centripetal: no loops / overshoot with uneven anchor spacing
SNAP_PX = 30  # a line end this close to the frame edge / another line is extended to touch it
END_DIRECTION_POINTS = 5  # the end direction is taken over this many points
MIN_REGION_PX = 100  # px: smaller pieces don't count as regions (a curve looping on itself, ~20 px)
MIN_CENTROID_FRAC = 0.05  # cross-check skipped if the centroids differ by < 5 % of the frame along ML

# --- Step 5: export ---
OUTLINE_TOLERANCE_PX = 0.5  # outline simplification: max deviation from the pixel border (px)

# --- Step 6: re-edit ---
ANCHOR_TOLERANCE_PX = 2.0  # export without anchors: outline -> anchors, max deviation (px); ~12 anchors per line
SEED_DOWNSAMPLE = 8  # the rebuilt striatum seed is placed on an 8x smaller mask (a full-size one takes ~0.27 s)


# ===========================================================================
#
#   STEP 1 -- ORIENTATION: dorsal + hemisphere -> medial
#
# ===========================================================================

def hemisphere_of(slice_label: str | None) -> str | None:
    """'3L' -> 'L', '2R' -> 'R', anything else (e.g. '3', None) -> None."""
    text = str(slice_label or "").strip().upper()
    return text[-1] if text[-1:] in ("L", "R") else None


def perpendicular_of(dorsal: str) -> tuple[str, str]:
    """The two directions at 90° to dorsal (the possible medial choices)."""
    i = DIRECTIONS.index(dorsal)
    return DIRECTIONS[(i + 1) % 4], DIRECTIONS[(i - 1) % 4]


def medial_from(dorsal: str, slice_label: str | None) -> str | None:
    """L slice -> dorsal turned 90° clockwise, R slice -> counter-clockwise (dorsal up + L -> right)."""
    hemisphere = hemisphere_of(slice_label)
    if hemisphere is None:
        return None
    clockwise, counter_clockwise = perpendicular_of(dorsal)
    return clockwise if hemisphere == "L" else counter_clockwise


def direction_labels(dorsal: str, medial: str) -> tuple[str, str]:
    """(x label, y label) naming the anatomy at each image side, e.g. ('← medial · lateral →', '← ventral · dorsal →').

    The y label is drawn rotated 90° (reads bottom -> top), so its '←' points down and '→' points up.
    """
    opposite = {d: DIRECTIONS[(i + 2) % 4] for i, d in enumerate(DIRECTIONS)}
    name = {dorsal: "dorsal", opposite[dorsal]: "ventral", medial: "medial", opposite[medial]: "lateral"}
    return f"← {name['left']}  ·  {name['right']} →", f"← {name['down']}  ·  {name['up']} →"


def dv_ml_direction(angle_deg: float, dorsal_vec: list, medial_vec: list) -> tuple[str, float, str | None]:
    """Image angle (0 = right, 90 = up) -> (nearest pole, tilt <= 45°, adjacent pole it tilts toward).

    Poles D / V / M / L come from the export's (x, y) unit vectors (y grows downwards).
    E.g. dorsal right + medial up, 295° -> ('L', 25.3, 'D') = 'L 25° D'; toward is None when tilt is 0.
    """
    def image_angle(vec: list) -> float:
        return float(np.degrees(np.arctan2(-vec[1], vec[0])) % 360)

    d_angle, m_angle = image_angle(dorsal_vec), image_angle(medial_vec)
    poles = {"D": d_angle, "V": (d_angle + 180) % 360, "M": m_angle, "L": (m_angle + 180) % 360}
    signed = {p: (angle_deg - a + 180) % 360 - 180 for p, a in poles.items()}  # -180..180, + = counter-clockwise
    pole = min(signed, key=lambda p: abs(signed[p]))
    tilt = abs(signed[pole])
    if tilt == 0:
        return pole, 0.0, None
    toward = next(p for p in poles if p != pole and abs((poles[p] - poles[pole] + 180) % 360 - 180) == 90
                  and ((poles[p] - poles[pole]) % 360 == 90) == (signed[pole] > 0))
    return pole, tilt, toward


# ===========================================================================
#
#   STEP 2 -- PREVIEW: mean of the first pages of a raw TIFF
#
# ===========================================================================

def raw_preview(raw_tiff_path: Path, n_frames: int = PREVIEW_FRAMES) -> np.ndarray:
    """Mean of the first n_frames pages -> (H, W) float32."""
    with tifffile.TiffFile(raw_tiff_path) as tif:
        n = min(n_frames, len(tif.pages))
        return np.mean([tif.pages[i].asarray() for i in range(n)], axis=0, dtype=np.float32)


# ===========================================================================
#
#   STEP 3 -- BOUNDARY: anchors -> curve -> snapped line -> regions -> checks
#
# ===========================================================================

# --- 3a. anchors -> curve ---

def _cr_segment(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Centripetal Catmull-Rom from p1 to p2 (Barry-Goldman form), POINTS_PER_SEGMENT samples, p2 excluded."""
    t1 = np.linalg.norm(p1 - p0) ** CATMULL_ROM_ALPHA
    t2 = t1 + np.linalg.norm(p2 - p1) ** CATMULL_ROM_ALPHA
    t3 = t2 + np.linalg.norm(p3 - p2) ** CATMULL_ROM_ALPHA
    t = np.linspace(t1, t2, POINTS_PER_SEGMENT, endpoint=False)[:, None]
    a1 = ((t1 - t) * p0 + t * p1) / t1
    a2 = ((t2 - t) * p1 + (t - t1) * p2) / (t2 - t1)
    a3 = ((t3 - t) * p2 + (t - t2) * p3) / (t3 - t2)
    b1 = ((t2 - t) * a1 + t * a2) / t2
    b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
    return ((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1)


def anchor_curve(anchors: np.ndarray) -> np.ndarray:
    """Smooth curve through every anchor (M, 2); the end tangents follow the first / last anchor pair."""
    keep = np.r_[True, np.any(np.diff(anchors, axis=0) != 0, axis=1)]  # a double-click repeats the last anchor
    anchors = np.asarray(anchors, dtype=float)[keep]
    if len(anchors) < 3:  # 1-2 anchors: the point / straight segment itself
        return anchors
    padded = np.vstack([2 * anchors[0] - anchors[1], anchors, 2 * anchors[-1] - anchors[-2]])  # phantom ends
    segments = [_cr_segment(*padded[i : i + 4]) for i in range(len(anchors) - 1)]
    return np.vstack([*segments, anchors[-1:]])


# --- 3b. snap line ends ---

def rasterize_lines(lines: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Bool (H, W) mask of the line pixels (8-connected chains, so they block 4-connected regions)."""
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for line in lines:
        pts = np.rint(line).astype(int)
        pts[:, 0] = pts[:, 0].clip(0, width - 1)
        pts[:, 1] = pts[:, 1].clip(0, height - 1)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=True):
            rr, cc = draw_line(y0, x0, y1, x1)
            mask[rr, cc] = True
    return mask


def _extend_end(end: np.ndarray, direction: np.ndarray, shape: tuple[int, int], others: np.ndarray) -> np.ndarray | None:
    """Walk from end along direction for up to SNAP_PX px; the point where it leaves the frame or hits another line."""
    height, width = shape
    for step in range(1, SNAP_PX + 1):
        x, y = end + direction * step
        if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
            return np.array([x.clip(0, width - 1), y.clip(0, height - 1)])
        if others[int(round(y)), int(round(x))]:
            return np.array([x, y])
    return None


def snap_ends(line: np.ndarray, shape: tuple[int, int], other_lines: list[np.ndarray]) -> np.ndarray:
    """Extend both ends of a line to the frame edge or another line, if within SNAP_PX."""
    height, width = shape
    line = line.copy()
    line[:, 0] = line[:, 0].clip(0, width - 1)
    line[:, 1] = line[:, 1].clip(0, height - 1)
    if len(line) < 2:
        return line
    others = rasterize_lines(other_lines, shape)
    k = min(END_DIRECTION_POINTS, len(line) - 1)
    ends = []
    for end, inner in ((line[0], line[k]), (line[-1], line[-1 - k])):
        direction = end - inner
        norm = np.hypot(*direction)
        ends.append(None if norm == 0 else _extend_end(end, direction / norm, shape, others))
    start, stop = ends
    parts = [line]
    if start is not None:
        parts.insert(0, start[None])
    if stop is not None:
        parts.append(stop[None])
    return np.vstack(parts)


# --- 3c. regions + side check ---

def label_regions(lines: list[np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Regions the lines cut the frame into: (labels, 0 on line pixels; number of regions >= MIN_REGION_PX)."""
    labels = label(~rasterize_lines(lines, shape), connectivity=1)
    areas = np.bincount(labels.ravel())[1:]
    return labels, int((areas >= MIN_REGION_PX).sum())


def region_at(labels: np.ndarray, seed: tuple[int, int] | None) -> np.ndarray | None:
    """Bool mask of the region containing seed (x, y); None if no seed, on a line, or a too-small region."""
    if seed is None:
        return None
    region = labels[seed[1], seed[0]]
    if region == 0:
        return None
    mask = labels == region
    return mask if mask.sum() >= MIN_REGION_PX else None


def lateral_check(striatum: np.ndarray, cortex: np.ndarray, medial: str) -> str | None:
    """Cortex must lie lateral (opposite of medial) to the striatum; returns a warning text or None."""
    lateral = np.array(UNIT_VECTORS[medial]) * -1
    (sy, sx), (cy, cx) = (np.argwhere(striatum).mean(axis=0), np.argwhere(cortex).mean(axis=0))
    offset = float(np.dot([cx - sx, cy - sy], lateral))
    if abs(offset) < MIN_CENTROID_FRAC * max(striatum.shape):
        return "cortex is neither clearly lateral nor medial to the striatum -- check skipped"
    if offset < 0:
        return "cortex lies MEDIAL to the striatum -- check dorsal / slice side"
    return None


# ===========================================================================
#
#   STEP 4 -- STORAGE: data/st_bd_draft.json
#
# ===========================================================================

_XY_PAIR = re.compile(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]")  # an indented [x, y] pair spread over 4 lines


def load_st_bd(path: Path) -> dict[str, dict]:
    """{stem: entry}; empty if the file does not exist yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_st_bd(path: Path, data: dict[str, dict]) -> None:
    """Write sorted by key, [x, y] pairs on one line, via a temp file so a crash never leaves a half-written JSON."""
    text = _XY_PAIR.sub(r"[\1, \2]", json.dumps(dict(sorted(data.items())), indent=2))
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


# ===========================================================================
#
#   STEP 5 -- EXPORT: draft entries -> data/bd_{date}_{serial}.json
#
# ===========================================================================

# --- 5a. file name + mask <-> outline ---

def bd_export_path(proc_list_path: Path) -> Path:
    """proc_20260922_000.txt (or ..._saion.txt) -> <same folder>/bd_20260922_000.json."""
    match = re.search(r"\d{8}_\d{3}", proc_list_path.stem)
    tag = match.group() if match else proc_list_path.stem.removeprefix("proc_")
    return proc_list_path.with_name(f"bd_{tag}.json")


def striatum_mask(entry: dict) -> np.ndarray | None:
    """Rebuild the striatum mask of a draft entry: anchors -> snapped lines -> region at the striatum seed."""
    if not entry.get("anchors_px") or not entry.get("striatum_seed_px"):
        return None
    shape = tuple(entry["image_shape"])
    lines: list[np.ndarray] = []
    for anchors in entry["anchors_px"]:
        lines.append(snap_ends(anchor_curve(np.array(anchors)), shape, lines))
    labels, _ = label_regions(lines, shape)
    return region_at(labels, tuple(entry["striatum_seed_px"]))


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Outer border of a bool mask as a closed polygon (N, 2) x / y, simplified to OUTLINE_TOLERANCE_PX."""
    contours = find_contours(np.pad(mask, 1).astype(float), 0.5)  # padding closes regions touching the frame edge
    border = max(contours, key=len) - 1
    return approximate_polygon(border, tolerance=OUTLINE_TOLERANCE_PX)[:, ::-1]


def outline_mask(outline: list | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Closed outline (N, 2) x / y -> bool mask; the reader's side of mask_outline()."""
    outline = np.asarray(outline, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    rr, cc = polygon(outline[:, 1], outline[:, 0], shape)
    mask[rr, cc] = True
    return mask


# --- 5b. exported entry ---

def export_entry(entry: dict) -> dict:
    """Draft entry -> exported entry: orientation labels + (x, y) unit vectors, and the striatum outline (10X)."""
    out = {key: entry[key] for key in ("obj", "slice", "image_shape", "dorsal", "medial")}
    out["dorsal_vec"] = list(UNIT_VECTORS[entry["dorsal"]])
    out["medial_vec"] = list(UNIT_VECTORS[entry["medial"]])
    mask = striatum_mask(entry)
    if mask is not None:
        out["striatum_area_px"] = int(mask.sum())
        out["striatum_outline_px"] = np.round(mask_outline(mask), 1).tolist()
        for key in ("anchors_px", "striatum_seed_px", "cortex_seed_px"):  # lets the popup re-edit the export
            out[key] = entry.get(key)
    return out


# ===========================================================================
#
#   STEP 6 -- RE-EDIT: exported entry -> draft entry (popup "Load bd file")
#
# ===========================================================================

def outline_to_anchors(outline: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    """Boundary lines of a striatum outline as anchors: the runs of outline points off the frame edge, simplified."""
    height, width = shape
    x, y = outline[:, 0], outline[:, 1]
    on_edge = (x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)  # the padded contour runs at -0.5 / size - 0.5
    if on_edge.all() or not on_edge.any():
        return []
    start = int(np.flatnonzero(on_edge)[0])  # rotate so the closed outline starts on the edge: runs never wrap
    points, inside = np.roll(outline, -start, axis=0), ~np.roll(on_edge, -start)
    breaks = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
    n = len(points)
    # each run plus its two edge neighbours (where the line met the frame), clipped onto the frame
    runs = [points[np.arange(a - 1, b + 1) % n] for a, b in zip(breaks[::2], breaks[1::2], strict=True) if b - a >= 2]
    lines = [np.column_stack([run[:, 0].clip(0, width - 1), run[:, 1].clip(0, height - 1)]) for run in runs]
    return [approximate_polygon(line, tolerance=ANCHOR_TOLERANCE_PX) for line in lines]


def _deepest_point(outline: np.ndarray, shape: tuple[int, int]) -> list[int]:
    """[x, y] of the striatum pixel farthest from its border, found on a SEED_DOWNSAMPLE-times smaller mask (fast)."""
    small_shape = (-(-shape[0] // SEED_DOWNSAMPLE), -(-shape[1] // SEED_DOWNSAMPLE))
    small = outline_mask((outline + 0.5) / SEED_DOWNSAMPLE - 0.5, small_shape)
    depth = distance_transform_edt(np.pad(small, 1))[1:-1, 1:-1]  # padding: the frame edge counts as a border
    row, col = np.unravel_index(np.argmax(depth), small_shape)
    half = SEED_DOWNSAMPLE // 2
    return [int(min(col * SEED_DOWNSAMPLE + half, shape[1] - 1)), int(min(row * SEED_DOWNSAMPLE + half, shape[0] - 1))]


def draft_from_export(entry: dict) -> dict:
    """Exported entry -> draft entry; anchors / seeds are taken as saved, or rebuilt from the outline if missing."""
    draft = {key: entry[key] for key in ("obj", "slice", "image_shape", "dorsal", "medial")}
    if "striatum_outline_px" not in entry:
        return draft
    if entry.get("anchors_px"):
        draft.update({key: entry.get(key) for key in ("anchors_px", "striatum_seed_px", "cortex_seed_px")})
        return draft
    shape = tuple(entry["image_shape"])
    outline = np.array(entry["striatum_outline_px"])
    draft["anchors_px"] = [np.round(a, 1).tolist() for a in outline_to_anchors(outline, shape)]
    draft["striatum_seed_px"] = _deepest_point(outline, shape)
    draft["cortex_seed_px"] = None
    return draft
