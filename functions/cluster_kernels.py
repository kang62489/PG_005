"""
Exact numba replacements for the scipy binary-morphology calls in categorization / clustering (CPU, serial, nogil).

  Step 1. Open + close : 4 separable square-window passes (erode, dilate, dilate, erode), scipy zero-border semantics
                         == binary_erosion/binary_dilation with a full (2r+1) x (2r+1) structure
  Step 2. Eps zone     : exact integer squared Euclidean distance (Felzenszwalb lower envelope) <= radius^2
                         == distance_transform_edt(~mask) <= radius

Kernels are serial and nogil, so they are safe to call from several Python threads at once
(one segment / frame per thread); a numba parallel kernel would not be.

Example:
    >>> cleaned = binary_open_close_square(frame > thresh, radius=3)    # 7x7 opening then closing
    >>> zone = within_distance(bright_mask, eps_px)                     # every pixel within eps_px of a bright pixel
"""

## Modules
# Third-party imports
import numpy as np
from numba import njit

# ===========================================================================
#
#   STEP 1 -- OPEN + CLOSE  (square window, separable)
#
# ===========================================================================


@njit(cache=True, nogil=True)
def _row_pass(src: np.ndarray, radius: int, erode: bool) -> np.ndarray:
    """1-D window pass along each row: erode = all True (window leaving the frame -> False), else any True."""
    height, width = src.shape
    full = 2 * radius + 1
    out = np.empty((height, width), dtype=np.bool_)
    for i in range(height):
        count = 0  # True pixels in the clipped window [j - radius, j + radius]
        for k in range(min(radius, width - 1) + 1):
            count += src[i, k]
        for j in range(width):
            out[i, j] = count == full if erode else count > 0
            if j + radius + 1 < width:
                count += src[i, j + radius + 1]
            if j - radius >= 0:
                count -= src[i, j - radius]
    return out


@njit(cache=True, nogil=True)
def _col_pass(src: np.ndarray, radius: int, erode: bool) -> np.ndarray:
    """Same as _row_pass along each column, swept row by row with one running count per column (contiguous reads)."""
    height, width = src.shape
    full = 2 * radius + 1
    out = np.empty((height, width), dtype=np.bool_)
    count = np.zeros(width, dtype=np.int64)
    for k in range(min(radius, height - 1) + 1):
        for j in range(width):
            count[j] += src[k, j]
    for i in range(height):
        for j in range(width):
            out[i, j] = count[j] == full if erode else count[j] > 0
        if i + radius + 1 < height:
            for j in range(width):
                count[j] += src[i + radius + 1, j]
        if i - radius >= 0:
            for j in range(width):
                count[j] -= src[i - radius, j]
    return out


@njit(cache=True, nogil=True)
def _square_morph(mask: np.ndarray, radius: int, erode: bool) -> np.ndarray:
    """Binary erosion / dilation with a full (2r+1)^2 square, border value 0 (a clipped window never counts full)."""
    return _col_pass(_row_pass(mask, radius, erode), radius, erode)


@njit(cache=True, nogil=True)
def binary_open_close_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Opening then closing (erode, dilate, dilate, erode) with a full (2r+1) x (2r+1) square."""
    out = _square_morph(mask, radius, True)
    out = _square_morph(out, radius, False)
    out = _square_morph(out, radius, False)
    return _square_morph(out, radius, True)


# ===========================================================================
#
#   STEP 2 -- EPS ZONE  (exact squared Euclidean distance transform)
#
# ===========================================================================


@njit(cache=True, nogil=True)
def within_distance(mask: np.ndarray, radius: int) -> np.ndarray:
    """True where the Euclidean distance to the nearest True pixel of mask is <= radius (mask must have a True)."""
    height, width = mask.shape
    inf = np.int64(height * height + width * width + 1)

    # --- 2a. per row: horizontal distance to the nearest True pixel ---
    row_dist = np.empty((height, width), dtype=np.int64)
    for i in range(height):
        last = -1
        for j in range(width):
            if mask[i, j]:
                last = j
            row_dist[i, j] = j - last if last >= 0 else inf
        nxt = -1
        for j in range(width - 1, -1, -1):
            if mask[i, j]:
                nxt = j
            if nxt >= 0 and nxt - j < row_dist[i, j]:
                row_dist[i, j] = nxt - j

    # --- 2b. per column: lower envelope of parabolas row_dist[q]^2 + (p - q)^2 ---
    radius_sq = np.int64(radius) * np.int64(radius)
    out = np.zeros((height, width), dtype=np.bool_)
    f = np.empty(height, dtype=np.int64)
    v = np.empty(height, dtype=np.int64)
    z = np.empty(height + 1, dtype=np.float64)
    for j in range(width):
        k = -1
        for q in range(height):
            if row_dist[q, j] >= inf:
                continue
            f[q] = row_dist[q, j] * row_dist[q, j]
            if k < 0:
                k = 0
                v[0] = q
                z[0] = -np.inf
                z[1] = np.inf
                continue
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
            while s <= z[k]:
                k -= 1
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = np.inf
        if k < 0:
            continue  # mask has no True pixel at all -> column stays False
        k = 0
        for p in range(height):
            while z[k + 1] < p:
                k += 1
            dq = p - v[k]
            out[p, j] = dq * dq + f[v[k]] <= radius_sq
    return out
