"""
Numba port of skimage.registration.optical_flow_tvl1 (CPU, prange over rows) -- same algorithm and defaults.

  Step 1. Pyramid : skimage pyramid_reduce (x2, min side 16) for both frames, coarse -> fine
  Step 2. Warp    : bilinear warp of the moving frame (edge mode) + np.gradient / rho_0 terms
  Step 3. Solve   : num_iter x (data-term step + 2 dual projection steps per flow component)
  Step 4. Resize  : nearest-neighbour upscale of the flow to the next level (values rescaled)

Mirrors skimage 0.26 _tvl1 in float32 (incl. its early-stop check) -> bit-identical flow.
Parallel numba kernels: call from one thread at a time (workqueue threading layer).

Example:
    >>> v, u = optical_flow_tvl1_numba(med[idx_from], med[idx_to])   # same (v, u) order as skimage
"""

## Modules
# Third-party imports
import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi
from skimage.transform import pyramid_reduce

# ===========================================================================
#
#   CONFIG  (skimage optical_flow_tvl1 / _coarse_to_fine defaults)
#
# ===========================================================================

# --- Step 1: pyramid -------------------------------------------------------
PYRAMID_DOWNSCALE = 2  # size ratio between pyramid levels
PYRAMID_NLEVEL = 10    # max number of levels
PYRAMID_MIN_SIZE = 16  # px: smallest allowed level side

# --- Step 3: solve ---------------------------------------------------------
ATTACHMENT = 15.0      # data-term weight (lambda); smaller -> smoother flow
TIGHTNESS = 0.3        # coupling between data and regularization terms (theta)
NUM_WARP = 5           # warps of the moving frame per level
NUM_ITER = 10          # fixed-point iterations per warp
TOL = 1e-4             # early stop: mean squared flow change per pixel
REG_NUM_ITER = 2       # dual projection steps per component (fixed in skimage)


# ===========================================================================
#
#   STEP 1 -- PYRAMID
#
# ===========================================================================


def _get_pyramid(img: np.ndarray) -> list[np.ndarray]:
    """Coarse-to-fine image pyramid (coarsest first), same calls as skimage _get_pyramid."""
    pyramid = [img]
    size = min(img.shape)
    count = 1
    while count < PYRAMID_NLEVEL and size > PYRAMID_DOWNSCALE * PYRAMID_MIN_SIZE:
        reduced = pyramid_reduce(pyramid[-1], PYRAMID_DOWNSCALE, channel_axis=None)
        pyramid.append(reduced)
        size = min(reduced.shape)
        count += 1
    return pyramid[::-1]


# ===========================================================================
#
#   STEP 2 -- WARP  (bilinear, edge mode + gradient / rho_0 terms)
#
# ===========================================================================


@njit(parallel=True, cache=True, nogil=True)
def _warp_bilinear(moving: np.ndarray, flow: np.ndarray, out: np.ndarray) -> None:
    """out[i, j] = moving sampled at (i + flow[0], j + flow[1]); coordinates clamped to the frame (edge mode)."""
    height, width = moving.shape
    for i in prange(height):
        for j in range(width):
            r = np.float64(flow[0, i, j] + np.float32(i))
            c = np.float64(flow[1, i, j] + np.float32(j))
            r = min(max(r, 0.0), height - 1.0)
            c = min(max(c, 0.0), width - 1.0)
            r0 = int(np.floor(r))
            c0 = int(np.floor(c))
            r1 = min(r0 + 1, height - 1)
            c1 = min(c0 + 1, width - 1)
            tr = r - r0
            tc = c - c0
            val = (
                moving[r0, c0] * (1.0 - tr) * (1.0 - tc)
                + moving[r0, c1] * (1.0 - tr) * tc
                + moving[r1, c0] * tr * (1.0 - tc)
                + moving[r1, c1] * tr * tc
            )
            out[i, j] = val


@njit(parallel=True, cache=True, nogil=True)
def _grad_terms(
    warped: np.ndarray, ref: np.ndarray, flow: np.ndarray, grad: np.ndarray, ni: np.ndarray, rho0: np.ndarray
) -> None:
    """grad = np.gradient(warped); ni = |grad|^2 (0 -> 1); rho0 = warped - ref - grad . flow."""
    height, width = warped.shape
    two = np.float32(2.0)
    for i in prange(height):
        for j in range(width):
            if i == 0:
                gy = warped[1, j] - warped[0, j]
            elif i == height - 1:
                gy = warped[i, j] - warped[i - 1, j]
            else:
                gy = (warped[i + 1, j] - warped[i - 1, j]) / two
            if j == 0:
                gx = warped[i, 1] - warped[i, 0]
            elif j == width - 1:
                gx = warped[i, j] - warped[i, j - 1]
            else:
                gx = (warped[i, j + 1] - warped[i, j - 1]) / two
            grad[0, i, j] = gy
            grad[1, i, j] = gx
            norm_sq = gy * gy + gx * gx
            ni[i, j] = np.float32(1.0) if norm_sq == 0 else norm_sq
            rho0[i, j] = (warped[i, j] - ref[i, j]) - (gy * flow[0, i, j] + gx * flow[1, i, j])


# ===========================================================================
#
#   STEP 3 -- SOLVE  (data term + dual projection)
#
# ===========================================================================


@njit(parallel=True, cache=True, nogil=True)
def _data_step(flow: np.ndarray, grad: np.ndarray, ni: np.ndarray, rho0: np.ndarray, f0: np.float32) -> None:
    """In-place thresholding step of the data term (skimage: flow_auxiliary update)."""
    height, width = ni.shape
    for i in prange(height):
        for j in range(width):
            gy = grad[0, i, j]
            gx = grad[1, i, j]
            rho = rho0[i, j] + (gy * flow[0, i, j] + gx * flow[1, i, j])
            if abs(rho) <= f0 * ni[i, j]:
                flow[0, i, j] -= rho * gy / ni[i, j]
                flow[1, i, j] -= rho * gx / ni[i, j]
            else:
                srho = f0 * np.sign(rho)
                flow[0, i, j] -= srho * gy
                flow[1, i, j] -= srho * gx


@njit(parallel=True, cache=True, nogil=True)
def _proj_step(flow_c: np.ndarray, proj: np.ndarray, dt: np.float32, f1: np.float32) -> None:
    """proj = (proj - dt * grad(flow_c)) / (1 + f1 * |grad(flow_c)|); forward differences, 0 on the last row/col."""
    height, width = flow_c.shape
    one = np.float32(1.0)
    for i in prange(height):
        for j in range(width):
            gy = flow_c[i + 1, j] - flow_c[i, j] if i < height - 1 else np.float32(0.0)
            gx = flow_c[i, j + 1] - flow_c[i, j] if j < width - 1 else np.float32(0.0)
            norm = np.sqrt(gy * gy + gx * gx) * f1 + one
            proj[0, i, j] = (proj[0, i, j] - dt * gy) / norm
            proj[1, i, j] = (proj[1, i, j] - dt * gx) / norm


@njit(parallel=True, cache=True, nogil=True)
def _div_step(flow_aux: np.ndarray, flow_c: np.ndarray, proj: np.ndarray) -> None:
    """flow_c = flow_aux + div(proj) (backward differences)."""
    height, width = flow_c.shape
    for i in prange(height):
        for j in range(width):
            d = -(proj[0, i, j] + proj[1, i, j])
            if i > 0:
                d += proj[0, i - 1, j]
            if j > 0:
                d += proj[1, i, j - 1]
            flow_c[i, j] = flow_aux[i, j] + d


@njit(parallel=True, cache=True, nogil=True)
def _sq_diff_sum(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of (a - b)^2 over all elements (float64 accumulator)."""
    flat_a = a.ravel()
    flat_b = b.ravel()
    total = 0.0
    for k in prange(flat_a.size):
        diff = np.float64(flat_a[k] - flat_b[k])
        total += diff * diff
    return total


def _tvl1_level(ref: np.ndarray, moving: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """One pyramid level of skimage _tvl1 (2-D, prefilter=False); flow (2, H, W) is updated in place and returned."""
    dt = np.float32(0.5 / 2)
    f0 = np.float32(ATTACHMENT * TIGHTNESS)
    f1 = np.float32(0.5 / 2 / TIGHTNESS)
    tol = TOL * ref.size

    shape = ref.shape
    warped = np.empty(shape, dtype=np.float32)
    grad = np.empty((2, *shape), dtype=np.float32)
    ni = np.empty(shape, dtype=np.float32)
    rho0 = np.empty(shape, dtype=np.float32)
    proj = np.zeros((2, 2, *shape), dtype=np.float32)

    flow_current = flow
    for _ in range(NUM_WARP):
        _warp_bilinear(moving, flow_current, warped)
        _grad_terms(warped, ref, flow_current, grad, ni, rho0)

        flow_first = flow_current  # replaced after the first data step (skimage's flow_previous alias)
        for it in range(NUM_ITER):
            _data_step(flow_current, grad, ni, rho0, f0)
            if it == 0:
                flow_first = flow_current.copy()
            flow_aux = flow_current
            flow_current = flow_aux.copy()
            for comp in range(2):
                for _ in range(REG_NUM_ITER):
                    _proj_step(flow_current[comp], proj[comp], dt, f1)
                    _div_step(flow_aux[comp], flow_current[comp], proj[comp])

        if _sq_diff_sum(flow_first, flow_current) < tol:
            break

    return flow_current


# ===========================================================================
#
#   STEP 4 -- RESIZE  (+ coarse-to-fine driver)
#
# ===========================================================================


def _resize_flow(flow: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upscale of (2, h, w) flow to shape, values multiplied by the scale factor (as skimage)."""
    scale = [n / o for n, o in zip(shape, flow.shape[1:], strict=True)]
    scale_factor = np.array(scale, dtype=flow.dtype)[:, np.newaxis, np.newaxis]
    return scale_factor * ndi.zoom(flow, [1, *scale], order=0, mode="nearest", prefilter=False)


def optical_flow_tvl1_numba(reference_image: np.ndarray, moving_image: np.ndarray) -> np.ndarray:
    """(2, H, W) float32 flow (v, u) -- drop-in for skimage optical_flow_tvl1 with default parameters."""
    ref = np.ascontiguousarray(reference_image, dtype=np.float32)
    mov = np.ascontiguousarray(moving_image, dtype=np.float32)
    if ref.shape != mov.shape:
        msg = "Input images should have the same shape"
        raise ValueError(msg)

    pyramid = list(zip(_get_pyramid(ref), _get_pyramid(mov), strict=True))
    flow = np.zeros((2, *pyramid[0][0].shape), dtype=np.float32)
    flow = _tvl1_level(*pyramid[0], flow)
    for level_ref, level_mov in pyramid[1:]:
        flow = _tvl1_level(level_ref, level_mov, np.ascontiguousarray(_resize_flow(flow, level_ref.shape)))
    return flow
