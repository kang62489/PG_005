"""Exploratory mask-change measurements; no inference of physical transport."""

import numpy as np
from scipy.ndimage import affine_transform, binary_erosion, gaussian_filter
from scipy.optimize import minimize
from skimage.measure import find_contours

from classes.region_analyzer import (
    CATEGORY_BRIGHT,
    _run_density_gated_cluster_seeker,
    compute_density_thresh,
    compute_eps_px,
    compute_window_px,
)


class HotspotDirectionAnalysis:
    """Measure all accepted mask pixels, independently of display-arrow selection."""

    def __init__(self, obj="60X", sectors=12) -> None:
        self.obj = obj
        self.sectors = sectors

    def cluster_mask(self, cat: np.ndarray) -> tuple[np.ndarray, int]:
        labels, centers, _ = _run_density_gated_cluster_seeker(
            cat == CATEGORY_BRIGHT, compute_eps_px(self.obj), compute_window_px(self.obj),
            compute_density_thresh(self.obj), z_frame=None,
        )
        return labels >= 0, len(centers)

    @staticmethod
    def smooth(mask: np.ndarray, sigma=3) -> np.ndarray:
        return gaussian_filter(mask.astype(float), sigma, mode="nearest") >= 0.5 if sigma else mask.copy()

    @staticmethod
    def center(mask: np.ndarray) -> np.ndarray:
        points = np.argwhere(mask)
        if not len(points):
            message = "Cannot choose a reference center from an empty hotspot."
            raise ValueError(message)
        return points.mean(axis=0)

    def geometry(self, shape, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.indices(shape)
        # Mathematical directions: east=0, north=90; images have downward-positive rows.
        angle = np.mod(np.arctan2(center[0] - rows, cols - center[1]), 2 * np.pi)
        sector = np.floor(angle / (2 * np.pi) * self.sectors).astype(int)
        return angle, sector

    def pair(self, old: np.ndarray, new: np.ndarray, center: np.ndarray, margin=0) -> dict:
        angle, sector = self.geometry(old.shape, center)
        valid = np.ones(old.shape, dtype=bool)
        if margin:
            valid[:margin] = valid[-margin:] = False
            valid[:, :margin] = valid[:, -margin:] = False
        gained, lost = new & ~old & valid, old & ~new & valid
        result = {"gained": gained, "lost": lost, "center": center}
        for name, mask in (("gain", gained), ("loss", lost)):
            values = angle[mask]
            result[name] = np.bincount(sector[mask], minlength=self.sectors)
            result[f"{name}_total"] = int(mask.sum())
            result[f"{name}_r1"] = float(abs(np.mean(np.exp(1j * values)))) if len(values) else np.nan
            result[f"{name}_r2"] = float(abs(np.mean(np.exp(2j * values)))) if len(values) else np.nan
            result[f"{name}_angle"] = float(np.angle(np.mean(np.exp(1j * values)))) if len(values) else np.nan
        # Midpoints of all marching-squares segments, weighted by Euclidean length.
        # No artificial closure along the image edge; endpoint availability can be zero.
        perimeter = np.zeros(self.sectors)
        for curve in find_contours(old.astype(float), 0.5):
            points = (curve[:-1] + curve[1:]) / 2
            lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            angles = np.mod(np.arctan2(center[0] - points[:, 0], points[:, 1] - center[1]), 2 * np.pi)
            bins = np.floor(angles / (2 * np.pi) * self.sectors).astype(int)
            in_view = ((points[:, 0] >= margin) & (points[:, 0] < old.shape[0] - margin)
                       & (points[:, 1] >= margin) & (points[:, 1] < old.shape[1] - margin))
            perimeter += np.bincount(bins[in_view], weights=lengths[in_view], minlength=self.sectors)
        result["perimeter"] = perimeter
        for name in ("gain", "loss"):
            result[f"{name}_normalized"] = np.divide(
                result[name], perimeter, out=np.full(self.sectors, np.nan), where=perimeter > 1,
            )
        edge = old & ~binary_erosion(old)
        border = np.zeros_like(old)
        # Categorizer erosion clears the outer few pixels even for truncated regions.
        # A wider band detects near-edge extent, without claiming proven truncation.
        border[:16] = border[-16:] = True
        border[:, :16] = border[:, -16:] = True
        result["border_fraction"] = float(np.sum(edge & border) / max(edge.sum(), 1))
        return result

    def sequence(self, masks, center: np.ndarray, margin=0) -> list[dict]:
        return [self.pair(a, b, center, margin) for a, b in zip(masks[:-1], masks[1:], strict=True)]

    @staticmethod
    def warp(mask: np.ndarray, center: np.ndarray, scale=1.0, shift=(0.0, 0.0)) -> np.ndarray:
        """Inverse-map a scale and translation; unobserved outside pixels are zero."""
        offset = center - (center + np.asarray(shift)) / scale
        return affine_transform(mask.astype(float), np.eye(2) / scale, offset=offset,
                                order=1, mode="constant", cval=0, prefilter=False)

    def fit_models(self, old: np.ndarray, new: np.ndarray, center: np.ndarray, only_center=False) -> list[dict]:
        """Exploratory spatial holdout comparison, not an independent biological test.

        Fit on alternating 128-pixel tiles at 4x reduced resolution, then score on
        the other tiles at full resolution. Do not declare a statistical winner.
        """
        step = 4
        # Gaussian prefilter makes the optimizer less sensitive to threshold plateaus.
        source = gaussian_filter(old.astype(float), 1)[::step, ::step]
        target = gaussian_filter(new.astype(float), 1)[::step, ::step]
        rows, cols = np.indices(source.shape)
        train = (rows // 32 + cols // 32) % 2 == 0
        small_center = center / step
        centroid_shift = (self.center(new) - self.center(old)) / step if new.any() and old.any() else np.zeros(2)
        size_ratio = float(np.sqrt(new.sum() / max(old.sum(), 1)))
        models = []
        for name, count in (("Unchanged", 0), ("Translation", 2), ("Centered scale", 1),
                            ("Scale + shift", 3), ("Disappearance", 0)):
            if only_center and name != "Centered scale":
                continue
            def decode(params, name=name) -> tuple[float, np.ndarray]:
                if name == "Translation":
                    return 1.0, np.asarray(params)
                if name == "Centered scale":
                    return float(params[0]), np.zeros(2)
                if name == "Scale + shift":
                    return float(params[0]), np.asarray(params[1:])
                return 1.0, np.zeros(2)

            def objective(params) -> float:
                scale, shift = decode(params)
                prediction = self.warp(source, small_center, scale, shift)
                return float(np.mean((prediction[train] - target[train]) ** 2))

            if count:
                shift_bound = max(old.shape) / step / 3
                bounds = [(0.1, 2.0)] if count == 1 else [(-shift_bound, shift_bound)] * 2
                if count == 3:
                    bounds = [(0.1, 2.0), *bounds]
                guesses = [np.zeros(2), centroid_shift] if count == 2 else [
                    np.array([1.0]), np.array([np.clip(size_ratio, 0.1, 2.0)]),
                ]
                if count == 3:
                    guesses = [np.array([1.0, 0.0, 0.0]), np.r_[np.clip(size_ratio, 0.1, 2), centroid_shift]]
                fits = [minimize(objective, np.clip(guess, *np.array(bounds).T), method="Powell", bounds=bounds,
                                 options={"maxiter": 50, "xtol": 0.002, "ftol": 0.0001}) for guess in guesses]
                best = min(fits, key=lambda fit: fit.fun)
                scale, shift = decode(best.x)
                at_bound = any(abs(value - lower) < 0.01 or abs(value - upper) < 0.01
                               for value, (lower, upper) in zip(best.x, bounds, strict=True))
                converged = bool(best.success)
            else:
                scale, shift, at_bound, converged = 1.0, np.zeros(2), False, True
            shift = shift * step
            prediction = self.warp(old, center, scale, shift) >= 0.5
            if name == "Disappearance":
                prediction = np.zeros_like(old)
            rows, cols = np.indices(old.shape)
            validation = (rows // 128 + cols // 128) % 2 == 1
            baseline = old ^ new
            error = prediction ^ new
            denominator = int(np.sum(baseline & validation))
            reduction = 1 - np.sum(error & validation) / denominator if denominator else np.nan
            union = np.sum(prediction | new)
            iou = np.sum(prediction & new) / union if union else 1.0
            models.append({"name": name, "parameters": count, "scale": scale, "shift": shift,
                           "prediction": prediction, "reduction": float(reduction), "iou": float(iou),
                           "at_bound": at_bound, "converged": converged})
        return models
