"""Descriptive sector analysis of CAT-bright area changes (not motion estimation)."""

import numpy as np


def analyze_directional_change(bright: np.ndarray, spike_index, n_sectors=8) -> dict:
    """Use a fixed spike-frame centroid and the largest fully observed centered circle.

    Rates are pixels/frame. CV uses gain per available sector pixel, correcting
    tiny raster-area differences. It describes nonuniformity, not significance.
    """
    if n_sectors not in (4, 8) or bright.ndim != 3:
        msg = "Expected a 3D bright mask and four or eight sectors."
        raise ValueError(msg)
    bright = bright.astype(bool)
    coords = np.argwhere(bright[spike_index])
    if not len(coords):
        msg = "The spike frame has no bright pixels; a reference center cannot be defined."
        raise ValueError(msg)
    cy, cx = coords.mean(axis=0)
    height, width = bright.shape[1:]
    radius = min(cx, cy, width - 1 - cx, height - 1 - cy)
    yy, xx = np.indices((height, width))
    roi = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    angle = np.mod(np.arctan2(xx - cx, cy - yy), 2 * np.pi)
    sector = np.floor((angle + np.pi / n_sectors) / (2 * np.pi / n_sectors)).astype(int) % n_sectors
    areas = np.bincount(sector[roi], minlength=n_sectors)
    if np.any(areas == 0):
        msg = "Reference center is too close to the image edge for sector analysis."
        raise ValueError(msg)
    offsets = np.arange(max(-1, -spike_index), min(4, len(bright) - spike_index - 1))
    gain, loss, changes, coverage = [], [], [], []
    for offset in offsets:
        before, after = bright[spike_index + offset : spike_index + offset + 2]
        gained, lost = after & ~before, before & ~after
        gain.append(np.bincount(sector[gained & roi], minlength=n_sectors))
        loss.append(np.bincount(sector[lost & roi], minlength=n_sectors))
        changes.append(np.where(gained, 2, np.where(lost, 3, np.where(before & after, 1, 0))))
        changed = gained | lost
        coverage.append(np.count_nonzero(changed & roi) / changed.sum() if changed.any() else np.nan)
    gain, loss = np.asarray(gain), np.asarray(loss)
    density = gain / areas
    cv = np.divide(density.std(axis=1), density.mean(axis=1), out=np.full(len(offsets), np.nan), where=density.mean(axis=1) > 0)
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] if n_sectors == 8 else ["N", "E", "S", "W"]
    return {"center": (cy, cx), "radius": radius, "roi": roi, "areas": areas, "offsets": offsets,
            "gain": gain, "loss": loss, "net": gain - loss, "cv": cv, "changes": changes,
            "coverage": np.asarray(coverage), "labels": labels}
