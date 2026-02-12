"""
Thresholding Methods Demo

Simple demonstration of how Otsu, Li, and Yen calculate thresholds.
Shows calculations + final comparison plot.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_li, threshold_otsu, threshold_yen


def demo_otsu() -> None:
    """
    Otsu's Method: Maximize between-class variance.

    Formula: variance = w0 * w1 * (mu0 - mu1)^2

    Where:
    - w0, w1 = weight (proportion) of each class
    - mu0, mu1 = mean of each class
    """
    print("=" * 60)
    print("OTSU'S METHOD: Maximize Between-Class Variance")
    print("=" * 60)

    pixels = np.array([1, 2, 2, 3, 3, 7, 8, 8, 9, 9])
    print(f"\nPixels: {list(pixels)}")
    print(f"Total: {len(pixels)} pixels")

    print("\n--- Testing each threshold ---\n")

    best_t, best_var = 0, 0

    for t in range(1, 10):
        c0 = pixels[pixels <= t]  # Class 0 (background)
        c1 = pixels[pixels > t]   # Class 1 (foreground)

        if len(c0) == 0 or len(c1) == 0:
            continue

        w0 = len(c0) / len(pixels)
        w1 = len(c1) / len(pixels)
        mu0 = np.mean(c0)
        mu1 = np.mean(c1)

        var = w0 * w1 * (mu0 - mu1) ** 2

        marker = " <-- BEST" if var > best_var else ""
        print(f"t={t}: c0={list(c0)}, c1={list(c1)}")
        print(f"      w0={w0:.1f}, w1={w1:.1f}, mu0={mu0:.1f}, mu1={mu1:.1f}")
        print(f"      var = {w0:.1f} * {w1:.1f} * ({mu0:.1f} - {mu1:.1f})^2 = {var:.3f}{marker}")
        print()

        if var > best_var:
            best_var = var
            best_t = t

    print(f"Result: threshold = {best_t} (max variance = {best_var:.3f})")
    print(f"skimage: {threshold_otsu(pixels):.1f}")


def demo_li() -> None:
    """
    Li's Method: Minimize cross-entropy.

    Cross-entropy measures "information loss" when replacing pixels with class means.
    Lower = better fit.
    """
    print("\n" + "=" * 60)
    print("LI'S METHOD: Minimize Cross-Entropy")
    print("=" * 60)

    pixels = np.array([1, 2, 2, 3, 3, 7, 8, 8, 9, 9], dtype=float)
    print(f"\nPixels: {list(pixels.astype(int))}")

    print("\n--- Testing each threshold ---\n")

    best_t, best_ce = 0, np.inf

    for t in range(1, 10):
        c0 = pixels[pixels <= t]
        c1 = pixels[pixels > t]

        if len(c0) == 0 or len(c1) == 0:
            continue

        mu0 = np.mean(c0)
        mu1 = np.mean(c1)

        # Cross-entropy: -sum(pixel * log(mean))
        ce0 = -np.sum(c0 * np.log(mu0 + 0.01))
        ce1 = -np.sum(c1 * np.log(mu1 + 0.01))
        total = ce0 + ce1

        marker = " <-- BEST" if total < best_ce else ""
        print(f"t={t}: mu0={mu0:.2f}, mu1={mu1:.2f}")
        print(f"      ce0={ce0:.1f}, ce1={ce1:.1f}, total={total:.1f}{marker}")
        print()

        if total < best_ce:
            best_ce = total
            best_t = t

    print(f"Result: threshold ~ {best_t} (min cross-entropy = {best_ce:.1f})")
    print(f"skimage: {threshold_li(pixels):.2f}")
    print("(Note: Real Li uses iterative optimization)")


def demo_yen() -> None:
    """
    Yen's Method: Maximize entropic correlation.

    Finds threshold where both classes have good "spread" (entropy).
    """
    print("\n" + "=" * 60)
    print("YEN'S METHOD: Maximize Entropic Correlation")
    print("=" * 60)

    pixels = np.array([1, 2, 2, 3, 3, 7, 8, 8, 9, 9])
    print(f"\nPixels: {list(pixels)}")

    print("\n--- Testing each threshold ---\n")

    best_t, best_crit = 0, -np.inf

    for t in range(1, 10):
        c0 = pixels[pixels <= t]
        c1 = pixels[pixels > t]

        if len(c0) == 0 or len(c1) == 0:
            continue

        p0 = len(c0) / len(pixels)
        p1 = len(c1) / len(pixels)

        # Simplified Yen criterion
        criterion = -np.log(p0**2 + p1**2 + 1e-10)

        marker = " <-- BEST" if criterion > best_crit else ""
        print(f"t={t}: p0={p0:.1f}, p1={p1:.1f}")
        print(f"      criterion = -log({p0:.1f}^2 + {p1:.1f}^2) = {criterion:.3f}{marker}")
        print()

        if criterion > best_crit:
            best_crit = criterion
            best_t = t

    print(f"Result: threshold ~ {best_t}")
    print(f"skimage: {threshold_yen(pixels):.1f}")
    print("(Note: Real Yen uses more complex formulation)")


def plot_comparison() -> None:
    """Plot histogram with all three thresholds for comparison."""
    # Create bimodal data (like real image)
    rng = np.random.default_rng(42)
    background = rng.normal(30, 8, 500)
    signal = rng.normal(75, 12, 200)
    pixels = np.concatenate([background, signal])
    pixels = np.clip(pixels, 0, 100)

    otsu = threshold_otsu(pixels)
    li = threshold_li(pixels)
    yen = threshold_yen(pixels)

    _fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Histogram with thresholds
    ax1 = axes[0]
    ax1.hist(pixels, bins=50, color="gray", alpha=0.7, edgecolor="black")
    ax1.axvline(otsu, color="red", linewidth=2, linestyle="-", label=f"Otsu = {otsu:.1f}")
    ax1.axvline(li, color="blue", linewidth=2, linestyle="--", label=f"Li = {li:.1f}")
    ax1.axvline(yen, color="green", linewidth=2, linestyle=":", label=f"Yen = {yen:.1f}")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Bimodal Distribution (background~30, signal~75)")
    ax1.legend()

    # Right: Classification results
    ax2 = axes[1]
    methods = ["Otsu", "Li", "Yen"]
    thresh_vals = [otsu, li, yen]
    colors = ["red", "blue", "green"]

    for i, (t, c) in enumerate(zip(thresh_vals, colors, strict=True)):
        bg = np.sum(pixels <= t)
        fg = np.sum(pixels > t)
        ax2.barh(i, bg, color="gray", height=0.6)
        ax2.barh(i, fg, left=bg, color=c, height=0.6, alpha=0.7)
        ax2.text(bg / 2, i, f"{bg}", ha="center", va="center", fontweight="bold", fontsize=10)
        ax2.text(bg + fg / 2, i, f"{fg}", ha="center", va="center", fontweight="bold", color="white", fontsize=10)

    ax2.set_yticks(range(3))
    ax2.set_yticklabels([f"{m} (t={t:.1f})" for m, t in zip(methods, thresh_vals, strict=True)])
    ax2.set_xlabel("Number of Pixels")
    ax2.set_title("Classification: Background (gray) vs Signal (color)")

    plt.tight_layout()
    plt.show()


def summary() -> None:
    """Print summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Otsu:  Maximize between-class variance
       Best when: clear bimodal histogram

Li:    Minimize cross-entropy (information loss)
       Best when: foreground is small

Yen:   Maximize entropic correlation
       Best when: complex histograms

For ACh imaging with z-scored data:
  -> li_double or otsu_double (two-pass) work best
""")


if __name__ == "__main__":
    demo_otsu()
    demo_li()
    demo_yen()
    summary()
    plot_comparison()
