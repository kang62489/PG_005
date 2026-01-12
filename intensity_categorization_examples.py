"""
Comparison of different intensity categorization methods for fluorescence imaging.

This script demonstrates 4 different approaches to categorize pixel intensities
into background/dim/bright groups, suitable for ACh imaging analysis.
"""

import os
import warnings

# Silence joblib/loky CPU warning on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
from typing import Tuple


def method_1_kmeans(image: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method 1: K-means clustering (your current approach).

    Pros: Adapts to data, no thresholds needed
    Cons: Requires scikit-learn, can be slow
    """
    # Flatten and reshape for sklearn
    pixels = image.flatten().reshape(-1, 1)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=100, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.flatten()

    # Sort by intensity (0=background, 1=dim, 2=bright)
    sorted_indices = np.argsort(centers)
    label_mapping = np.zeros(n_clusters, dtype=int)
    label_mapping[sorted_indices] = np.arange(n_clusters)
    sorted_labels = label_mapping[labels]

    # Reshape back to image
    categorized = sorted_labels.reshape(image.shape)

    return categorized, centers[sorted_indices]


def method_2_percentiles(image: np.ndarray, low_pct: float = 33, high_pct: float = 67) -> Tuple[np.ndarray, Tuple]:
    """
    Method 2: Percentile-based thresholding.

    Pros: Very fast, simple, predictable
    Cons: Fixed proportions, doesn't adapt to distribution
    """
    threshold_low = np.percentile(image, low_pct)
    threshold_high = np.percentile(image, high_pct)

    categorized = np.zeros_like(image, dtype=int)
    categorized[image > threshold_low] = 1
    categorized[image > threshold_high] = 2

    return categorized, (threshold_low, threshold_high)


def method_3_zscore(
    image: np.ndarray,
    z_threshold_1: float = 0.5,
    z_threshold_2: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method 3: Z-score based categorization.

    Pros: Adapts to each image, works with noisy data
    Cons: Requires choosing z-score thresholds
    """
    # Calculate z-scores
    mean = np.mean(image)
    std = np.std(image)
    z_scores = (image - mean) / (std + 1e-10)

    # Categorize
    categorized = np.zeros_like(image, dtype=int)
    categorized[z_scores > z_threshold_1] = 1
    categorized[z_scores > z_threshold_2] = 2

    return categorized, z_scores


def method_4_otsu(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method 4: Multi-Otsu automatic thresholding.

    Pros: Fully automatic, fast, histogram-based
    Cons: May not work if groups aren't well-separated

    Note: Requires scikit-image
    """
    try:
        from skimage.filters import threshold_multiotsu

        thresholds = threshold_multiotsu(image, classes=3)
        categorized = np.digitize(image, bins=thresholds)

        return categorized, thresholds
    except ImportError:
        print("Warning: scikit-image not available, using percentile method instead")
        return method_2_percentiles(image)


def compare_methods(image: np.ndarray, title: str = "Frame 0") -> plt.Figure:
    """
    Compare all 4 methods side-by-side.

    Args:
        image: 2D numpy array (z-scored frame from your analysis)
        title: Title for the figure

    Returns:
        matplotlib Figure
    """
    # Apply all methods
    cat1, centers = method_1_kmeans(image)
    cat2, thresholds2 = method_2_percentiles(image)
    cat3, z_scores = method_3_zscore(image)
    cat4, thresholds4 = method_4_otsu(image)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    im0 = axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Z-scored Image')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Method 1: K-means
    im1 = axes[0, 1].imshow(cat1, cmap='viridis', vmin=0, vmax=2)
    axes[0, 1].set_title(f'Method 1: K-means\nCenters: {centers}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, ticks=[0, 1, 2])

    # Method 2: Percentiles
    im2 = axes[0, 2].imshow(cat2, cmap='viridis', vmin=0, vmax=2)
    axes[0, 2].set_title(f'Method 2: Percentiles (33%, 67%)\nThresholds: {thresholds2[0]:.2f}, {thresholds2[1]:.2f}')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, ticks=[0, 1, 2])

    # Method 3: Z-score
    im3 = axes[1, 0].imshow(cat3, cmap='viridis', vmin=0, vmax=2)
    axes[1, 0].set_title('Method 3: Z-score (0.5, 1.5)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, ticks=[0, 1, 2])

    # Method 4: Otsu
    im4 = axes[1, 1].imshow(cat4, cmap='viridis', vmin=0, vmax=2)
    axes[1, 1].set_title(f'Method 4: Multi-Otsu (auto)\nThresholds: {thresholds4[0]:.2f}, {thresholds4[1]:.2f}')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, ticks=[0, 1, 2])

    # Histogram comparison
    axes[1, 2].hist(image.flatten(), bins=100, alpha=0.7, color='gray', edgecolor='black')
    axes[1, 2].axvline(thresholds2[0], color='blue', linestyle='--', label=f'Percentile: {thresholds2[0]:.2f}')
    axes[1, 2].axvline(thresholds2[1], color='blue', linestyle='--', label=f'Percentile: {thresholds2[1]:.2f}')
    axes[1, 2].axvline(thresholds4[0], color='red', linestyle=':', label=f'Otsu: {thresholds4[0]:.2f}')
    axes[1, 2].axvline(thresholds4[1], color='red', linestyle=':', label=f'Otsu: {thresholds4[1]:.2f}')
    axes[1, 2].set_title('Intensity Distribution')
    axes[1, 2].set_xlabel('Z-score')
    axes[1, 2].set_ylabel('Pixel Count')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.3)

    plt.suptitle(f'Comparison of Intensity Categorization Methods - {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def calculate_statistics(image: np.ndarray, categorized: np.ndarray) -> dict:
    """Calculate statistics for each category."""
    stats = {}
    for category in [0, 1, 2]:
        mask = categorized == category
        pixels_in_category = image[mask]
        stats[category] = {
            'count': np.sum(mask),
            'percentage': np.sum(mask) / image.size * 100,
            'mean_intensity': np.mean(pixels_in_category) if len(pixels_in_category) > 0 else 0,
            'std_intensity': np.std(pixels_in_category) if len(pixels_in_category) > 0 else 0
        }
    return stats


def print_method_comparison(image: np.ndarray):
    """Print detailed comparison of all methods."""
    print("\n" + "="*80)
    print("INTENSITY CATEGORIZATION METHOD COMPARISON")
    print("="*80)

    methods = [
        ("K-means", method_1_kmeans(image)),
        ("Percentiles", method_2_percentiles(image)),
        ("Z-score", method_3_zscore(image)),
        ("Multi-Otsu", method_4_otsu(image))
    ]

    for name, (categorized, _) in methods:
        print(f"\n{name:15s}")
        print("-" * 80)
        stats = calculate_statistics(image, categorized)
        for cat_idx, cat_name in enumerate(['Background', 'Dim', 'Bright']):
            s = stats[cat_idx]
            print(f"  {cat_name:12s}: {s['count']:7d} pixels ({s['percentage']:5.1f}%) | "
                  f"Mean: {s['mean_intensity']:6.2f} | Std: {s['std_intensity']:6.2f}")


# ============================================================================
# RECOMMENDATION FOR YOUR ACH IMAGING
# ============================================================================

def recommended_approach_for_ach_imaging(zscore_frame: np.ndarray) -> np.ndarray:
    """
    RECOMMENDED: Hybrid approach combining z-score and k-means.

    This works best for ACh imaging because:
    1. You've already z-score normalized (baseline corrected)
    2. K-means adapts to each segment
    3. Z-score thresholds have biological meaning

    Use this when you want to:
    - Identify ACh release regions (bright pixels)
    - Separate true signal from noise
    - Maintain consistency across segments
    """
    # Step 1: Use z-score to remove obvious background (z < 0.5)
    background_mask = zscore_frame <= 0.5

    # Step 2: Apply k-means only to potential signal pixels
    signal_pixels = zscore_frame[~background_mask].flatten().reshape(-1, 1)

    if len(signal_pixels) > 0:
        # K-means on signal pixels (2 groups: dim vs bright)
        kmeans = KMeans(n_clusters=2, random_state=100, n_init=10)
        signal_labels = kmeans.fit_predict(signal_pixels)
        centers = kmeans.cluster_centers_.flatten()

        # Sort labels
        sorted_indices = np.argsort(centers)
        label_mapping = np.zeros(2, dtype=int)
        label_mapping[sorted_indices] = [1, 2]  # 1=dim, 2=bright
        sorted_signal_labels = label_mapping[signal_labels]

        # Combine results
        result = np.zeros_like(zscore_frame, dtype=int)
        result[~background_mask] = sorted_signal_labels
    else:
        result = np.zeros_like(zscore_frame, dtype=int)

    return result


if __name__ == "__main__":
    """
    Example usage: Test on synthetic ACh imaging data.
    """
    print("Generating synthetic fluorescence image...")

    # Create synthetic fluorescence image (similar to your ACh imaging)
    np.random.seed(42)
    image = np.random.randn(256, 256) * 0.5  # Background noise

    # Add some "ACh release" spots
    y, x = np.ogrid[-128:128, -128:128]

    # Bright spot (high ACh release)
    bright_spot = np.exp(-((x-50)**2 + (y-30)**2) / 400) * 3
    image += bright_spot

    # Dim spot (moderate ACh release)
    dim_spot = np.exp(-((x+40)**2 + (y+50)**2) / 600) * 1.5
    image += dim_spot

    # Another bright spot
    bright_spot2 = np.exp(-((x-30)**2 + (y-60)**2) / 300) * 2.5
    image += bright_spot2

    print(f"Image stats: Mean={image.mean():.2f}, Std={image.std():.2f}, "
          f"Min={image.min():.2f}, Max={image.max():.2f}")

    # Print comparison
    print_method_comparison(image)

    # Visualize comparison
    fig = compare_methods(image, title="Synthetic ACh Imaging Frame")
    plt.savefig('intensity_categorization_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison figure: intensity_categorization_comparison.png")

    # Show recommended approach
    recommended = recommended_approach_for_ach_imaging(image)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Z-scored Image')
    axes[0].axis('off')

    axes[1].imshow(recommended, cmap='viridis', vmin=0, vmax=2)
    axes[1].set_title('RECOMMENDED: Hybrid Z-score + K-means')
    axes[1].axis('off')

    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046,
                 ticks=[0, 1, 2], label='0=Background, 1=Dim, 2=Bright')
    plt.tight_layout()
    plt.savefig('recommended_categorization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved recommended approach: recommended_categorization.png")

    plt.show()

    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print("""
For YOUR ACh imaging analysis, I recommend:

1. CURRENT APPROACH (K-means) ✓
   - Already working well in your code
   - Use for: Identifying ACh release zones
   - Keep: process_segment_kmeans_concatenated() approach

2. QUICK ALTERNATIVE (Z-score thresholds)
   - Use when: You want faster, more consistent results
   - Thresholds: 0.5 (background/dim), 1.5 (dim/bright)
   - Add to: Your frequency analysis section

3. HYBRID APPROACH (Recommended for new analysis)
   - Combine z-score pre-filtering + k-means on signal
   - Best of both worlds
   - Faster than pure k-means
   - More adaptive than pure thresholding

WHEN TO USE EACH:
- Single frame analysis → Method 3 (Z-score) - fastest
- Segment-wise consistency → Method 1 (K-means) - your current
- Fully automatic → Method 4 (Multi-Otsu) - no parameters
- Quick screening → Method 2 (Percentiles) - predictable
    """)
