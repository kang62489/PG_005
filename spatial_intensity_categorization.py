"""
Spatial-Aware Intensity Categorization for Fluorescence Imaging

K-means problem: Treats each pixel independently, ignores spatial connectivity
Solution: Methods that consider both intensity AND spatial relationships

For ACh imaging, spatial connectivity matters because:
- True ACh release forms coherent patches (spatially connected)
- Noise creates isolated bright pixels (spatially scattered)
- We want to identify regions, not just pixel intensities
"""

import os
import warnings

# Silence joblib/loky CPU warning on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import label, binary_dilation, binary_erosion
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN


def method_1_connected_components(
    image: np.ndarray,
    threshold_dim: float = 0.5,
    threshold_bright: float = 1.5,
    min_size: int = 10
) -> tuple[np.ndarray, dict]:
    """
    Method 1: Connected Components Analysis

    Process:
    1. Threshold into dim/bright regions
    2. Find spatially connected regions
    3. Filter by size (remove noise)

    Pros: Simple, fast, effective for clear signals
    Cons: Requires threshold tuning

    Best for: Well-separated ACh patches
    """
    # Create binary masks for each category
    dim_mask = (image > threshold_dim) & (image <= threshold_bright)
    bright_mask = image > threshold_bright

    # Find connected components
    labeled_dim, num_dim = label(dim_mask)
    labeled_bright, num_bright = label(bright_mask)

    # Filter small regions (noise)
    for region_id in range(1, num_dim + 1):
        if np.sum(labeled_dim == region_id) < min_size:
            labeled_dim[labeled_dim == region_id] = 0

    for region_id in range(1, num_bright + 1):
        if np.sum(labeled_bright == region_id) < min_size:
            labeled_bright[labeled_bright == region_id] = 0

    # Combine into single categorized image
    categorized = np.zeros_like(image, dtype=int)
    categorized[labeled_dim > 0] = 1  # Dim regions
    categorized[labeled_bright > 0] = 2  # Bright regions

    # Statistics
    stats = {
        'num_dim_regions': len(np.unique(labeled_dim)) - 1,  # -1 to exclude 0
        'num_bright_regions': len(np.unique(labeled_bright)) - 1,
        'dim_sizes': [np.sum(labeled_dim == i) for i in range(1, num_dim + 1) if np.sum(labeled_dim == i) >= min_size],
        'bright_sizes': [np.sum(labeled_bright == i) for i in range(1, num_bright + 1) if np.sum(labeled_bright == i) >= min_size]
    }

    return categorized, stats


def method_2_watershed(
    image: np.ndarray,
    min_distance: int = 10,
    threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Method 2: Watershed Segmentation

    Treats intensity as topography:
    - Peaks = ACh release sites
    - Basins = spatially coherent regions

    Process:
    1. Find local maxima (peaks)
    2. Grow regions from peaks (watershed)
    3. Each region gets labeled by peak intensity

    Pros: Classic method, handles touching regions well
    Cons: Sensitive to noise, needs smoothing

    Best for: Identifying individual ACh release sites
    """
    # Smooth image to reduce noise
    smoothed = ndimage.gaussian_filter(image, sigma=1.5)

    # Find local maxima (ACh release sites)
    mask = smoothed > threshold
    distance = ndimage.distance_transform_edt(mask)

    # Find peaks with minimum distance between them
    local_max = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask,
        exclude_border=False
    )

    # Create markers for watershed
    markers = np.zeros_like(image, dtype=int)
    for idx, (y, x) in enumerate(local_max):
        markers[y, x] = idx + 1

    # Watershed segmentation
    labels = watershed(-smoothed, markers, mask=mask)

    # Categorize based on mean intensity of each region
    categorized = np.zeros_like(image, dtype=int)
    for region_id in np.unique(labels):
        if region_id == 0:  # Background
            continue
        region_mask = labels == region_id
        mean_intensity = np.mean(image[region_mask])

        if mean_intensity > 1.5:
            categorized[region_mask] = 2  # Bright
        elif mean_intensity > 0.5:
            categorized[region_mask] = 1  # Dim

    return categorized, labels


def method_3_dbscan_spatial(
    image: np.ndarray,
    threshold: float = 0.5,
    eps: float = 3.0,
    min_samples: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Method 3: DBSCAN (Density-Based Spatial Clustering)

    Considers BOTH intensity AND spatial proximity:
    - Clusters pixels that are close in space AND similar in intensity
    - Automatically finds number of clusters
    - Labels isolated pixels as noise

    Process:
    1. Extract pixels above threshold
    2. Cluster using spatial coordinates + intensity
    3. Categorize clusters by mean intensity

    Pros: Finds arbitrary-shaped clusters, handles noise well
    Cons: Parameters (eps, min_samples) need tuning

    Best for: Complex ACh patterns with noise
    """
    # Get pixels above threshold
    mask = image > threshold
    coords = np.argwhere(mask)  # (y, x) coordinates
    intensities = image[mask].reshape(-1, 1)

    if len(coords) == 0:
        return np.zeros_like(image, dtype=int), np.zeros_like(image, dtype=int)

    # Combine spatial coordinates and intensity
    # Scale intensity to match spatial scale (important!)
    intensity_scale = 10.0  # How much to weight intensity vs space
    features = np.hstack([coords, intensities * intensity_scale])

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(features)

    # Create full image with cluster labels
    labeled_image = np.zeros_like(image, dtype=int)
    for idx, (y, x) in enumerate(coords):
        if cluster_labels[idx] != -1:  # -1 is noise in DBSCAN
            labeled_image[y, x] = cluster_labels[idx] + 1

    # Categorize each cluster by mean intensity
    categorized = np.zeros_like(image, dtype=int)
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Noise
            continue
        cluster_mask = labeled_image == (cluster_id + 1)
        mean_intensity = np.mean(image[cluster_mask])

        if mean_intensity > 1.5:
            categorized[cluster_mask] = 2  # Bright
        elif mean_intensity > 0.5:
            categorized[cluster_mask] = 1  # Dim

    return categorized, labeled_image


def method_4_morphological_cleanup(
    image: np.ndarray,
    threshold_dim: float = 0.5,
    threshold_bright: float = 1.5,
    kernel_size: int = 3
) -> tuple[np.ndarray, dict]:
    """
    Method 4: K-means + Morphological Post-processing

    Enhance your current k-means approach:
    1. Run k-means (intensity-based)
    2. Clean up using morphological operations
    3. Enforce spatial connectivity

    Operations:
    - Opening (erosion → dilation): Remove isolated pixels
    - Closing (dilation → erosion): Fill small holes

    Pros: Improves existing method, easy to add
    Cons: Still starts with non-spatial k-means

    Best for: Enhancing your current pipeline
    """
    # Initial categorization (like k-means)
    categorized = np.zeros_like(image, dtype=int)
    categorized[image > threshold_dim] = 1
    categorized[image > threshold_bright] = 2

    # Create structuring element
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    struct = ndimage.iterate_structure(struct, kernel_size)

    # Process each category separately
    dim_mask = categorized == 1
    bright_mask = categorized == 2

    # Morphological opening: Remove small isolated regions
    dim_opened = binary_erosion(dim_mask, structure=struct)
    dim_opened = binary_dilation(dim_opened, structure=struct)

    bright_opened = binary_erosion(bright_mask, structure=struct)
    bright_opened = binary_dilation(bright_opened, structure=struct)

    # Morphological closing: Fill small holes
    dim_closed = binary_dilation(dim_opened, structure=struct)
    dim_closed = binary_erosion(dim_closed, structure=struct)

    bright_closed = binary_dilation(bright_opened, structure=struct)
    bright_closed = binary_erosion(bright_closed, structure=struct)

    # Combine
    cleaned = np.zeros_like(image, dtype=int)
    cleaned[dim_closed] = 1
    cleaned[bright_closed] = 2

    # Statistics
    stats = {
        'before': {
            'dim_pixels': np.sum(dim_mask),
            'bright_pixels': np.sum(bright_mask)
        },
        'after': {
            'dim_pixels': np.sum(dim_closed),
            'bright_pixels': np.sum(bright_closed)
        },
        'removed_noise': {
            'dim': np.sum(dim_mask) - np.sum(dim_closed),
            'bright': np.sum(bright_mask) - np.sum(bright_closed)
        }
    }

    return cleaned, stats


def method_5_region_growing(
    image: np.ndarray,
    seed_threshold: float = 2.0,
    growth_threshold: float = 0.5,
    max_diff: float = 0.5
) -> tuple[np.ndarray, int]:
    """
    Method 5: Region Growing from Seeds

    Process:
    1. Find seed pixels (high intensity)
    2. Grow regions by adding similar neighboring pixels
    3. Stop when neighbors are too different

    Pros: Guarantees spatial connectivity, intuitive
    Cons: Sensitive to seed selection

    Best for: When you trust high-intensity peaks as true ACh sites
    """
    # Find seed pixels (bright spots)
    seeds = image > seed_threshold
    labeled_seeds, num_seeds = label(seeds)

    # Initialize region map
    regions = np.zeros_like(image, dtype=int)
    visited = np.zeros_like(image, dtype=bool)

    # For each seed, grow region
    region_id = 1
    for seed_label in range(1, num_seeds + 1):
        # Get seed coordinates
        seed_coords = np.argwhere(labeled_seeds == seed_label)
        if len(seed_coords) == 0:
            continue

        seed_y, seed_x = seed_coords[0]
        seed_value = image[seed_y, seed_x]

        # Region growing using flood fill
        to_check = [(seed_y, seed_x)]
        regions[seed_y, seed_x] = region_id
        visited[seed_y, seed_x] = True

        while to_check:
            y, x = to_check.pop(0)

            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = y + dy, x + dx

                    # Check bounds
                    if ny < 0 or ny >= image.shape[0] or nx < 0 or nx >= image.shape[1]:
                        continue

                    # Skip if already visited
                    if visited[ny, nx]:
                        continue

                    # Check if neighbor is similar enough and above threshold
                    neighbor_value = image[ny, nx]
                    if (neighbor_value > growth_threshold and
                        abs(neighbor_value - seed_value) < max_diff):
                        regions[ny, nx] = region_id
                        visited[ny, nx] = True
                        to_check.append((ny, nx))

        region_id += 1

    # Categorize regions by intensity
    categorized = np.zeros_like(image, dtype=int)
    for rid in range(1, region_id):
        region_mask = regions == rid
        mean_intensity = np.mean(image[region_mask])

        if mean_intensity > 1.5:
            categorized[region_mask] = 2  # Bright
        else:
            categorized[region_mask] = 1  # Dim

    return categorized, num_seeds


def visualize_spatial_methods(image: np.ndarray, title: str = "Spatial Methods Comparison") -> plt.Figure:
    """
    Compare all spatial-aware methods side-by-side.
    """
    # Apply all methods
    cat1, stats1 = method_1_connected_components(image)
    cat2, labels2 = method_2_watershed(image)
    cat3, labels3 = method_3_dbscan_spatial(image)
    cat4, stats4 = method_4_morphological_cleanup(image)
    cat5, num_seeds5 = method_5_region_growing(image)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Original
    im0 = axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('INPUT: Z-scored Frame\n(Baseline-corrected intensity)',
                         fontweight='bold', fontsize=7)
    axes[0, 0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    cbar0.set_label('Z-score', fontsize=8)

    # Method 1: Connected Components
    im1 = axes[0, 1].imshow(cat1, cmap='viridis', vmin=0, vmax=2)
    axes[0, 1].set_title(
        f'METHOD 1: Connected Components\n'
        f'(Filters isolated pixels)\n'
        f'Found: {stats1["num_dim_regions"]} dim + {stats1["num_bright_regions"]} bright regions',
        fontweight='bold', fontsize=8
    )
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, ticks=[0, 1, 2])
    cbar1.set_ticklabels(['Background', 'Dim ACh', 'Bright ACh'], fontsize=7)

    # Method 2: Watershed
    im2 = axes[0, 2].imshow(cat2, cmap='viridis', vmin=0, vmax=2)
    axes[0, 2].set_title('METHOD 2: Watershed\n(Identifies individual ACh sites)\n(Good for overlapping regions)',
                         fontweight='bold', fontsize=8)
    axes[0, 2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, ticks=[0, 1, 2])
    cbar2.set_ticklabels(['Background', 'Dim ACh', 'Bright ACh'], fontsize=7)

    # Method 3: DBSCAN
    im3 = axes[1, 0].imshow(cat3, cmap='viridis', vmin=0, vmax=2)
    axes[1, 0].set_title('METHOD 3: DBSCAN\n(Density-based clustering)\n(Best noise handling)',
                         fontweight='bold', fontsize=8)
    axes[1, 0].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, ticks=[0, 1, 2])
    cbar3.set_ticklabels(['Background', 'Dim ACh', 'Bright ACh'], fontsize=7)

    # Method 4: Morphological
    im4 = axes[1, 1].imshow(cat4, cmap='viridis', vmin=0, vmax=2)
    removed = stats4['removed_noise']
    axes[1, 1].set_title(
        f'METHOD 4: Morphological Cleanup\n'
        f'(Erosion + Dilation)\n'
        f'Removed: {removed["dim"] + removed["bright"]} isolated pixels',
        fontweight='bold', fontsize=8
    )
    axes[1, 1].axis('off')
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, ticks=[0, 1, 2])
    cbar4.set_ticklabels(['Background', 'Dim ACh', 'Bright ACh'], fontsize=7)

    # Method 5: Region Growing
    im5 = axes[1, 2].imshow(cat5, cmap='viridis', vmin=0, vmax=2)
    axes[1, 2].set_title(f'METHOD 5: Region Growing\n(Grows from bright seeds)\nSeeds: {num_seeds5}',
                         fontweight='bold', fontsize=8)
    axes[1, 2].axis('off')
    cbar5 = plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, ticks=[0, 1, 2])
    cbar5.set_ticklabels(['Background', 'Dim ACh', 'Bright ACh'], fontsize=7)

    plt.suptitle(f'{title}\nAll 5 Methods Consider Spatial Connectivity (Unlike K-means)',
                 fontsize=10, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig


def demonstrate_spatial_difference():
    """
    Demonstrate the difference between k-means (non-spatial) and spatial methods.
    """
    print("\n" + "="*80)
    print("WHY SPATIAL CONNECTIVITY MATTERS FOR ACh IMAGING")
    print("="*80)

    # Create synthetic image with:
    # 1. True ACh patch (spatially connected bright region)
    # 2. Noise (scattered bright pixels)

    image = np.random.randn(128, 128) * 0.3  # Background noise

    # True ACh release: spatially connected
    y, x = np.ogrid[-64:64, -64:64]
    true_ach = np.exp(-((x-20)**2 + (y-10)**2) / 150) * 3
    image += true_ach

    # Noise: scattered bright pixels
    noise_locations = np.random.choice(128*128, size=50, replace=False)
    noise_y = noise_locations // 128
    noise_x = noise_locations % 128
    image[noise_y, noise_x] += np.random.uniform(2, 3, size=50)

    # Method comparison
    from sklearn.cluster import KMeans

    # K-means (non-spatial)
    pixels = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=100, n_init=10)
    kmeans_labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)
    label_map = np.zeros(3, dtype=int)
    label_map[sorted_idx] = [0, 1, 2]
    kmeans_result = label_map[kmeans_labels].reshape(image.shape)

    # Spatial method (connected components)
    spatial_result, _ = method_1_connected_components(image, min_size=20)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(image, cmap='gray')
    axes[0].set_title('INPUT: Synthetic ACh Image\nTrue Patch (connected) + Noise (isolated)',
                      fontsize=9, fontweight='bold')
    axes[0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046)
    cbar0.set_label('Z-score', fontsize=8)

    im1 = axes[1].imshow(kmeans_result, cmap='viridis', vmin=0, vmax=2)
    axes[1].set_title('❌ K-means (CURRENT METHOD)\nNO Spatial Awareness\nClassifies isolated noise as "bright signal"',
                      fontsize=9, fontweight='bold', color='red')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, ticks=[0, 1, 2])
    cbar1.set_ticklabels(['Background', 'Dim', 'Bright'], fontsize=7)

    im2 = axes[2].imshow(spatial_result, cmap='viridis', vmin=0, vmax=2)
    axes[2].set_title('✓ Spatial Method (NEW)\nConsiders Connectivity\nRemoves isolated noise pixels',
                      fontsize=9, fontweight='bold', color='green')
    axes[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, ticks=[0, 1, 2])
    cbar2.set_ticklabels(['Background', 'Dim', 'Bright'], fontsize=7)

    plt.suptitle('K-means vs Spatial Methods: Why Spatial Connectivity Matters\n(Coherent patches vs scattered noise)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================================
# RECOMMENDATION FOR YOUR ACH IMAGING PIPELINE
# ============================================================================

def recommended_spatial_pipeline(zscore_frame: np.ndarray, min_region_size: int = 20) -> dict:
    """
    RECOMMENDED: Two-stage pipeline for ACh imaging

    Stage 1: Identify potential ACh regions (spatial)
    Stage 2: Categorize by intensity

    Returns dict with:
        - categorized: Final categorization (0=bg, 1=dim, 2=bright)
        - regions: Individual region labels
        - stats: Region statistics
    """
    # Stage 1: Find spatially connected regions
    threshold = 0.5  # Z-score threshold for ACh signal
    mask = zscore_frame > threshold

    # Get connected components
    labeled_regions, num_regions = label(mask)

    # Filter by size (remove noise)
    cleaned_regions = np.zeros_like(labeled_regions)
    region_stats = []

    valid_region_id = 1
    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        region_size = np.sum(region_mask)

        if region_size >= min_region_size:
            # Keep this region
            cleaned_regions[region_mask] = valid_region_id

            # Calculate statistics
            mean_intensity = np.mean(zscore_frame[region_mask])
            max_intensity = np.max(zscore_frame[region_mask])
            region_stats.append({
                'id': valid_region_id,
                'size': region_size,
                'mean_z': mean_intensity,
                'max_z': max_intensity
            })

            valid_region_id += 1

    # Stage 2: Categorize regions by mean intensity
    categorized = np.zeros_like(zscore_frame, dtype=int)

    for stat in region_stats:
        region_mask = cleaned_regions == stat['id']

        if stat['mean_z'] > 1.5:
            categorized[region_mask] = 2  # Bright ACh release
        else:
            categorized[region_mask] = 1  # Dim ACh release

    return {
        'categorized': categorized,
        'regions': cleaned_regions,
        'stats': region_stats,
        'num_valid_regions': len(region_stats)
    }


if __name__ == "__main__":
    print("="*80)
    print("SPATIAL-AWARE INTENSITY CATEGORIZATION FOR ACh IMAGING")
    print("="*80)

    # Generate synthetic ACh imaging data
    print("\nGenerating synthetic ACh imaging frame...")
    np.random.seed(42)
    image = np.random.randn(256, 256) * 0.5

    # Add ACh release patches (spatially connected)
    y, x = np.ogrid[-128:128, -128:128]

    # Bright patch
    bright_patch = np.exp(-((x-50)**2 + (y-30)**2) / 400) * 3
    image += bright_patch

    # Dim patch
    dim_patch = np.exp(-((x+40)**2 + (y+50)**2) / 600) * 1.5
    image += dim_patch

    # Another bright patch
    bright_patch2 = np.exp(-((x-30)**2 + (y-60)**2) / 300) * 2.5
    image += bright_patch2

    # Add some noise pixels (isolated)
    noise_idx = np.random.choice(256*256, size=100, replace=False)
    noise_y = noise_idx // 256
    noise_x = noise_idx % 256
    image[noise_y, noise_x] += np.random.uniform(1.5, 2.5, size=100)

    print(f"Image stats: Mean={image.mean():.2f}, Std={image.std():.2f}")

    # Demonstrate spatial difference
    print("\n" + "="*80)
    print("COMPARISON: K-means vs Spatial Methods")
    print("="*80)
    fig1 = demonstrate_spatial_difference()
    plt.savefig('spatial_vs_nonspatial_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: spatial_vs_nonspatial_comparison.png")

    # Compare all spatial methods
    print("\n" + "="*80)
    print("COMPARING ALL SPATIAL METHODS")
    print("="*80)
    fig2 = visualize_spatial_methods(image, "Spatial Methods for ACh Imaging")
    plt.savefig('spatial_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: spatial_methods_comparison.png")

    # Test recommended pipeline
    print("\n" + "="*80)
    print("RECOMMENDED PIPELINE FOR YOUR CODE")
    print("="*80)
    result = recommended_spatial_pipeline(image, min_region_size=20)

    print(f"\nFound {result['num_valid_regions']} spatially connected ACh regions:")
    for stat in result['stats']:
        category = 'BRIGHT' if stat['mean_z'] > 1.5 else 'DIM'
        print(f"  Region {stat['id']:2d}: {stat['size']:4d} pixels | "
              f"Mean z={stat['mean_z']:5.2f} | Max z={stat['max_z']:5.2f} | {category}")

    # Visualize recommended result
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(image, cmap='gray')
    axes[0].set_title('STEP 1: Input Z-scored Image\n(Your preprocessed data)',
                      fontsize=9, fontweight='bold')
    axes[0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046)
    cbar0.set_label('Z-score', fontsize=8)

    im1 = axes[1].imshow(result['regions'], cmap='nipy_spectral')
    axes[1].set_title(f'STEP 2: Identify Spatial Regions\n(Each color = 1 ACh patch)\nn={result["num_valid_regions"]} regions found',
                      fontsize=9, fontweight='bold')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046)
    cbar1.set_label('Region ID', fontsize=8)

    im2 = axes[2].imshow(result['categorized'], cmap='viridis', vmin=0, vmax=2)
    axes[2].set_title('STEP 3: Categorize by Intensity\n(Within each spatial region)\nBlue=Background | Green=Dim | Yellow=Bright',
                      fontsize=9, fontweight='bold')
    axes[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, ticks=[0, 1, 2])
    cbar2.set_ticklabels(['Background', 'Dim ACh', 'Bright ACh'], fontsize=7)

    plt.suptitle('RECOMMENDED: Two-Stage Spatial Pipeline\nStep 1: Find Regions → Step 2: Categorize', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('recommended_spatial_pipeline.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: recommended_spatial_pipeline.png")

    plt.show()

    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print("""
FOR YOUR ACh IMAGING ANALYSIS:

1. PROBLEM WITH CURRENT K-MEANS:
   ✗ Treats each pixel independently
   ✗ Isolated noise pixels get same weight as true signal
   ✗ No spatial coherence in identified regions

2. BEST SPATIAL-AWARE METHODS:

   ⭐⭐⭐ Method 1: Connected Components (RECOMMENDED)
       - Simple, fast, effective
       - Filters isolated noise automatically
       - Easy to integrate with your existing code
       - Use: method_1_connected_components(frame, min_size=20)

   ⭐⭐ Method 2: Watershed
       - Best for identifying individual ACh release sites
       - Good when patches touch/overlap
       - Use: method_2_watershed(frame)

   ⭐⭐ Method 3: DBSCAN
       - Sophisticated, handles complex patterns
       - Automatically finds clusters
       - Use: method_3_dbscan_spatial(frame)

   ⭐ Method 4: Morphological Post-processing
       - Quick fix for your current k-means
       - Add after existing code
       - Use: method_4_morphological_cleanup(frame)

   ⭐ Method 5: Region Growing
       - Best when you trust bright peaks
       - Grows regions from seeds
       - Use: method_5_region_growing(frame)

3. INTEGRATION INTO YOUR CODE:

   Replace this (line 1136-1140 in cluster_analysis.py):
   ```python
   clustered_frames, cluster_centers = process_segment_kmeans_concatenated(
       averaged_segment, n_clusters=3
   )
   ```

   With this:
   ```python
   from functions.spatial_processing import categorize_spatial_connected

   result = categorize_spatial_connected(
       averaged_segment,
       threshold_dim=0.5,
       threshold_bright=1.5,
       min_region_size=20
   )
   clustered_frames = [result['categorized']]
   ```

4. WHY THIS MATTERS FOR YOUR RESEARCH:
   - True ACh release forms spatially coherent patches
   - Noise creates random isolated pixels
   - Spatial methods = better signal-to-noise ratio
   - More biologically meaningful regions

NEXT STEPS:
1. Run this script: python spatial_intensity_categorization.py
2. Compare results with your current k-means
3. Choose method that best captures your ACh patches
4. I can help integrate it into cluster_analysis.py
    """)
