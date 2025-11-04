"""
K-means Clustering Demonstration

Simple demonstration of k-means algorithm with a 64x64 image showing:
1. Initial centroid placement
2. First few iterations (assignment and update steps)
3. Final clustering results
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def create_simple_image(size: int = 64) -> np.ndarray:
    """
    Create a simple 64x64 synthetic image with 3 distinct regions.

    Returns:
        image: 2D array with 3 regions of different intensities
    """
    image = np.zeros((size, size))

    # Region 1: Background (low intensity ~0.2)
    image[:, :] = 0.2 + np.random.normal(0, 0.05, (size, size))

    # Region 2: Medium intensity circle in upper-left (~0.5)
    y, x = np.ogrid[:size, :size]
    circle1_mask = (x - 20)**2 + (y - 20)**2 <= 15**2
    image[circle1_mask] = 0.5 + np.random.normal(0, 0.05, np.sum(circle1_mask))

    # Region 3: High intensity circle in lower-right (~0.8)
    circle2_mask = (x - 44)**2 + (y - 44)**2 <= 15**2
    image[circle2_mask] = 0.8 + np.random.normal(0, 0.05, np.sum(circle2_mask))

    return np.clip(image, 0, 1)


def kmeans_step_by_step(data: np.ndarray, n_clusters: int = 3, max_iterations: int = 10, random_state: int = 42):
    """
    Perform k-means clustering with detailed step-by-step output.

    Args:
        data: Flattened pixel data (n_pixels, 1)
        n_clusters: Number of clusters
        max_iterations: Maximum iterations to run
        random_state: Random seed for reproducibility

    Returns:
        labels: Final cluster assignments
        centers: Final cluster centers
        history: Dictionary containing iteration history
    """
    np.random.seed(random_state)

    # Step 1: Initialize centroids randomly from data points
    n_samples = data.shape[0]
    random_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centers = data[random_indices].copy()

    print("\n" + "="*70)
    print("K-MEANS CLUSTERING DEMONSTRATION")
    print("="*70)
    print(f"\nDataset: {n_samples} pixels, {n_clusters} clusters")
    print(f"\nStep 1: INITIALIZATION")
    print("-" * 70)
    print(f"Randomly selected {n_clusters} data points as initial centroids:")
    for i, center in enumerate(centers):
        print(f"  Centroid {i}: {center[0]:.4f}")

    history = {
        'centers': [centers.copy()],
        'labels': [],
        'distances': []
    }

    for iteration in range(max_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*70}")

        # Step 2: Assignment step - assign each point to nearest centroid
        distances = np.zeros((n_samples, n_clusters))
        for i, center in enumerate(centers):
            distances[:, i] = np.abs(data[:, 0] - center[0])

        labels = np.argmin(distances, axis=1)
        avg_distance = np.mean(np.min(distances, axis=1))

        print(f"\nStep 2a: ASSIGNMENT")
        print("-" * 70)
        print("Assigning each pixel to nearest centroid...")
        for i in range(n_clusters):
            count = np.sum(labels == i)
            print(f"  Cluster {i}: {count} pixels assigned ({count/n_samples*100:.1f}%)")
        print(f"  Average distance to nearest centroid: {avg_distance:.4f}")

        history['labels'].append(labels.copy())
        history['distances'].append(avg_distance)

        # Step 3: Update step - calculate new centroids
        old_centers = centers.copy()
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centers[i] = cluster_points.mean()

        print(f"\nStep 2b: UPDATE CENTROIDS")
        print("-" * 70)
        print("Calculating new centroids as mean of assigned points:")
        for i in range(n_clusters):
            change = abs(centers[i, 0] - old_centers[i, 0])
            print(f"  Centroid {i}: {old_centers[i, 0]:.4f} -> {centers[i, 0]:.4f} (change: {change:.4f})")

        history['centers'].append(centers.copy())

        # Check for convergence
        center_change = np.max(np.abs(centers - old_centers))
        print(f"\nMax centroid change: {center_change:.6f}")

        if center_change < 1e-4:
            print(f"\nConverged after {iteration + 1} iterations!")
            break

        # Only show detailed output for first 3 iterations
        if iteration >= 2 and iteration < max_iterations - 1:
            print("\n... (continuing iterations until convergence) ...")
            # Continue silently
            continue

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nFinal cluster centers (sorted by intensity):")
    sorted_indices = np.argsort(centers[:, 0])
    for rank, i in enumerate(sorted_indices):
        count = np.sum(labels == i)
        print(f"  Cluster {i} (intensity {centers[i, 0]:.4f}): {count} pixels")

    return labels, centers, history


def visualize_results(image: np.ndarray, labels: np.ndarray, centers: np.ndarray, history: dict):
    """
    Visualize original image, clustering results, and convergence.
    """
    height, width = image.shape
    clustered_image = labels.reshape(height, width)

    # Sort clusters by intensity for consistent coloring
    sorted_indices = np.argsort(centers[:, 0])
    label_mapping = np.zeros(len(centers), dtype=int)
    label_mapping[sorted_indices] = np.arange(len(centers))
    clustered_image_sorted = label_mapping[clustered_image]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image (64x64)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Clustered image
    colors = ['darkblue', 'yellow', 'red']
    cmap = ListedColormap(colors[:len(centers)])
    im = axes[0, 1].imshow(clustered_image_sorted, cmap=cmap, vmin=0, vmax=len(centers)-1)
    axes[0, 1].set_title('K-means Clustering Result', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0, 1], ticks=range(len(centers)))
    cbar.set_label('Cluster ID', rotation=270, labelpad=20)

    # Histogram showing data distribution and centroids
    axes[1, 0].hist(image.flatten(), bins=50, alpha=0.7, color='gray', edgecolor='black')
    colors_line = ['darkblue', 'gold', 'red']
    for i, center in enumerate(centers):
        axes[1, 0].axvline(center[0], color=colors_line[i], linewidth=3,
                          linestyle='--', label=f'Centroid {i}')
    axes[1, 0].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Pixel Distribution with Final Centroids', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Convergence plot
    iterations = range(len(history['distances']))
    axes[1, 1].plot(iterations, history['distances'], 'o-', linewidth=2, markersize=8, color='darkgreen')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Average Distance to Centroid', fontsize=12)
    axes[1, 1].set_title('Convergence: Distance Over Iterations', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(iterations)

    plt.tight_layout()
    plt.savefig('kmeans_demo_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'kmeans_demo_results.png'")
    # plt.show()  # Commented out to avoid blocking


def main():
    """Main demonstration function."""
    print("\nGenerating synthetic 64x64 image with 3 distinct regions...")
    image = create_simple_image(size=64)

    print(f"Image shape: {image.shape}")
    print(f"Pixel value range: [{image.min():.3f}, {image.max():.3f}]")

    # Flatten image for k-means
    data = image.reshape(-1, 1)

    # Run k-means with step-by-step output
    labels, centers, history = kmeans_step_by_step(data, n_clusters=3, max_iterations=10)

    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(image, labels, centers, history)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
