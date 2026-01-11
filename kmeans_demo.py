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
    Visualize step-by-step k-means iterations showing pixel assignments and centroid updates.
    """
    height, width = image.shape
    n_iterations = len(history['labels'])
    colors = ['darkblue', 'yellow', 'red']
    cmap = ListedColormap(colors[:len(centers)])
    
    # Create figure with subplots for each iteration plus original and convergence
    n_cols = min(4, n_iterations + 2)  # Max 4 columns
    n_rows = max(2, (n_iterations + 3) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()
    
    plot_idx = 0
    
    # Original image
    axes_flat[plot_idx].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes_flat[plot_idx].set_title('Original Image', fontsize=12, fontweight='bold')
    axes_flat[plot_idx].axis('off')
    plot_idx += 1
    
    # Show each iteration
    for iter_idx in range(n_iterations):
        if plot_idx >= len(axes_flat):
            break
            
        # Get labels for this iteration
        iter_labels = history['labels'][iter_idx]
        clustered_image = iter_labels.reshape(height, width)
        
        # Sort clusters by their centroid values for consistent coloring
        current_centers = history['centers'][iter_idx + 1]  # +1 because centers[0] is initial
        sorted_indices = np.argsort(current_centers[:, 0])
        label_mapping = np.zeros(len(current_centers), dtype=int)
        label_mapping[sorted_indices] = np.arange(len(current_centers))
        clustered_image_sorted = label_mapping[clustered_image]
        
        im = axes_flat[plot_idx].imshow(clustered_image_sorted, cmap=cmap, vmin=0, vmax=len(centers)-1)
        
        # Show centroid values in title
        center_vals = [f"{current_centers[i, 0]:.3f}" for i in sorted_indices]
        title = f"Iteration {iter_idx + 1}\nCentroids: {', '.join(center_vals)}"
        axes_flat[plot_idx].set_title(title, fontsize=10, fontweight='bold')
        axes_flat[plot_idx].axis('off')
        plot_idx += 1
    
    # Convergence plot
    if plot_idx < len(axes_flat):
        iterations = range(len(history['distances']))
        axes_flat[plot_idx].plot(iterations, history['distances'], 'o-', linewidth=2, markersize=6, color='darkgreen')
        axes_flat[plot_idx].set_xlabel('Iteration', fontsize=10)
        axes_flat[plot_idx].set_ylabel('Avg Distance', fontsize=10)
        axes_flat[plot_idx].set_title('Convergence', fontsize=12, fontweight='bold')
        axes_flat[plot_idx].grid(True, alpha=0.3)
        axes_flat[plot_idx].set_xticks(iterations)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_detailed_steps(image: np.ndarray, labels: np.ndarray, centers: np.ndarray, history: dict):
    """
    Visualize detailed step-by-step process with pixel scatter plots.
    """
    data = image.flatten()
    n_iterations = len(history['labels'])
    colors = ['darkblue', 'orange', 'red']
    
    fig, axes = plt.subplots(2, n_iterations, figsize=(4*n_iterations, 8))
    if n_iterations == 1:
        axes = axes.reshape(-1, 1)
    
    for iter_idx in range(n_iterations):
        # Top row: Image with cluster assignments
        iter_labels = history['labels'][iter_idx]
        clustered_image = iter_labels.reshape(image.shape)
        
        # Sort for consistent coloring
        current_centers = history['centers'][iter_idx + 1]
        sorted_indices = np.argsort(current_centers[:, 0])
        label_mapping = np.zeros(len(current_centers), dtype=int)
        label_mapping[sorted_indices] = np.arange(len(current_centers))
        clustered_image_sorted = label_mapping[clustered_image]
        
        cmap = ListedColormap(colors[:len(centers)])
        axes[0, iter_idx].imshow(clustered_image_sorted, cmap=cmap, vmin=0, vmax=len(centers)-1)
        axes[0, iter_idx].set_title(f'Iteration {iter_idx + 1}\nCluster Assignments', fontsize=12)
        axes[0, iter_idx].axis('off')
        
        # Bottom row: Scatter plot of pixel values
        for cluster_id in range(len(centers)):
            cluster_mask = iter_labels == cluster_id
            cluster_pixels = data[cluster_mask]
            if len(cluster_pixels) > 0:
                # Use jitter for better visibility
                y_vals = np.random.normal(cluster_id, 0.05, len(cluster_pixels))
                axes[1, iter_idx].scatter(cluster_pixels, y_vals, 
                                        c=colors[label_mapping[cluster_id]], alpha=0.6, s=1)
        
        # Plot centroids
        for i, center_val in enumerate(current_centers[:, 0]):
            mapped_i = label_mapping[i]
            axes[1, iter_idx].scatter(center_val, mapped_i, c='black', s=100, marker='x', linewidth=3)
            axes[1, iter_idx].scatter(center_val, mapped_i, c=colors[mapped_i], s=60, marker='x', linewidth=2)
        
        axes[1, iter_idx].set_xlabel('Pixel Intensity')
        axes[1, iter_idx].set_ylabel('Cluster')
        axes[1, iter_idx].set_title(f'Pixel Distribution\nCentroids: {[f"{c[0]:.3f}" for c in current_centers[sorted_indices]]}')
        axes[1, iter_idx].set_yticks(range(len(centers)))
        axes[1, iter_idx].grid(True, alpha=0.3)
        axes[1, iter_idx].set_ylim(-0.5, len(centers)-0.5)
    
    plt.tight_layout()
    plt.show()


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
    print("\nGenerating step-by-step visualization...")
    visualize_results(image, labels, centers, history)
    
    print("\nGenerating detailed scatter plot visualization...")
    visualize_detailed_steps(image, labels, centers, history)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
