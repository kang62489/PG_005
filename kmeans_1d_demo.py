"""
1D K-means Clustering Demonstration

Simple demonstration of k-means algorithm with a series of numbers showing:
1. Initial data points and random centroids
2. Step-by-step iterations with assignment and update steps
3. Evolution of cluster assignments and centroid positions
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_data(n_points=100, random_state=42):
    """
    Generate 1D data with 3 distinct clusters.
    
    Returns:
        data: 1D array of data points
    """
    np.random.seed(random_state)
    
    # Create 3 clusters
    cluster1 = np.random.normal(10, 2, 30)   # Around 10
    cluster2 = np.random.normal(30, 3, 40)   # Around 30  
    cluster3 = np.random.normal(55, 2.5, 30) # Around 55
    
    data = np.concatenate([cluster1, cluster2, cluster3])
    np.random.shuffle(data)  # Shuffle to mix the clusters
    
    return data


def kmeans_1d_step_by_step(data, n_clusters=3, max_iterations=10, random_state=42):
    """
    Perform 1D k-means clustering with detailed step-by-step output.
    
    Returns:
        labels: Final cluster assignments
        centers: Final cluster centers
        history: Dictionary containing iteration history
    """
    np.random.seed(random_state)
    
    # Step 1: Initialize centroids randomly from data range
    data_min, data_max = data.min(), data.max()
    centers = np.random.uniform(data_min, data_max, n_clusters)
    
    print("\n" + "="*80)
    print("1D K-MEANS CLUSTERING DEMONSTRATION")
    print("="*80)
    print(f"\nDataset: {len(data)} data points, {n_clusters} clusters")
    print(f"Data range: [{data_min:.2f}, {data_max:.2f}]")
    print(f"\nStep 1: INITIALIZATION")
    print("-" * 80)
    print(f"Randomly initialized {n_clusters} centroids:")
    for i, center in enumerate(centers):
        print(f"  Centroid {i}: {center:.3f}")
    
    history = {
        'centers': [centers.copy()],
        'labels': [],
        'distances': [],
        'data': data.copy()
    }
    
    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*80}")
        
        # Step 2: Assignment step - assign each point to nearest centroid
        distances = np.abs(data[:, np.newaxis] - centers)
        labels = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        avg_distance = np.mean(min_distances)
        
        print(f"\nStep 2a: ASSIGNMENT")
        print("-" * 80)
        print("Assigning each data point to nearest centroid...")
        for i in range(n_clusters):
            count = np.sum(labels == i)
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                cluster_range = f"[{cluster_points.min():.2f}, {cluster_points.max():.2f}]"
                print(f"  Cluster {i}: {count} points {cluster_range} -> Centroid {centers[i]:.3f}")
            else:
                print(f"  Cluster {i}: 0 points -> Centroid {centers[i]:.3f}")
        print(f"  Average distance to nearest centroid: {avg_distance:.3f}")
        
        history['labels'].append(labels.copy())
        history['distances'].append(avg_distance)
        
        # Step 3: Update step - calculate new centroids
        old_centers = centers.copy()
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centers[i] = cluster_points.mean()
            # If no points assigned, keep the old centroid
        
        print(f"\nStep 2b: UPDATE CENTROIDS")
        print("-" * 80)
        print("Calculating new centroids as mean of assigned points:")
        for i in range(n_clusters):
            change = abs(centers[i] - old_centers[i])
            print(f"  Centroid {i}: {old_centers[i]:.3f} -> {centers[i]:.3f} (change: {change:.3f})")
        
        history['centers'].append(centers.copy())
        
        # Check for convergence
        center_change = np.max(np.abs(centers - old_centers))
        print(f"\nMax centroid change: {center_change:.6f}")
        
        if center_change < 1e-3:
            print(f"\nConverged after {iteration + 1} iterations!")
            break
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nFinal cluster centers (sorted):")
    sorted_indices = np.argsort(centers)
    for rank, i in enumerate(sorted_indices):
        count = np.sum(labels == i)
        cluster_points = data[labels == i]
        if count > 0:
            cluster_range = f"[{cluster_points.min():.2f}, {cluster_points.max():.2f}]"
            print(f"  Cluster {i} (center: {centers[i]:.3f}): {count} points {cluster_range}")
    
    return labels, centers, history


def visualize_kmeans_evolution(data, labels, centers, history):
    """
    Visualize the step-by-step evolution of k-means clustering.
    """
    n_iterations = len(history['labels'])
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Create figure with subplots for each iteration
    n_cols = min(3, n_iterations)
    n_rows = (n_iterations + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_iterations == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for iter_idx in range(n_iterations):
        ax = axes[iter_idx] if n_iterations > 1 else axes[0]
        
        # Get data for this iteration
        iter_labels = history['labels'][iter_idx]
        iter_centers = history['centers'][iter_idx + 1]  # +1 because centers[0] is initial
        
        # Plot data points colored by cluster assignment
        for cluster_id in range(len(iter_centers)):
            cluster_mask = iter_labels == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) > 0:
                # Add some jitter for visualization
                y_jitter = np.random.normal(cluster_id, 0.1, len(cluster_data))
                ax.scatter(cluster_data, y_jitter, c=colors[cluster_id], 
                          alpha=0.6, s=30, label=f'Cluster {cluster_id}')
        
        # Plot centroids
        for i, center in enumerate(iter_centers):
            ax.axvline(center, color=colors[i], linewidth=3, linestyle='--', alpha=0.8)
            ax.scatter(center, i, c='black', s=200, marker='x', linewidth=4)
            ax.scatter(center, i, c=colors[i], s=100, marker='x', linewidth=3)
        
        ax.set_title(f'Iteration {iter_idx + 1}\nCentroids: {[f"{c:.2f}" for c in iter_centers]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Data Value')
        ax.set_ylabel('Cluster')
        ax.set_yticks(range(len(iter_centers)))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, len(iter_centers) - 0.5)
        
        if iter_idx == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_iterations, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_convergence(history):
    """
    Show convergence plots and centroid movement.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Average distance convergence
    iterations = range(len(history['distances']))
    ax1.plot(iterations, history['distances'], 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Distance to Centroid')
    ax1.set_title('Convergence: Distance Over Iterations', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iterations)
    
    # Plot 2: Centroid movement over iterations
    colors = ['red', 'blue', 'green']
    for centroid_id in range(len(history['centers'][0])):
        centroid_positions = [centers[centroid_id] for centers in history['centers']]
        iterations_centers = range(len(centroid_positions))
        ax2.plot(iterations_centers, centroid_positions, 'o-', 
                linewidth=2, markersize=8, color=colors[centroid_id], 
                label=f'Centroid {centroid_id}')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Centroid Position')
    ax2.set_title('Centroid Movement Over Iterations', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(history['centers'])))
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration function."""
    print("\nGenerating 1D dataset with 100 points in 3 clusters...")
    data = generate_data(n_points=100)
    
    print(f"Data statistics:")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Mean: {data.mean():.2f}")
    print(f"  Std: {data.std():.2f}")
    
    # Run k-means with step-by-step output
    labels, centers, history = kmeans_1d_step_by_step(data, n_clusters=3, max_iterations=10)
    
    # Visualize results
    print("\nGenerating step-by-step visualization...")
    visualize_kmeans_evolution(data, labels, centers, history)
    
    print("\nGenerating convergence plots...")
    visualize_convergence(history)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()