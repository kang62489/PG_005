import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set random seed for reproducibility
np.random.seed(42)

# Create 512x512 image
height, width = 512, 512
image = np.zeros((height, width))

# Define cluster centers and sizes
cluster1_center = (150, 200)  # Bright cluster 1
cluster1_radius = 60
cluster1_intensity = 200

cluster2_center = (350, 300)  # Bright cluster 2  
cluster2_radius = 80
cluster2_intensity = 180

background_intensity = 50

# Create coordinate grids
y, x = np.ogrid[:height, :width]

# Create cluster 1 (circular bright region)
dist1 = np.sqrt((x - cluster1_center[0])**2 + (y - cluster1_center[1])**2)
mask1 = dist1 <= cluster1_radius
image[mask1] = cluster1_intensity

# Create cluster 2 (circular bright region)
dist2 = np.sqrt((x - cluster2_center[0])**2 + (y - cluster2_center[1])**2)
mask2 = dist2 <= cluster2_radius
image[mask2] = cluster2_intensity

# Set background for remaining pixels
background_mask = ~(mask1 | mask2)
image[background_mask] = background_intensity

# Add some noise to make it realistic
noise = np.random.normal(0, 10, (height, width))
image = image + noise
image = np.clip(image, 0, 255)  # Keep values in valid range

# Visualize the original image
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image (3 clusters)')
plt.colorbar()

# Now do k-means clustering to recover the 3 clusters
# Reshape image to 1D array for clustering
pixels = image.reshape(-1, 1)

# Apply k-means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(pixels)

# Reshape back to image shape
clustered_image = cluster_labels.reshape(height, width)

plt.subplot(1, 3, 2)
plt.imshow(clustered_image, cmap='viridis')
plt.title('K-means Result (k=3)')
plt.colorbar()

# Show cluster centers (mean intensities)
centers = kmeans.cluster_centers_.flatten()
plt.subplot(1, 3, 3)
plt.bar(range(3), centers)
plt.title('Cluster Centers (Mean Intensities)')
plt.xlabel('Cluster')
plt.ylabel('Intensity')

plt.tight_layout()
plt.show()

print("Cluster centers (mean intensities):")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center:.1f}")