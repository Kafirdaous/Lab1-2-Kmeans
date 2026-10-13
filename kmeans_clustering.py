import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from skimage import data

# Dataset 1: Generate 1000 points arranged around 4 centers.
X, _ = make_blobs(n_samples=1000, centers=4, random_state=42)

# Plot the generated points before clustering.
plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='gray', s=10)
plt.title('Generated Data Before K-Means')
plt.savefig('kmeans_data_before.png')
plt.close()

# Find 4 clusters and assign one cluster label to each point.
model = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = model.fit_predict(X)

# Plot the clusters and mark the cluster centers with red X symbols.
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            color='red', marker='X', s=180, label='Cluster Centers')
plt.title('Generated Data After K-Means')
plt.legend()
plt.savefig('kmeans_data_after.png')
plt.close()

# Dataset 2: Load and display the original coffee image.
coffee = data.coffee()
plt.figure()
plt.imshow(coffee)
plt.axis('off')
plt.title('Coffee Before K-Means')
plt.savefig('kmeans_image_before.png')
plt.close()

# Each row represents one RGB pixel.
pixels = coffee.reshape(-1, 3)

# Cluster the image into 4 representative RGB colors.
image_model = KMeans(n_clusters=4, random_state=42, n_init=10)
image_labels = image_model.fit_predict(pixels)

# Replace each pixel by the center color of its cluster.
new_pixels = image_model.cluster_centers_[image_labels]
new_coffee = np.clip(new_pixels.reshape(coffee.shape), 0, 255).astype(np.uint8)

plt.figure()
plt.imshow(new_coffee)
plt.axis('off')
plt.title('Coffee After K-Means (4 Colors)')
plt.savefig('kmeans_image_after.png')
plt.close()

print('K-Means complete: 4 PNG images saved.')
