import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image as sk_image
from skimage import data
from skimage.transform import rescale
from skimage.filters import gaussian

# Dataset 1: Generate 1000 points in two concentric circles.
X, _ = make_circles(n_samples=1000, factor=0.5, noise=0.05,
                    random_state=42)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='gray', s=10)
plt.axis('equal')
plt.title('Generated Data Before Spectral Clustering')
plt.savefig('spectral_data_before.png')
plt.close()

# Identify the two circles as separate clusters.
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           random_state=42)
labels = model.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.axis('equal')
plt.title('Generated Data After Spectral Clustering')
plt.savefig('spectral_data_after.png')
plt.close()

# Dataset 2: Load and display the grayscale image of coins.
coins = data.coins()
plt.figure()
plt.imshow(coins, cmap='gray')
plt.axis('off')
plt.title('Coins Before Spectral Clustering')
plt.savefig('spectral_image_before.png')
plt.close()

# Smooth and downscale the image to make clustering manageable.
small_coins = rescale(gaussian(coins, sigma=2), 0.2)

# Each image pixel becomes a vertex in a graph of neighboring pixels.
graph = sk_image.img_to_graph(small_coins)
graph.data = np.exp(-10 * graph.data / graph.data.std())

# Cluster the graph into 25 regions.
region_labels = spectral_clustering(graph, n_clusters=25,
                                    assign_labels='discretize',
                                    random_state=42)
regions = region_labels.reshape(small_coins.shape)

plt.figure()
plt.imshow(regions, cmap='nipy_spectral')
plt.axis('off')
plt.title('Coins After Spectral Clustering (25 Regions)')
plt.savefig('spectral_image_after.png')
plt.close()

print('Spectral clustering complete: 4 PNG images saved.')
