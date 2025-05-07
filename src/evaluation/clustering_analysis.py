# %% [code]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def perform_kmeans(latent_vectors, num_clusters=3):
    """Run K-means clustering on latent space representations."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    return cluster_labels, kmeans.inertia_


def perform_dbscan(latent_vectors, eps=0.5, min_samples=5):
    """Run DBSCAN clustering on latent space representations."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(latent_vectors)
    return cluster_labels


def evaluate_clustering(latent_vectors, cluster_labels):
    """Compute silhouette score for clustering evaluation.

    Returns None if the clustering does not produce enough clusters.
    """
    unique_labels = set(cluster_labels)
    # Remove noise label (-1) if present for evaluation
    if -1 in unique_labels and len(unique_labels) > 1:
        unique_labels.remove(-1)
    if len(unique_labels) < 2:
        return None  # Not enough clusters for silhouette score
    try:
        return silhouette_score(latent_vectors, cluster_labels)
    except Exception as e:
        print("Silhouette score calculation error:", e)
        return None


def visualize_clusters(latent_vectors, cluster_labels, title):
    """Scatter plot for clustered latent representations."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=cluster_labels, cmap='viridis', s=20, alpha=0.7)
    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    plt.show()


if __name__ == "__main__":
    # Reduce dataset size to prevent memory issues: 200 samples, 8 latent dims.
    num_samples = 200
    latent_dim = 8
    latent_vectors = np.random.rand(num_samples, latent_dim).astype(np.float32)

    # ---- Run K-means clustering ----------------
    try:
        kmeans_labels, inertia = perform_kmeans(latent_vectors, num_clusters=3)
        kmeans_silhouette = evaluate_clustering(latent_vectors, kmeans_labels)
        print("K-means Inertia:", inertia)
        print("K-means Silhouette Score:", kmeans_silhouette)
        visualize_clusters(latent_vectors, kmeans_labels, "K-means Clustering of Latent Space")
    except Exception as e:
        print("K-means clustering encountered an error:", e)

    # ---- Run DBSCAN clustering ------------------
    try:
        dbscan_labels = perform_dbscan(latent_vectors, eps=0.5, min_samples=5)
        dbscan_silhouette = evaluate_clustering(latent_vectors, dbscan_labels)
        print("DBSCAN Silhouette Score:", dbscan_silhouette)
        visualize_clusters(latent_vectors, dbscan_labels, "DBSCAN Clustering of Latent Space")
    except Exception as e:
        print("DBSCAN clustering encountered an error:", e)
