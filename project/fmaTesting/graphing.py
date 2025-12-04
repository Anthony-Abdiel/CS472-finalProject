import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

#DEBUGGING for HDBSCAN errors
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False


#display function for PCA graph
def plot_pca_scatter(X, labels, title="PCA Cluster Visualization"):
    """2D PCA scatter plot of labeled clusters."""
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='tab20', s=10)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

#display function for silhouette scores
def plot_silhouette(X, labels, title="Silhouette Plot"):
    """Silhouette visualization for any algorithm."""
    if len(set(labels)) < 2:
        print("Cannot plot silhouette — only one cluster produced.")
        return

    sil_values = silhouette_samples(X, labels)
    sil_avg = silhouette_score(X, labels)

    plt.figure(figsize=(8, 6))
    y_lower = 10

    # iterate through each label
    for cluster in np.unique(labels):
        cluster_sil = sil_values[labels == cluster]
        cluster_sil.sort()
        size_c = len(cluster_sil)
        y_upper = y_lower + size_c

        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_sil, alpha=0.7)
        plt.text(0, (y_lower + y_upper) / 2, str(cluster))
        y_lower = y_upper + 10

    #display graph
    plt.axvline(x=sil_avg, color="red", linestyle="--")
    plt.title(title + f"  (avg={sil_avg:.3f})")
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    plt.show()

#elbow graph for KMEANS
def plot_elbow_kmeans(X, k_range=range(2, 15)):
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), inertias, marker='o')
    plt.title("KMeans Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


#calculates the silhouette score on a range of clusters (KMEANS)
def plot_silhouette_vs_k(X, k_range=range(2, 15)):
    sils = []

    #iterate through each cluster
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10)
        labels = km.fit_predict(X)
        sils.append(silhouette_score(X, labels))

    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), sils, marker='o')
    plt.title("KMeans Silhouette Score vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

#display function for GMM
def plot_gmm_aic_bic(X, k_range=range(2, 15)):
    aics, bics = [], []

    for k in k_range:
        gmm = GaussianMixture(n_components=k)
        gmm.fit(X)
        aics.append(gmm.aic(X))
        bics.append(gmm.bic(X))

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, aics, label="AIC", marker='o')
    plt.plot(k_range, bics, label="BIC", marker='o')
    plt.title("GMM AIC/BIC Curve")
    plt.xlabel("Components (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

#display function for agglomerative
def plot_dendrogram(X, sample_size=300, method="ward"):
    """Plot dendrogram on a subset of the data."""
    if len(X) > sample_size:
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        X = X[idx]

    Z = linkage(X, method=method)
    #display
    plt.figure(figsize=(12, 5))
    dendrogram(Z, no_labels=True)
    plt.title("Clustering Dendrogram")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

#display function for DBSCAN
def plot_dbscan_k_distance(X, k=5):
    """Plot the k-distance graph used to choose DBSCAN eps."""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    k_dist = np.sort(distances[:, k-1])

    plt.figure(figsize=(8, 6))
    plt.plot(k_dist)
    plt.title(f"DBSCAN k-distance Plot (k={k})")
    plt.ylabel("Distance to kth nearest neighbor")
    plt.xlabel("Points sorted by distance")
    plt.grid(True)
    plt.show()


#HDBSCAN display function
def plot_hdbscan_tree(clusterer):
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not installed.")
        return

    clusterer.condensed_tree_.plot(select_clusters=True,
                                   label_clusters=True)
    plt.title("HDBSCAN Condensed Cluster Tree")
    plt.show()
