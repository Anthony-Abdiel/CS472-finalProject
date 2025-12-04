import utils
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap

from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)   # safer than None
pd.set_option('display.max_colwidth', None)

#tracks = utils.load('fma_metadata/tracks.csv')
echonest = utils.load('fma_metadata/echonest.csv')
#features = utils.load('fma_metadata/features.csv')

#collapsing colums from a hierarchical structure into a single flat list
echonest.columns = ['_'.join(col) for col in echonest.columns]


#echonest.columns = ['_'.join(col).strip() for col in echonest.columns]

#Filtering the audio and temporal features from the list of columns
audio_cols = [c for c in echonest.columns if "audio_features" in c]
temporal_cols = [c for c in echonest.columns if "temporal_features" in c]

#extracting the data from the columns
X_audio = echonest[audio_cols].astype(float)
X_temp = echonest[temporal_cols].astype(float)

#verifying we have the right columns
#print("Audio features:", audio_cols)
#print("Temporal features:", temporal_cols)


# Now we have nice, clean, and streamlined data! Next Up: Scaling and Dimensional Reduction

#defining a scaler for the temporal data
scaler_temp = StandardScaler()
X_temp_scaled = scaler_temp.fit_transform(X_temp)

#defining a scaler for the audio features data
scaler_audio = StandardScaler()
X_audio_scaled = scaler_audio.fit_transform(X_audio)


# The set of Temporal features is comprised of ~200 float values that are, themselves, dimensionally
# reduced values representing the evolution of the timbre over time.
# We need to reduce this number of features to an easier to cluster and more concise
# set of features that capture the esense of the data. ---> Enter: PCA

#Making the X data for the PCA only test
pca_temp = PCA(n_components=30, random_state=42)
X_temp_pca = pca_temp.fit_transform(X_temp_scaled)

#Making the X data for the Hybrid approach
pca_temp_hybrid = PCA(n_components=80, random_state=42)
X_temp_hybrid = pca_temp_hybrid.fit_transform(X_temp_scaled)

#converting to dataframes
df_audio_scaled = pd.DataFrame(X_audio_scaled)
df_temp_scaled = pd.DataFrame(X_temp_scaled)
df_temp_pca = pd.DataFrame(X_temp_pca)
df_temp_hybrid = pd.DataFrame(X_temp_hybrid)


X_hybrid = pd.concat([df_audio_scaled, df_temp_hybrid], axis=1)

#making the X data for the UMAP only approach
X_all = pd.concat([df_audio_scaled, df_temp_scaled], axis=1)


umap_reducer = umap.UMAP(
    n_components=10,
    n_neighbors=30,
    min_dist=0.1,
    random_state=None
)


# Now we are ready to combine the features into one feature set
X_final_pca = np.concatenate([X_audio_scaled, X_temp_pca], axis=1)
X_final_umap = umap_reducer.fit_transform(X_all)
X_final_hybrid = umap_reducer.fit_transform(X_hybrid)


# Sweet! Now we have a 20-dimensional set of clean and descriptive data!

# Testing out the clustering algorithms now


try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️ HDBSCAN not installed; skipping HDBSCAN tests.")


def safe_silhouette(X, labels):
    """Compute silhouette score safely."""
    # labels must have at least 2 clusters
    if len(set(labels)) < 2:
        return float("nan")  # cannot compute silhouette
    return silhouette_score(X, labels)


def timed(func):
    """Decorator to measure runtime of clustering tests."""
    def wrapper(*args, **kwargs):
        start = time.time()
        labels = func(*args, **kwargs)
        end = time.time()
        return labels, end - start
    return wrapper


# -------------------- CLUSTERING METHODS --------------------

@timed
def test_kmeans(X, k=10):
    km = KMeans(n_clusters=k, init="random", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    sil = safe_silhouette(X, labels)
    print(f"KMeans (k={k}): silhouette={sil:.4f}")
    return labels


@timed
def test_birch(X, k=10):
    birch = Birch(n_clusters=k)
    labels = birch.fit_predict(X)
    sil = safe_silhouette(X, labels)
    print(f"Birch (k={k}): silhouette={sil:.4f}")
    return labels


@timed
def test_gmm(X, k=10):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X)
    sil = safe_silhouette(X, labels)
    print(f"GMM (k={k}): silhouette={sil:.4f}")
    return labels


@timed
def test_agglomerative(X, k=10):
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X)
    sil = safe_silhouette(X, labels)
    print(f"Agglomerative (k={k}): silhouette={sil:.4f}")
    return labels


@timed
def test_dbscan(X, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    sil = safe_silhouette(X, labels)
    print(f"DBSCAN (eps={eps}, min_samples={min_samples}): silhouette={sil}")
    return labels


@timed
def test_hdbscan(X, min_cluster_size=15):
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available.")
        return np.zeros(len(X)), 0  # dummy result
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)
    sil = safe_silhouette(X, labels)
    print(f"HDBSCAN (min_cluster_size={min_cluster_size}): silhouette={sil}")
    return labels

def test_spectral(X, k=10):

    try:
        spectral = SpectralClustering(
            n_clusters=k,
            affinity='nearest_neighbors',  # MUCH better than RBF for high-dimensional audio
            n_neighbors=20,                # tweakable
            assign_labels='kmeans',        # stable default
            random_state=42
        )

        labels = spectral.fit_predict(X)

        # Compute silhouette only if meaningful
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            sil = silhouette_score(X, labels)
            print(f"Spectral (k={k}): silhouette={sil:.4f}")
        else:
            sil = float("nan")
            print("Spectral: silhouette=N/A (only one cluster produced).")

        return labels, sil

    except Exception as e:
        print("Spectral clustering failed:", e)
        return None, None
    
def test_kmeans_plus(X, k=10):
    km = KMeans(
        n_clusters=k,
        init="k-means++",  # explicitly use k-means++
        n_init=10,         # consistent with sklearn defaults
        random_state=42
    )
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"KMeans++ (k={k}): silhouette={sil:.4f}")
    return labels, sil

# -------------------- RUN ALL TESTS --------------------

def run_all_tests(X, k=10):
    print("\n=== Running Clustering Benchmarks ===\n")

    results = []

    tests = [
        ("KMeans", test_kmeans, {"k": k}),
        ("KMeans++", test_kmeans_plus, {"k": k}),
        ("Birch", test_birch, {"k": k}),
        ("GMM", test_gmm, {"k": k}),
        #("Spectral Clustering", test_spectral, {"k": k}),
        ("Agglomerative", test_agglomerative, {"k": k}),
        ("DBSCAN", test_dbscan, {"eps": 0.7, "min_samples": 5}),
        
    ]

    if HDBSCAN_AVAILABLE:
        tests.append(("HDBSCAN", test_hdbscan, {"min_cluster_size": 15}))

    for name, func, params in tests:
        print(f"\n--- {name} ---")
        labels, runtime = func(X, **params)
        sil = safe_silhouette(X, labels)
        results.append({
            "algorithm": name,
            "silhouette": sil,
            "runtime_sec": runtime
        })

    print("\n=== Benchmarking Complete ===\n")
    return results

print()
print()
print("-----------------Test 1: Using PCA for Dimensionality Reduction-----------------")
run_all_tests(X_final_pca, k=20)
print("-----------------End Test 1-----------------")
print()
print()

print("-----------------Test 2: Using UMAP for Dimensionality Reduction-----------------")
run_all_tests(X_final_umap, k=20)
print("-----------------End Test 2-----------------")
print()
print()

print("-----------------Test 3: Using PCA AND UMAP for Dimensionality Reduction-----------------")
run_all_tests(X_final_hybrid, k=20)
print("-----------------End Test 3-----------------")
print()
print()