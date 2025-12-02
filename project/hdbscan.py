import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#create the data frame to hold all song data for the scan
data_frame = pd.read_csv("acoustic_features.csv", sep="\t")

#drop the ID for easier processing
numeric_data_frame = data_frame.drop( columns = ["song_id"] )

#scale the data for processing
scaler = StandardScaler()
scaled_songs = scaler.fit_transform( numeric_data_frame )

#run the HDBSCAN
clustered_songs = hdbscan.HDBSCAN( min_cluster_size = 15, min_samples = 3,
                                                         metric = "euclidean" )

#label the clusters
labeled_clusters = clustered_songs.fit_predict( scaled_songs )

#add each cluster to the data frame
data_frame[ "hdbscan_cluster" ] = labeled_clusters
data_frame[ "cluster_probability" ] = clustered_songs.probabilities_

#collect cluster count values
n_clusters = 0

for clusters in labeled_clusters:
    #check for noise point
    if labeled_clusters[clusters] != -1:
        n_clusters += 1

#count the number of noise points identified within the data
n_noise = list(labeled_clusters).count(-1)

print("HDBSCAN Results:")
print("----------------------------")
print(f"Number of clusters found: {n_clusters}")
print(f"Number of noise points:   {n_noise}")
print("\nCluster sizes:")
print(pd.Series(labeled_clusters).value_counts())

#Graphing components
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_songs)

plt.figure(figsize=(8,6))
plt.scatter( reduced[:, 0], reduced[:, 1], c=labeled_clusters, cmap="viridis", s=10, alpha=0.9 )
plt.title("HDBSCAN Clusters (PCA projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
