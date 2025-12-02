import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#helper function to create the "elbow chart" for cluster optimization
def find_opt_clusters( scale, test_min = 2, test_max = 5, graph = True):
    #initialize graph values
    ss_score_vals = []
    inert_scores = []

    #initialize return values
    best_k = 1
    best_score = -math.inf
    best_score_ratio = -1

    for vals in range( test_min, test_max + 1 ):
        #run KMeans++ for the test value K
        test_kmeans = KMeans(n_clusters=vals, init='k-means++', n_init=10, random_state=42)
        test_kmeans.fit( scale )
        #get the inertia score
        inert_scores.append(test_kmeans.inertia_)
        #get the SS score
        test_score = silhouette_score(scale, test_kmeans.labels_)
        ss_score_vals.append(test_score)

        test_score_ratio = test_score / test_score

        #check for optimum score
        if( test_score_ratio > best_score_ratio ):
            #set the return values
            best_score = test_score
            best_score_ratio = test_score_ratio
            best_k = vals

    #once all points have been passed handle display
    if graph:
        plt.figure(figsize=(12,5))
        # Inertia (Elbow) plot
        plt.subplot(1,2,1)
        plt.plot(range(test_min, test_max+1), inert_scores, 'o-', color='blue')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        # Silhouette plot
        plt.subplot(1,2,2)
        plt.plot(range(test_min, test_max+1), ss_score_vals, 'o-', color='green')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
        plt.legend()
        plt.tight_layout()
        plt.show()       

    #otherwise assume best K is max
    return best_k

#create the data frame to hold all song data for the scan
data_frame = pd.read_csv("acoustic_features.csv", sep="\t")
#collect song samples from CSV
sample = data_frame.sample(n=1000, random_state=42).reset_index(drop=True)

#drop the ID for easier processing
numeric_sample = sample.drop( columns = ["song_id"] )

#scale the data for processing
scaler = StandardScaler()
scaled_songs = scaler.fit_transform( numeric_sample )

# find the optimum number of clusters for the sample
n_clusters = find_opt_clusters( scaled_songs, test_min = 2, test_max = 500, graph = True)

#run the KMeans++ clustering algorithm
kmeans_clusters = KMeans( n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42 )

#fit the data to the scale
kmeans_clusters.fit( scaled_songs )

#label the clusters
sample['cluster'] = kmeans_clusters.labels_

#set efficiency variables
ss_score = silhouette_score( scaled_songs, kmeans_clusters.labels_ )
inertia = kmeans_clusters.inertia_


#KMeans++ stats display
print("KMeans++ Results:")
print("----------------------------")
print(f"Silhouette Score: {ss_score:.3f}")
print(f"Inertia:   {inertia}")

#Graphing components
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_songs)

plt.figure(figsize=(8,6))
plt.scatter( reduced[:, 0], reduced[:, 1], c=sample["cluster"], cmap="viridis", s=10, alpha=0.9 )
plt.title(f'K-Means++ Clusters (PCA projection): k = {n_clusters}')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
