import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

#HARD CODE N_CLUSTERS HERE IF DESIRED
n_clusters = 15

#helper function to create the "elbow chart" for cluster optimization
def find_opt_clusters( scale, test_min = 2, test_max = 5, graph = True):
    #initialize graph values
    ss_score_vals = []

    #initialize return values
    best_k = 1
    best_score = -math.inf

    for vals in range( test_min, test_max + 1 ):
        #run KMeans++ for the test value K
        test_birch = Birch( n_clusters = vals )
        test_birch.fit( scale )
        #collect the labels from the test clusters
        test_labels = test_birch.predict(scale)
        #get the SS score
        test_score = silhouette_score(scale, test_labels)
        ss_score_vals.append(test_score)

        #check for optimum score
        if( test_score > best_score ):
            #set the return values
            best_score = test_score
            best_k = vals

    #once all points have been passed handle display
    if graph:
        plt.figure(figsize=(6,4))
        # display the chart
        plt.plot(range(test_min, test_max+1), ss_score_vals, 'o-', color='green')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        #label where the optimum K is selected
        plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
        plt.legend()
        plt.tight_layout()
        plt.show()       

    #otherwise assume best K is max
    print(f"The Optimum Silhouette Score is: { best_score }")
    return best_k

#set display heat map and dendrogram to default
display_extras = False

#create the data frame to hold all song data for the scan
data_frame = pd.read_csv("acoustic_features.csv", sep="\t")
#collect song samples from CSV
sample = data_frame.sample(n=1000, random_state=42).reset_index(drop=True)

#drop the ID for easier processing
numeric_sample = sample.drop( columns = ["song_id"] )

#scale the data for processing
scaler = StandardScaler()
scaled_songs = scaler.fit_transform( numeric_sample )

# find the optimum number of clusters for the sample if not hard coded
if( n_clusters is None ):
    n_clusters = find_opt_clusters( scaled_songs, test_min = 2, test_max = 500, graph = True)
#otherwise set the hard coded n_clusters to be the max
else:
    n_clusters = find_opt_clusters( scaled_songs, test_min = 2, test_max = n_clusters, graph = True)

#check for displayable cluster sizes
if( n_clusters < 15):
    display_extras = True

#run the BIRCH cluster
birch_clusters = Birch( n_clusters = n_clusters )
birch_clusters.fit( scaled_songs )

#label each cluster
birch_labels = birch_clusters.predict( scaled_songs )
sample["cluster"] = birch_labels

#BIRCH stats
pca = PCA(n_components = 2)
reduced = pca.fit_transform(scaled_songs)

#check for heatmap display
if( display_extras ):
    #set the scale for the heatmap display
    scaled_data_frame = pd.DataFrame(scaled_songs, columns=numeric_sample.columns)
    scaled_data_frame['cluster'] = birch_labels

    # calcualte the mean of each feature for each cluster
    cluster_means = scaled_data_frame.groupby('cluster').mean()

    # graph the heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(cluster_means, annot=True, cmap='viridis', fmt=".2f")
    plt.title("Average Feature Values per BIRCH Cluster")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

    #handle dendrogram display
    linked = linkage(scaled_songs, method='complete')
    plt.figure(figsize=(12, 6))
    dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=False, no_labels=True)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

#graph results
plt.figure(figsize=(8,6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=birch_labels, cmap="viridis", s=10, alpha=0.9)
plt.title(f'BIRCH Clusters (PCA projection): k = {n_clusters}')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()
