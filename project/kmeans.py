import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Note: The expected columns are as follows:
# |Key|Mode|Time Signature|Acousticness|Danceability|Energy|Instrumentalness|Liveness|Loudness|Speechiness|Valence|Tempo|
# 
# 12 columns

def main():
    # read the data from the CSV file with all of the music features
    # and slice a subset out of it. Reset the indecies for readability
    data = pd.read_csv("acoustic_features.csv", sep="\t")
    dataSubset = data.sample(n=1000, random_state=42).reset_index(drop=True)

    print(dataSubset["tempo"])
    print()
    print()

    # specifying the specific features we want to include (Only numeric features)
    features = [ 'duration_ms', 'key', 'mode', 'time_signature', 'acousticness', 
                 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness']
    
    #creating a new data set containing only the specified columns
    X = dataSubset[features]

    #scaling the data to ensure all features contribute to clustering equally
    scaler = StandardScaler()
    X_Scaled = scaler.fit_transform(X)

    # run kMeans on the data
    kmeans = KMeans(n_clusters=8, random_state=42)
    kmeans.fit(X_Scaled)

    #adding a new column, the cluster label, for every song in the dataset
    dataSubset['cluster'] = kmeans.labels_

    # ---------------------- -  Visualization  - ---------------------------------------
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_Scaled)

    plt.scatter(reduced[:, 0], reduced[:, 1], c=dataSubset['cluster'], cmap='viridis', s=10)
    plt.title('K-Means Clusters (PCA projection)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  




if __name__ == '__main__':
    main()

