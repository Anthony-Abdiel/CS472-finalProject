import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#set BIC test values
test_min = 1
test_max = 500

#helper function to find the optimum BIC value within a set range
def find_optimumBIC(scaled_songs, test_min = 1, test_max = 5):
    #set initial values for return
    min_score = math.inf
    min_val = test_min
    prev_score = 0

    #iteritivley check each test value
    for val in range( test_min, test_max + 1):
        #run a GMM to test
        test_gmm = GaussianMixture( n_components=val, covariance_type='full', random_state=42 )
        #fit and scale the test GMM results
        test_gmm.fit(scaled_songs)
        bic = test_gmm.bic(scaled_songs)

        #check for optimum score
        if( bic < min_score ):
            #set the return values
            min_score = bic
            min_val = val

        #check for score increasing
        if( prev_score > bic ):
            #return the minimum value if less optimum values are recorded
            return min_val

        #increment prev score
        prev_score = val

    #return the minimum value
    return min_val


#create the data frame to hold all song data for the scan
data_frame = pd.read_csv("acoustic_features.csv", sep="\t")

#drop the ID for easier processing
numeric_data_frame = data_frame.drop( columns = ["song_id"] )

#scale the data for processing
scaler = StandardScaler()
scaled_songs = scaler.fit_transform( numeric_data_frame )

opt_comps = find_optimumBIC(scaled_songs, test_min, test_max )

#run the guassian clusters
gmm = GaussianMixture( n_components=opt_comps, covariance_type='full', random_state=42 )

#scale the data
gmm.fit(scaled_songs)

# abel each cluster
gmm_labels = gmm.predict(scaled_songs)

# calculate the efficiency of predictions
gmm_probabilities = gmm.predict_proba(scaled_songs).max(axis=1)

#insert each label into the frame
data_frame["gmm_cluster"] = gmm_labels
data_frame["gmm_probability"] = gmm_probabilities

#GMM cluster stats
print("GMM RESULTS")
print("----------------------------")
print( f"Number of clusters: { len( set( gmm_labels ) ) }" )
print("\nCluster sizes:")
print(pd.Series(gmm_labels).value_counts())

#variance levels detected / efficiency
print("\nModel Quality:")
print(f"AIC: {gmm.aic(scaled_songs)}")
print(f"BIC: {gmm.bic(scaled_songs)}")

#graphing the analysis data
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_songs)

#display the figure
plt.figure(figsize=(8, 6))
plt.scatter( reduced[:, 0], reduced[:, 1], c=gmm_labels, cmap="viridis", s=10, alpha=0.9 )
plt.title(f"GMM Clusters: { len( set( gmm_labels ) ) }")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
