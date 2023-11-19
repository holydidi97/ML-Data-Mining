#Project LSI Dimitra_Kassari June 2023
#Part_B_Clustering

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances

# Load the dataset
df = pd.read_csv('bc_no0.csv')

X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)
n_clusters = 2

# Perform K-means clustering 10 times
n_runs = 10
dunn_scores = []
silhouette_scores = []
purity_scores = []
rand_scores = []

for i in range(n_runs):
    # Initialize K-means algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=i)
    
    # Fit the model to the data
    kmeans.fit(X)
    
    # Calculate Dunn index
    centroids = kmeans.cluster_centers_
    distances = pairwise_distances(X, centroids)
    min_distances = np.min(distances, axis=1)
    max_intracluster_distance = np.max(min_distances)
    intercluster_distances = pairwise_distances(centroids)
    min_intercluster_distance = np.min(intercluster_distances[np.nonzero(intercluster_distances)])
    dunn_index = min_intercluster_distance / max_intracluster_distance
    dunn_scores.append(dunn_index)
    
    # Calculate Silhouette coefficient
    labels = kmeans.labels_
    silhouette_coefficient = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_coefficient)
    
    # Calculate Purity
    true_labels = df['gene_type']
    contingency_mat = contingency_matrix(true_labels, labels)
    purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
    purity_scores.append(purity)
    
    # Calculate Rand measure
    rand_index = adjusted_rand_score(true_labels, labels)
    rand_scores.append(rand_index)

# Save the metrics to a text file
with open('kmeans_metrics.txt', 'w') as file:
    file.write("Dunn Index:\n")
    file.write(str(dunn_scores))
    file.write("\n\nSilhouette Coefficient:\n")
    file.write(str(silhouette_scores))
    file.write("\n\nPurity:\n")
    file.write(str(purity_scores))
    file.write("\n\nRand Measure:\n")
    file.write(str(rand_scores))

print("Metrics saved to kmeans_metrics.txt")


import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


df = pd.read_csv('bc_no0.csv')


X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)


n_clusters = 2  
linkage = 'ward'  


metrics_file = open('ward_metrics.txt', 'w')
for i in range(10):
   
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clustering.fit(X)

    
    y_pred = clustering.labels_

    
    dunn_index = metrics.davies_bouldin_score(X, y_pred)
    silhouette_coefficient = metrics.silhouette_score(X, y_pred)
    purity = metrics.accuracy_score(df['gene_type'], y_pred)
    rand_measure = metrics.adjusted_rand_score(df['gene_type'], y_pred)

   
    metrics_file.write(f"Run {i+1}:\n")
    metrics_file.write(f"Dunn Index: {dunn_index}\n")
    metrics_file.write(f"Silhouette Coefficient: {silhouette_coefficient}\n")
    metrics_file.write(f"Purity: {purity}\n")
    metrics_file.write(f"Rand Measure: {rand_measure}\n")
    metrics_file.write("\n")


metrics_file.close()


import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

# Load the dataset
df = pd.read_csv('bc_no0.csv')

# Split the dataset into features and target variable
X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)


n_clusters = 2  
linkage = 'single' 


metrics_file = open('slink_metrics.txt', 'w')
for i in range(10):
 
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clustering.fit(X)

    # Perform predictions
    y_pred = clustering.labels_

 
    dunn_index = metrics.davies_bouldin_score(X, y_pred)
    silhouette_coefficient = metrics.silhouette_score(X, y_pred)
    purity = metrics.accuracy_score(df['gene_type'], y_pred)
    rand_measure = metrics.adjusted_rand_score(df['gene_type'], y_pred)


    metrics_file.write(f"Run {i+1}:\n")
    metrics_file.write(f"Dunn Index: {dunn_index}\n")
    metrics_file.write(f"Silhouette Coefficient: {silhouette_coefficient}\n")
    metrics_file.write(f"Purity: {purity}\n")
    metrics_file.write(f"Rand Measure: {rand_measure}\n")
    metrics_file.write("\n")


metrics_file.close()


import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


df = pd.read_csv('bc_no0.csv')


X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)


n_clusters = 2  
linkage = 'complete' 


metrics_file = open('maxlink_metrics.txt', 'w')
for i in range(10):
    
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clustering.fit(X)

   
    y_pred = clustering.labels_

    
    dunn_index = metrics.davies_bouldin_score(X, y_pred)
    silhouette_coefficient = metrics.silhouette_score(X, y_pred)
    purity = metrics.accuracy_score(df['gene_type'], y_pred)
    rand_measure = metrics.adjusted_rand_score(df['gene_type'], y_pred)

   
    metrics_file.write(f"Run {i+1}:\n")
    metrics_file.write(f"Dunn Index: {dunn_index}\n")
    metrics_file.write(f"Silhouette Coefficient: {silhouette_coefficient}\n")
    metrics_file.write(f"Purity: {purity}\n")
    metrics_file.write(f"Rand Measure: {rand_measure}\n")
    metrics_file.write("\n")


metrics_file.close()


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics


df = pd.read_csv('bc_no0.csv')


X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)


eps = 0.5  
min_samples = 5  


metrics_file = open('dbscan_metrics.txt', 'w')
for i in range(10):
 
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clustering.fit(X)

   
    y_pred = clustering.labels_

  
    dunn_index = metrics.davies_bouldin_score(X, y_pred)
    silhouette_coefficient = metrics.silhouette_score(X, y_pred)
    purity = metrics.accuracy_score(df['gene_type'], y_pred)
    rand_measure = metrics.adjusted_rand_score(df['gene_type'], y_pred)

    
    metrics_file.write(f"Run {i+1}:\n")
    metrics_file.write(f"Dunn Index: {dunn_index}\n")
    metrics_file.write(f"Silhouette Coefficient: {silhouette_coefficient}\n")
    metrics_file.write(f"Purity: {purity}\n")
    metrics_file.write(f"Rand Measure: {rand_measure}\n")
    metrics_file.write("\n")


metrics_file.close()


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("dunn_index.xlsx")


algorithms = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]
columns = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]


accuracy = []


for col in columns:
    accuracy.append(df[col].values)


plt.boxplot(accuracy)
plt.xticks(range(1, len(algorithms) + 1), algorithms)
plt.ylabel("dunn_index")


plt.title("Dunn_Indeex Comparison of  Kmeans, Ward, Slink, Maxlink, DBSCAN")
plt.savefig("dn_boxplot.png")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("purity.xlsx")


algorithms = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]
columns = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]


accuracy = []


for col in columns:
    accuracy.append(df[col].values)


plt.boxplot(accuracy)
plt.xticks(range(1, len(algorithms) + 1), algorithms)
plt.ylabel("purity")


plt.title("Purity Comparison of  Kmeans, Ward, Slink, Maxlink, DBSCAN")
plt.savefig("purity_boxplot.png")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("rm.xlsx")


algorithms = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]
columns = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]


accuracy = []


for col in columns:
    accuracy.append(df[col].values)


plt.boxplot(accuracy)
plt.xticks(range(1, len(algorithms) + 1), algorithms)
plt.ylabel("Rand Meassure")


plt.title("Rand Meassure Comparison of  Kmeans, Ward, Slink, Maxlink, DBSCAN")
plt.savefig("rm_boxplot.png")

plt.show()


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("sc.xlsx")


algorithms = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]
columns = [ "Kmeans", "Ward","Slink","Maxlink", "DBSCAN"]


accuracy = []


for col in columns:
    accuracy.append(df[col].values)


plt.boxplot(accuracy)
plt.xticks(range(1, len(algorithms) + 1), algorithms)
plt.ylabel("sihlouette_coefficient")


plt.title("Sihlouette_Coefficient Comparison of  Kmeans, Ward, Slink, Maxlink, DBSCAN")
plt.savefig("sc_boxplot.png")

plt.show()


