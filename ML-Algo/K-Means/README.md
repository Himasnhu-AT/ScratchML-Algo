## K-Means Clustering Algorithm

### Packages used:

- `Numpy`
- `Pandas`

##### Usage of these pakages to vizualize data, and generate random data:

- `seaborn`
- `matplotlib.pyplot`
- `sklearn.datasets`
- `sklearn.metrics`
- `sklearn.preprocessing`
- `random`

### Brief explanation:

Unsupervised algorithm to group data points into `k` different clusters.

```python
class kmeans:

    def __init__(self, no_of_clusters, max_iters):
        self.no_of_clusters = no_of_clusters
        self.max_iters = max_iters

        self.clusters = [[] for i in range(self.no_of_clusters)]
        self.centroids = []

    #Find the centroid closest to a data point
    def find_nearest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    #Add a data point to the nearest centroid
    def assign_to_nearest_centroid(self, centroids):
        clusters = [[] for i in range(self.no_of_clusters)]
        for idx, sample in enumerate(self.x):
            nearest_centroid = self.find_nearest_centroid(sample, centroids)
            clusters[nearest_centroid].append(idx)
        return clusters

    #Update the value of the centroids (mean of coordinates of points in each cluster)
    def update_centroids(self, clusters):
        centroids = np.zeros((self.no_of_clusters, self.no_of_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids

    #Check if the new centroid coordiates are the same as the previous coordinates
    def check_for_convergence(self, current_centroids, centroids):
        distances = [euclidean_distance(current_centroids[i], centroids[i]) for i in range(self.no_of_clusters)]
        return sum(distances) == 0

    #Return the community structure
    def get_cluster_labels(self, clusters):
        labels = np.empty(self.no_of_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels


    def predict(self,x):
        self.x = x
        self.no_of_samples, self.no_of_features = x.shape

        #Initialize cluster centers randomly
        random_sample_indices = np.random.choice(self.no_of_samples, self.no_of_clusters, replace=False)
        self.centroids = [self.x[idx] for idx in random_sample_indices]

        #Modify cluster centers
        for i in range(self.max_iters):

            #assign points to clusters
            self.clusters = self.assign_to_nearest_centroid(self.centroids)

            #update centroid coordinatess
            current_centroids = self.centroids
            self.centroids = self.update_centroids(self.clusters)

            #check if the new and old centroid coordinates are the same
            if self.check_for_convergence(current_centroids, self.centroids):
                break


        #Finally, return the recovered community structure
        return self.get_cluster_labels(self.clusters)
```

This Python class kmeans implements the K-Means clustering algorithm. Here's how it works:

- The `__init__` method initializes the number of clusters, maximum iterations, clusters, and centroids.

- The `find_nearest_centroid` method calculates the Euclidean distance between a data point and all centroids, and returns the index of the closest centroid.

- The `assign_to_nearest_centroid` method assigns each data point to the cluster of the nearest centroid.

- The `update_centroids` method calculates the new centroids by computing the mean of all data points in each cluster.

- The `check_for_convergence` method checks if the centroids have stopped moving (i.e., the current centroids are the same as the previous centroids).

- The `get_cluster_labels` method assigns a label to each data point based on the cluster it belongs to.

- The `predict` method is the main method that applies the K-Means algorithm. It first initializes the centroids randomly. Then, in each iteration, it assigns data points to the nearest centroid, updates the centroids, and checks for convergence. If the centroids have converged, it breaks the loop. Finally, it returns the cluster labels of the data points.
