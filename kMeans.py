import random 

# Function to calculatethe Euclidean distance between two points
def euclidean_distance(a, b):
    
    distance_squared = 0
    for i in range(len(a)):
        coordinate_difference_squared = (a[i] - b[i]) ** 2
        distance_squared += coordinate_difference_squared
    distance = distance_squared ** 0.5
    return distance

# Function to cluster the given data using N runs and return the best clusters
def kmeans(data, k, n_runs):
    
    best_clusters = None
    best_distortion = float('inf')
    
    for _ in range(n_runs):
        # Initialize centroids randomly
        centroids = random.sample(data, k)

        for iteration in range(100):
            # Assign each data point to its nearest centroid
            clusters = [[] for _ in range(k)]
            for point in data:
                distances = [euclidean_distance(point[:-1], centroid[:-1]) for centroid in centroids]
                nearest_centroid = distances.index(min(distances))
                clusters[nearest_centroid].append(point)

            new_centroids = []
            for cluster in clusters:
                
                if not cluster:
                    new_centroids.append(centroids[clusters.index(cluster)])
                else:

                    centroid = []
                    for i in range(len(cluster[0])):
                        values = [point[i] for point in cluster]
                        if isinstance(values[0], float):
                            centroid.append(sum(values) / len(cluster))
                    new_centroids.append(tuple(centroid + ['']))

            # Check if the centroids have converged
            if new_centroids == centroids:
                break

            centroids = new_centroids

        # Calculate distortion
        distortion = 0
        for i in range(len(centroids)):
            centroid_features = centroids[i][:-1]
            for j in range(len(clusters[i])):
                point_features = clusters[i][j][:-1]
                distance = euclidean_distance(centroid_features, point_features) ** 2
                distortion += distance

        # Check if this run is the best so far
        if distortion < best_distortion:
            best_clusters = clusters
            best_distortion = distortion

    return best_clusters


def predict_classes(test_data, clusters):

    # Determine majority class for each cluster
    cluster_classes = []
    for cluster in clusters:
        classes = [point[-1] for point in cluster]
        majority_class = max(set(classes), key=classes.count)
        cluster_classes.append(majority_class)
    
    # Predict the class of each data point in test_data
    predictions = []
    for point in test_data:
        distances = [euclidean_distance(point[:-1], centroid[0]) for centroid in clusters]
        nearest_centroid = distances.index(min(distances))
        predicted_class = cluster_classes[nearest_centroid]
        predictions.append(predicted_class)
    
    return predictions