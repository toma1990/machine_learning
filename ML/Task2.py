import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\\Users\\Tom\\Desktop\\ML\\Task2_dataset_dog_breeds.csv')
print(dataset)


# Calculate the euclidean distance of 2 vectors.
def compute_euclidean_distance(vec_1, vec_2):
    """
    vec_1: vector 1
    vec_2: vector 2
    return: the euclidean distance of Vector 1 and Vector 2
    """

    distance = np.sqrt(np.sum(np.square(vec_1 - vec_2)))
    return distance


# Randomly initialise centroids.
def initialise_centroids(dataset, k):
    """
    dataset: a given dataset
    k: the number of clusters
    return: a NumPy array of k points randomly picked from the given dataset as initial centroids
    """

    pointNum, dimension = dataset.shape
    centroids = np.zeros((k, dimension))
    pointIndexList = [index for index in range(pointNum)]
    pointIndexList_picked = random.sample(pointIndexList, k)

    for count in range(k):  # range: [0, k)
        centroids[count, :] = dataset[pointIndexList_picked[count], :]

    return centroids

# Cluster a given dataset into k groups.
def kmeans(dataset, k):
    """
    dataset: a given dataset with all four features
    k: the number of clusters
    return: a NumPy array of final centroids after clustering and
            a dictionary storing each point's cluster and each iteration step's objective function
    """

    pointNum = dataset.shape[0]
    centroids = initialise_centroids(dataset, k)  # randomly initialise centroids
    hasChangedCluster = True

    """
    The value paired with the key ClusterNo is a list storing each point's cluster;
    The value paired with the key ObjectiveFunction is a list storing each iteration step's objective function
    """

    cluster_assigned = {'ClusterNo': [0 for count in range(pointNum)], 'ObjectiveFunction': []}

    # loop for iteration until none of the cluster assignments change
    while hasChangedCluster:
        hasChangedCluster = False
        totalDistance = 0

        for i in range(pointNum):  # range: [0, pointNum)
            distanceList = []

            # loop to store the euclidean distance of a centroid and the point
            for j in range(k):  # range: [0, k)
                distance = compute_euclidean_distance(centroids[j, :], dataset[i, :])
                distanceList.append(distance)

            minDistance = min(distanceList)
            centroidIndex = distanceList.index(minDistance)  # select the centroid who is closest to the point
            totalDistance += minDistance

            if cluster_assigned['ClusterNo'][i] != centroidIndex:
                hasChangedCluster = True
                cluster_assigned['ClusterNo'][i] = centroidIndex  # update the point's cluster

        # loop to update each cluster's centroid
        for count in range(k):  # range: [0, k)
            """
            First, this cluster's points are marked with True (1) and the rest with False (0).
            Second, it will get the indexes of points marked with 1 which is also the indexes of the cluster's points 
            in the dataset
            """
            clusterPoints = dataset[np.nonzero(np.array(cluster_assigned['ClusterNo']) == count)[0], :]
            centroids[count, :] = np.average(clusterPoints, axis=0)
            # the new centroid is the average of the cluster's points

        cluster_assigned['ObjectiveFunction'].append(totalDistance)  # store this iteration step's objective function

    return centroids, cluster_assigned


dataset = np.array(dataset)  # Dataset with all four features
height_tail = dataset[:, [0, 1]]
height_leg = dataset[:, [0, 2]]

for k in [2, 3]:
    centroids, cluster_assigned = kmeans(dataset, k)  # cluster the dataset into k groups
    colours = ['Purple', 'Blue', 'Green']

    # plot the scatter plots "height - tail length" and "height - leg length"
    for data in [height_tail, height_leg]:
        plt.xlabel('Height')

        if data is height_tail:
            plt.title('Height - Tail Length (k = {})'.format(k))
            plt.ylabel('Tail Length')

            for count in range(k):  # range: [0, k)
                plt.scatter(centroids[count, 0], centroids[count, 1], s=100, c=colours[count], marker='X')
        else:
            plt.title('Height - Leg Length (k = {})'.format(k))
            plt.ylabel('Leg Length')

            for count in range(k):  # range: [0, k)
                plt.scatter(centroids[count, 0], centroids[count, 2], s=100, c=colours[count], marker='X')

        for count in range(data.shape[0]):  # range: [0, data.shape[0])
            colourIndex = cluster_assigned['ClusterNo'][count]  # set different colours to clusters
            plt.scatter(data[count, 0], data[count, 1], s=10, c=colours[colourIndex], alpha=0.5)

        plt.show()

    iterationStep = len(cluster_assigned['ObjectiveFunction'])

    # plot the line plot "iteration step - objective function"
    plt.title('k = {}'.format(k))
    plt.xlabel('Iteration Step')
    plt.ylabel('Objective Function Value')
    plt.xticks(range(1, iterationStep + 1))
    plt.plot([count for count in range(1, iterationStep + 1)], cluster_assigned['ObjectiveFunction'])
    plt.show()
