import sys
import random
import numpy as np
from numpy.linalg import norm

def distance(row1, row2, method="Euclidean"):
    if method == "Euclidean":
        return norm(row1 - row2)
    elif method == "Cosine":
        return (1 - ((np.dot(row1, row2) / (norm(row1) * norm(row2)))))
    elif method == "Jarcard":
        return (1 - ((np.sum(np.minimum(row1, row2)) / np.sum(np.maximum(row1, row2)))))

def meanInstance(rowList):
    numRows = len(rowList)
    if (numRows == 0):
        return
    means = np.mean(rowList, axis=0)
    return means

def assignAll(data, centroids, method="Euclidean"):
    clusters = []
    for i in range(len(centroids)):
        clusters.append([])

    for row in data:
        clusterIndex = assign(row, centroids, method)
        clusters[clusterIndex].append(row)

    for i in range(len(clusters)):
        clusters[i] = np.array(clusters[i])

    return clusters

def assign (row, centroids, method="Euclidean"):
    minDistance = distance(row, centroids[0], method)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(row, centroids[i], method)
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def computeCentroids(clusters):
    centroids = np.empty((len(clusters), len(clusters[0][0])))
    for i in range(len(clusters)):
        centroid = meanInstance(clusters[i])
        centroids[i] = centroid
    return centroids

def computeSSE(clusters, centroids, method="Euclidean", returnSSEs=False):
    results = []
    for i in range(len(centroids)):
        centroid = np.copy(centroids[i])
        cluster = np.copy(clusters[i])
        result = 0
        for instance in cluster:
            result += distance(centroid, instance) ** 2
        results.append(result)

    if returnSSEs:
        return results
    else:
        return sum(results)

def getAccuracy(data, labels, centroids, method="Euclidean"):
    assigned = np.apply_along_axis(assign, axis=1, arr=data, centroids=centroids, method=method)
    centroid_label_count = np.zeros((len(centroids), 10))
    for i in range(len(labels)):
        centroid_label_count[int(assigned[i])][int(labels[i])] += 1
    centroid_labels = np.argmax(centroid_label_count, axis=1)
    correct = 0
    total = 0
    for i in range(len(assigned)):
        if centroid_labels[int(assigned[i])] == int(labels[i]):
            correct += 1
        total += 1
    return correct / total

def fixEmpty(clusters, centroids):
    newCentroids = np.empty((len(clusters), len(clusters[0][0])))
    didFix = False
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        if len(cluster) == 0:
            print("Empty Cluster found")
            didFix = True
            SSEs = computeSSE(clusters, centroids, returnSSEs=True)
            worstSSE = SSEs.index(max(SSEs))
            newCentroids[i] = random.choice(clusters[worstSSE])
        else:
            newCentroids[i] = centroids[i]

    if didFix:
        return newCentroids
    else:
        return False



def kmeans(rawData, k, method="Euclidean", condition="noChange", startingCentroids=None):
    result = {}
    data = np.copy(rawData)

    # randomly select k initial centroids, if no inital ones are specified
    if startingCentroids is not None:
        centroids = startingCentroids
    else:
        centroids = np.copy(data[np.random.choice(data.shape[0], k, replace=False)])

    prev_centroids = np.ones_like(centroids)
    iteration = 0
    prev_SSE = sys.maxsize
    SSE = sys.maxsize - 1

    if condition == "noChange":
        while not np.array_equal(centroids, prev_centroids):
            iteration += 1
            prev_centroids = np.copy(centroids)
            clusters = assignAll(data, centroids, method)
            centroids = fixEmpty(clusters, centroids)
            if centroids:
                clusters = assignAll(data, centroids, method)
            centroids = computeCentroids(clusters)

    elif condition == "SSE":
        while SSE < prev_SSE:
            iteration += 1
            clusters = assignAll(data, centroids, method)
            centroids = fixEmpty(clusters, centroids)
            if centroids:
                clusters = assignAll(data, centroids, method)
            centroids = computeCentroids(clusters)
            prev_SSE = SSE
            SSE = computeSSE(clusters, centroids, method)

    elif condition == "Iterations":
        while iteration < 50:
            iteration += 1
            clusters = assignAll(data, centroids, method)
            centroids = fixEmpty(clusters, centroids)
            if centroids:
                clusters = assignAll(data, centroids, method)
            centroids = computeCentroids(clusters)

    elif condition == "All":
        while (not np.array_equal(centroids, prev_centroids)) and SSE < prev_SSE and iteration < 100:
            iteration += 1
            prev_centroids = np.copy(centroids)
            clusters = assignAll(data, centroids, method)
            centroids = fixEmpty(clusters, centroids)
            if centroids:
                clusters = assignAll(data, centroids, method)
            centroids = computeCentroids(clusters)
            prev_SSE = SSE
            SSE = computeSSE(clusters, centroids, method)

    SSE = computeSSE(clusters, centroids, method)

    result["clusters"] = clusters
    result["centroids"] = centroids
    result["SSE"] = SSE
    result["numIterations"] = iteration
    return result

# TESTING
raw_data = np.loadtxt("kmeans_data/data.csv", delimiter=',')
data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))  # normalize data
labels = np.loadtxt("kmeans_data/label.csv", delimiter=',')
methods = ["Euclidean", "Cosine", "Jarcard"]
conditions = ["noChange", "SSE", "Iterations"]
k = 10
startCentroids = np.copy(data[np.random.choice(data.shape[0], k, replace=False)])
#Q1 / #Q2 / Q3 (change condition to "All" for 3

for method in methods:
    results = kmeans(data, k, method=method, condition="All", startingCentroids=startCentroids)
    print(method + ": ")
    print(f"    SSE: {results['SSE']:,.2f}")
    print(f"    NumIterations: {results['numIterations']}")
    acc = getAccuracy(data, labels, results["centroids"], method=method)
    print(f"    Accuracy: {acc * 100:,.2f}")

#Q4
for method in methods:
    for condition in conditions:
        results = kmeans(data, k, method=method, condition=condition, startingCentroids=startCentroids)
        print(f"{method} + {condition}")
        print(f"    SSE: {results['SSE']:,.2f}")
        print(f"    NumIterations: {results['numIterations']}")
        acc = getAccuracy(data, labels, results["centroids"], method=method)
        print(f"    Accuracy: {acc * 100:,.2f}")


