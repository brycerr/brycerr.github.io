"""
Bryce Rothschadl
Dr. Hien Nguyen
COMPSCI 767-01: Big Data and Data Mining
2025-04-17
"""

import csv
import math
import random


class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid
        self.items = []

    def calc_new_centroid(self):
        new_centroid = [0] * len(self.centroid)

        # add up all points respective to index
        for i in range(len(self.items)):
            for j in range(len(self.items[i].points)):
                new_centroid[j] += self.items[i].points[j]

        # divide by total number of points respective to index
        for i in range(len(new_centroid)):
            new_centroid[i] /= len(self.items)

        return new_centroid


class State:
    def __init__(self, name, points):
        self.name = name
        self.points = points
        self.cluster = -1
        self.distance_to_cluster = math.inf

    def __str__(self):
        return f"{self.name}: {self.points}"


def read_input(path):
    data = []
    with open(path, newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            name = row[0]
            points = row[1:5]   # cuts out the state name column
            for i in range(len(points)):
                points[i] = float(points[i])
            data.append(State(name, points))
    return data


def euclidean_distance(p, q):
    d = 0
    for i in range(len(p)):
        d += (q[i] - p[i])**2

    return math.sqrt(d)


def scale_data(data):
    num_features = len(data[0].points)
    num_data_points = len(data)
    means = [0.0] * num_features
    std_devs = [0.0] * num_features

    # means
    for i in range(num_features):
        total = 0.0
        for j in range(num_data_points):
            total += data[j].points[i]
        means[i] = total / num_data_points

    # standard deviations
    for i in range(num_features):
        total = 0.0
        for j in range(num_data_points):
            total += (data[j].points[i] - means[i]) ** 2
        std_devs[i] = math.sqrt(total / num_data_points)

    # scale each feature of every data point using the Z formula
    for j in range(num_data_points):
        for i in range(num_features):
            if std_devs[i] != 0:
                data[j].points[i] = (data[j].points[i] - means[i]) / std_devs[i]

    return data


def k_means(data, k_val, max_iterations):
    clusters = []

    # 1. Randomly initialize k centroids from data_points
    print(f"Initial centroids:")
    temp = []

    for i in range(k_val):
        rand_state = random.choice(data)
        print(rand_state)
        clusters.append(Cluster(rand_state.points))

        # ensure each state can only be selected once
        temp.append(rand_state)
        data.remove(rand_state)
    print()

    # put the removed centroids back in
    for i in range(len(temp)):
        data.append(temp[i])

    # 2. Repeat until convergence or max_iterations reached:
    iter_counter = 0
    is_converged = False

    while iter_counter < max_iterations and not is_converged:
        # clear previous cluster assignments
        for i in range(len(clusters)):
            clusters[i].items = []

        # a. Assign each data point to the nearest centroid
        #    - Use Euclidean distance
        for i in range(len(data)):
            min_dist = math.inf
            closest_cluster = -1
            for j in range(len(clusters)):
                dist = euclidean_distance(data[i].points, clusters[j].centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = j

            data[i].distance_to_cluster = min_dist
            data[i].cluster = closest_cluster

            clusters[closest_cluster].items.append(data[i])

        # b. Recalculate centroids as the mean of points in each cluster
        new_centroids = [0] * len(clusters)
        for i in range(len(clusters)):
            new_centroids[i] = clusters[i].calc_new_centroid()

        # c. Check for convergence (e.g., centroids do not change)
        check_convergence = True
        for i in range(len(clusters)):
            for j in range(len(clusters[i].centroid)):
                tol = 0.001
                diff = abs(new_centroids[i][j] - clusters[i].centroid[j])
                if diff > tol:
                    check_convergence = False
        is_converged = check_convergence

        for i in range(len(clusters)):
            clusters[i].centroid = new_centroids[i]

        iter_counter += 1

    # 3. Return final clusters and centroids
    print(f"Total number of iterations: {iter_counter}\n")
    return clusters


def main():
    path = "USArrests.csv"
    data = read_input(path)
    data = scale_data(data)

    print("K-Means:\n")
    k = 4
    max_iterations = 100
    clusters = k_means(data, k, max_iterations)
    for i in range(len(clusters)):
        print(f"Cluster[{i}]")
        print(f"Centroid: {clusters[i].centroid}")
        for j in range(len(clusters[i].items)):
            print(clusters[i].items[j])
        print()


if __name__ == "__main__":
    main()
