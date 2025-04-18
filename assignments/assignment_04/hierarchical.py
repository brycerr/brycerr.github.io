"""
Bryce Rothschadl
Dr. Hien Nguyen
COMPSCI 767-01: Big Data and Data Mining
2025-04-17
"""

import csv
import math


class Cluster:
    def __init__(self):
        self.states = []


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


def create_distance_matrix(clusters):
    n = len(clusters)
    dm = [[None] * n] * n
    # print(dm)

    for i in range(n):
        for j in range(i + 1, n):
            for i_state in clusters[i].states:
                for j_state in clusters[j].states:
                    # print(f"i: {i_state.name}, j: {j_state.name}")
                    dist = euclidean_distance(i_state.points, j_state.points)
                    dm[i][j] = dist
    # print(dm)
    return dm


def min_pairwise_distance(cluster_a, cluster_b):
    min_dist = math.inf
    for a_state in cluster_a.states:
        for b_state in cluster_b.states:
            dist = euclidean_distance(a_state.points, b_state.points)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def max_pairwise_distance(cluster_a, cluster_b):
    max_dist = -math.inf
    for a_state in cluster_a.states:
        for b_state in cluster_b.states:
            dist = euclidean_distance(a_state.points, b_state.points)
            if dist > max_dist:
                max_dist = dist
    return max_dist


def avg_pairwise_distance(cluster_a, cluster_b):
    total = 0
    count = 0
    for a_state in cluster_a.states:
        for b_state in cluster_b.states:
            dist = euclidean_distance(a_state.points, b_state.points)
            total += dist
            count += 1
    if count != 0:
        avg_dist = total / count
    else:
        avg_dist = -1
    return avg_dist


def hierarchical(data, linkage):
    num_clusters = 4    # desired clusters at the end
    clusters = []

    # 1. Start with each data point as its own cluster
    for i in range(len(data)):
        new_cluster = Cluster()
        new_cluster.states.append(data[i])
        clusters.append(new_cluster)

    # 2. Compute pairwise distances between all clusters
    dm = create_distance_matrix(clusters)

    # 3. Repeat until number of clusters == 4:
    while len(clusters) > num_clusters:
        # a. Find the two closest clusters based on linkage method:
        #    - Single: minimum pairwise distance
        #    - Complete: maximum pairwise distance
        #    - Average: mean pairwise distance

        min_dist = float(math.inf)
        pair_to_merge = (None, None)

        for i in range(len(dm)):
            for j in range(i + 1, len(dm)):
                if linkage == "single":
                    dist = min_pairwise_distance(clusters[i], clusters[j])
                elif linkage == "complete":
                    dist = max_pairwise_distance(clusters[i], clusters[j])
                elif linkage == "average":
                    dist = avg_pairwise_distance(clusters[i], clusters[j])

                if dist < min_dist:
                    min_dist = dist
                    pair_to_merge = (i, j)

        # b. Merge the two closest clusters
        i, j = pair_to_merge

        clusters[i].states.extend(clusters[j].states)

        # remove old cluster
        clusters.pop(j)

        # c. Update the distance matrix
        dm = create_distance_matrix(clusters)

    # 4. Return the 4 clusters
    return clusters


def main():
    path = "USArrests.csv"
    data = read_input(path)
    data = scale_data(data)

    print("Hierarchical Clustering:\n")
    print("================================================================")
    print("Single Linkage:\n")
    linkage = "single"
    clusters = hierarchical(data, linkage)
    for i in range(len(clusters)):
        print(f"Cluster[{i}]")
        for state in clusters[i].states:
            print(state)
        print()

    print("================================================================")
    print("Complete Linkage:\n")
    linkage = "complete"
    clusters = hierarchical(data, linkage)
    for i in range(len(clusters)):
        print(f"Cluster[{i}]")
        for state in clusters[i].states:
            print(state)
        print()

    print("================================================================")
    print("Average Linkage:\n")
    linkage = "average"
    clusters = hierarchical(data, linkage)
    for i in range(len(clusters)):
        print(f"Cluster[{i}]")
        for state in clusters[i].states:
            print(state)
        print()


if __name__ == "__main__":
    main()
