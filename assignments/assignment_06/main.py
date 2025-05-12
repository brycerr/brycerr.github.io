"""
Bryce Rothschadl
Dr. Hien Nguyen
COMPSCI 767-01: Big Data and Data Mining
2025-05-14
"""

import csv

import matplotlib.pyplot as plt
import numpy as np  # only used for computing eigenvalues and vectors


def read_data(path):
    data = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter=' ')
        next(reader)
        for row in reader:
            temp = []
            for col in row:
                temp.append(int(col))
            data.append(temp)
    return data


def center_data(data):
    # compute the means of each column
    n = len(data[0]) - 1    # should be 960 (30 * 32 image size), but I don't want to hardcode this number
    col_means = [0 for _ in range(n)]
    for row in data:
        for i in range(n):
            col_means[i] += row[i]

    for i in range(len(col_means)):
        col_means[i] /= len(data)
    # print(col_means)

    # center the data for use in PCA
    centered_data = []
    for row in data:
        centered_row = []
        for i in range(n):
            centered_row.append(row[i] - col_means[i])
        centered_data.append(centered_row)
    # print(centered_data)

    return centered_data


def my_pca(data):
    # (can use the built-in functions to compute eigenvalues and eigen vectors)

    # create covariance matrix
    matrix_x = np.array(data)
    cov_matrix = np.dot(matrix_x.T, matrix_x) / (matrix_x.shape[0] - 1)

    # eigen decomposition
    w, v = np.linalg.eigh(cov_matrix)       # w: eigenvalues, v: eigenvectors
    # print(w.tolist())
    # print(v.tolist())

    # sort the two lists
    sorted_indices = np.argsort(w)[::-1]    # sort the indices of w in reverse order
    latent = []     # sorted eigen values in descending order
    coeff = []      # corresponding sorted eigen vector
    for i in sorted_indices:
        latent.append(w[i])
        coeff.append(v[:, i].tolist())
    # print(latent)
    # print(coeff)

    return coeff, latent


def eigen_faces(data, num_faces, x, y):
    coeff, latent = my_pca(data)

    # convert the lists of eigen faces into 30 * 32 images
    for i in range(num_faces):
        eigen_face = coeff[i]
        eigen_face_image = []
        for j in range(x):
            temp = eigen_face[j*y:(j+1)*y]
            eigen_face_image.append(temp)

        # display image
        plt.imshow(eigen_face_image, cmap='gray')
        plt.title(f"Eigenface {i + 1}")
        plt.show()


def proportion_of_variance(data, var=0.90):
    # c2
    coeff, latent = my_pca(data)
    w_total = 0     # sum of all eigenvalues
    for i in range(len(latent)):
        w_total += latent[i]
    # print(f"w_total: {w_total}")

    pov = [0 for _ in range(len(latent))]
    for i in range(len(latent)):
        pov[i] = latent[i] / w_total

    # calculate cumulative sum of elements
    cum_sum_pov = []
    current = 0
    for i in range(len(pov)):
        current += pov[i]
        cum_sum_pov.append(current)
    # print(f"Cumulative Sum: {cum_sum_pov}")

    # c3
    # find index k of the first principal component for which the cumulative sum of elements exceeds 90
    k = -1
    for i in range(len(cum_sum_pov)):
        if cum_sum_pov[i] >= var:
            k = i
            break
    # print(f"k: {k}")
    if k == -1:
        print(f"Something went very wrong. No k is greater than {var}.")

    # c4
    # plot the cumulative function
    plt.title("Cumulative Function")
    plt.plot([x for x in range(len(cum_sum_pov))], cum_sum_pov)
    plt.axhline(y=var, color='red', linestyle='--', label=f"Variance threshold")
    plt.plot(k, cum_sum_pov[k], marker='o', color='red', linestyle='', label=f"k = {k} Principal Components")
    plt.grid(True)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Proportion of Variance")
    plt.legend()
    plt.show()

    return k


def main():
    num_faces = 0
    face_x = 30
    face_y = 32

    train_path = "face_train_data_960.txt"
    test_path = "face_test_data_960.txt"

    raw_train_data = read_data(train_path)
    # print(raw_train_data)
    train_data = center_data(raw_train_data)

    eigen_faces(train_data, num_faces, face_x, face_y)

    # c
    k = proportion_of_variance(train_data)
    print(f"k: {k}")

    # TODO: d
    # Project the training data and test data to the K principle components that we find in section c for K={1,3,5,7}.

    # TODO: e
    # Use the first k components (k= {50, 100}) to approximate the first five images in the training set


if __name__ == "__main__":
    main()
