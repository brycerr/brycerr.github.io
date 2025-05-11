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
    # TODO: implement pca
    # (can use the built-in functions to compute eigenvalues and eigen vectors)

    # create covariance matrix
    matrix_x = np.array(data)
    cov_matrix = np.dot(matrix_x.T, matrix_x) / (matrix_x.shape[0] - 1)

    # eigen decomposition
    w, v = np.linalg.eigh(cov_matrix)    # w: eigenvalues, v: eigenvectors
    # print(w.tolist())
    # print(v.tolist())

    # sort the two lists
    sorted_indices = np.argsort(w)[::-1]    # sorts the indices of w in reverse order

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

    # TODO: display the first given eigen faces using pyplot
    #   - customizable in main function
    #   - example: plt.imshow(eigen_face_image, cmap='gray')

    # convert the lists of eigen faces into 30 * 32 images
    for i in range(num_faces):
        eigen_face = coeff[i]
        eigen_face_image = []
        for j in range(x):
            temp = eigen_face[j*y:(j+1)*32]
            eigen_face_image.append(temp)

        # display image
        plt.imshow(eigen_face_image, cmap='gray')
        plt.show()


def proportion_of_variance(data):
    # TODO: c1
    # TODO: c2
    # TODO: c3
    # TODO: c4
    return  # number of eigen vectors that can explain 90% of cariance


# TODO:
#   c
#   d
#   e


def main():
    num_faces = 10
    face_x = 30
    face_y = 32

    train_path = "face_train_data_960.txt"
    test_path = "face_test_data_960.txt"

    raw_train_data = read_data(train_path)
    # print(raw_train_data)
    train_data = center_data(raw_train_data)

    eigen_faces(train_data, num_faces, face_x, face_y)


if __name__ == "__main__":
    main()
