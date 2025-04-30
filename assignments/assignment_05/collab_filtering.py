"""
Bryce Rothschadl
Dr. Hien Nguyen
COMPSCI 767-01: Big Data and Data Mining
2025-04-30
"""

from collections import defaultdict
import csv
import math


def read_data(path):
    data = []
    with open(path, newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            temp = []
            for col in row:
                temp.append(float(col))
            data.append(temp)
    return data


def create_user_rating_matrix(train_data):
    user_ratings = defaultdict(dict)
    for user, movie, rating in train_data:
        user_ratings[user][movie] = rating
    return user_ratings


def calc_similarity(a_ratings, u_ratings):
    # a_ratings: ratings for active user a
    # u_ratings: ratings for another user u
    shared_movies = set(a_ratings.keys()).intersection(u_ratings.keys())
    n = len(shared_movies)
    if n == 0:
        return 0

    u_sum = 0
    a_sum = 0

    a_sum_sq = 0
    u_sum_sq = 0

    product_sum = 0

    for movie in shared_movies:
        a_sum += a_ratings[movie]
        u_sum += u_ratings[movie]

        a_sum_sq += a_ratings[movie] ** 2
        u_sum_sq += u_ratings[movie] ** 2

        product_sum += a_ratings[movie] * u_ratings[movie]

    numerator = product_sum - (a_sum * u_sum / n)
    denominator = math.sqrt((a_sum_sq - a_sum ** 2 / n) * (u_sum_sq - u_sum ** 2 / n))

    if denominator == 0:
        return 0

    return numerator / denominator


def predict(user_id, movie_id, user_ratings):
    target_ratings = user_ratings.get(user_id, {})

    if movie_id in target_ratings:
        return target_ratings[movie_id]

    numerator = 0
    denominator = 0

    for other_user, ratings in user_ratings.items():
        if other_user == user_id or movie_id not in ratings:
            continue

        sim = calc_similarity(user_ratings[user_id], ratings)
        if sim <= 0:
            continue

        numerator += sim * ratings[movie_id]
        denominator += sim

        return numerator / denominator


def write_predictions(path, predictions):
    with open(path, newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    header = rows[0] + ['predicted_rating']
    new_rows = [header]

    for row, prediction in zip(rows[1:], predictions):
        new_rows.append(row + [prediction])

    new_path = f"{path.strip('.csv')}_with_predicted_ratings.csv"
    with open(new_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)


def main():
    print("Running...")

    train_path = "ratings_small_training.csv"
    test_path = "ratings_small_test.csv"

    train_data = read_data(train_path)
    # print(train_data)
    test_data = read_data(test_path)
    # print(test_data)

    user_rating_matrix = create_user_rating_matrix(train_data)

    predictions = []
    for user_id, movie_id in test_data:
        prediction = predict(user_id, movie_id, user_rating_matrix)
        predictions.append(prediction)

    write_predictions(test_path, predictions)

    print("Done.")


if __name__ == "__main__":
    main()
