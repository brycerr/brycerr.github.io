import csv


def calc_missing(path):
    missing = [0, 0, 0, 0, 0]   # id, title, platform, release_date, score
    total = 0
    with open(path, 'r', newline='', encoding="utf-8") as file:
        next(file)  # skip header line
        reader = csv.reader(file)
        for row in reader:
            total += 1
            for i in range(len(row)):
                if row[i] == "N/A":
                    missing[i] += 1
    return missing, total


def report_missing(missing, total):
    title_missing = "{:5.1f}".format(missing[1] / total * 100)
    platform_missing = "{:5.1f}".format(missing[2] / total * 100)
    release_date_missing = "{:5.1f}".format(missing[3] / total * 100)

    print("Missing Values")
    print(f"Title:          {title_missing}%")
    print(f"Platform:       {platform_missing}%")
    print(f"Release Date:   {release_date_missing}%")


def main():
    path = "../assignment_01/tableA.csv"
    # path = "../assignment_01/tableB.csv"

    results = calc_missing(path)
    report_missing(results[0], results[1])


if __name__ == '__main__':
    main()
