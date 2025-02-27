import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns


def calc_missing(df):
    missing = df.isnull().sum()
    total = len(df)
    return missing, total


def report_missing(missing, total):
    print("Missing Values Report")
    for column in missing.index:
        # fraction = f"{missing[column] : 4.0f}/{total}"
        missing_num = str.zfill(str(missing[column]), 4)
        # percentage = missing[1] / total * 100
        percentage = "{:5.1f}".format(missing[column] / total * 100)
        print(f"{column : <12} : {missing_num}/{total} : ( {percentage}% )")


def classify_attribute(df):
    classifications = {}
    for column in df.columns:
        if column == "Platform":
            classifications[column] = 'textual'
        elif pd.api.types.is_numeric_dtype(df[column]):
            classifications[column] = 'numeric'
        elif pd.api.types.is_bool_dtype(df[column]):
            classifications[column] = 'boolean'
        else:
            classifications[column] = 'textual'
    return classifications


def textual_report(df):
    print("Textual Report")
    for column in df.select_dtypes(include=['object']):
        lengths = df[column].str.len()
        avg_length = lengths.mean()
        min_length = lengths.min()
        max_length = lengths.max()
        print(column)
        print(f" Average length  : {avg_length:.2f}")
        print(f" Min length      : {min_length:.2f}")
        print(f" Max length      : {max_length:.2f}")
        print()


def create_histograms(df):
    for column in df.select_dtypes(include=['object']):
        lengths = df[column].dropna().str.len()

        # IQR (detecting outliers)
        q1 = lengths.quantile(0.25)
        q3 = lengths.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = lengths[(lengths < lower_bound) | (lengths > upper_bound)]

        plt.figure(figsize=(10, 6))
        sns.histplot(lengths, kde=True, bins=30)
        plt.title(f"Histogram of Character Lengths for {column} in Table B")
        plt.xlabel("Character Length")
        plt.ylabel("Frequency")

        if not outliers.empty:
            for outlier in outliers:
                plt.axvline(outlier, color='r', linestyle='--')

            custom_lines = [Line2D([0], [0], color='r', linestyle='--', lw=2)]
            plt.legend(custom_lines, ['Outliers'], loc='upper right')

        plt.show()

        if not outliers.empty:
            print(f"Outliers detected in {column} (character lengths):\n{sorted(outliers.to_list(), key=int)}\n")
        else:
            print(f"No outliers detected in {column} (character lengths).\n")


def main():
    # path = "../assignment_01/tableA.csv"
    path = "../assignment_01/tableB.csv"

    df = pd.read_csv(path)

    # Missing Values Report
    missing, total = calc_missing(df)
    report_missing(missing, total)

    print()

    # Classification Report
    classifications = classify_attribute(df)
    print("Classification Report")
    for column, d_type in classifications.items():
        print(f"{column : <12} : {d_type}")

    print()

    # Textual Analysis Report
    textual_report(df)

    print()

    # Histograms
    create_histograms(df)


if __name__ == '__main__':
    main()
