"""
Bryce Rothschadl
CS767: Big Data and Data Mining
Assignment 03: Matching
"""

import Levenshtein
import pandas as pd
import time


def main():
    l_table_path = "../assignment_01/tableA.csv"
    r_table_path = "../assignment_01/tableB.csv"

    l_table = pd.read_csv(l_table_path)
    r_table = pd.read_csv(r_table_path)

    matches = find_matches(l_table, r_table)
    create_table_c(matches)


def is_match(a, b, a_date, b_date, threshold=0.80):
    # use Levenshtein distance (edit distance) to check the similarity of strings a and b
    if Levenshtein.ratio(a, b) >= threshold:
        # check release year to more accurately match games
        if a_date == b_date:
            return True
    return False


def find_matches(l_table, r_table):
    # record elapsed time
    time_start = time.time()
    print(f"[{time.ctime()}] Finding matches...")

    # find matches between the two tables
    matches = []
    i = 0

    for index_a, row_a in l_table.iterrows():

        for index_b, row_b in r_table.iterrows():
            # check string similarity
            if is_match(row_a['Title'], row_b['Title'], row_a['Release_Date'], row_b['Release_Date']):
                i = i + 1
                temp = {
                    'ID':               i,
                    'ltable_id':        row_a['ID'],
                    'rtable_id':        row_b['ID'],
                    'ltable_title':     row_a['Title'],
                    'rtable_title':     row_b['Title'],
                    'rtable_platform':  row_b['Platform'],
                    'ltable_date':      row_a['Release_Date'],
                    'rtable_date':      row_b['Release_Date'],
                    'ltable_rating':    row_a['Metascore'],
                    'rtable_rating':    row_b['Score']
                }
                matches.append(temp)

        # progress update (for debugging purposes)
        if index_a % 100 == 0:
            print(f"index_a: {index_a}, Number of Matches:{len(matches)}")

    # record elapsed time
    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f"[{time.ctime()}] Found {len(matches)} matches in {time_elapsed} seconds.\n")

    return matches


def create_table_c(matches):
    # record elapsed time
    time_start = time.time()
    print(f"[{time.ctime()}] Creating Table C...")

    # write matches to Table C
    table_c_df = pd.DataFrame(matches)
    table_c_df.to_csv('tableC.csv', index=False)

    # record elapsed time
    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f"[{time.ctime()}] Wrote {len(matches)} matches to Table C in {time_elapsed} seconds.\n")


if __name__ == "__main__":
    main()

    # for debug purposes:
    # a = "Ratchet & Clank (PS4)"
    # b = "Ratchet & Clank"
    #
    # a = "Final Fantasy IX"
    # b = "Final Fantasy XVI"
    #
    # print(Levenshtein.ratio(a, b))
