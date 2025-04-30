"""
Bryce Rothschadl
Dr. Hien Nguyen
COMPSCI 767-01: Big Data and Data Mining
2025-04-30
"""

import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


def create_graph(path):
    g = nx.DiGraph()

    with open(path, newline='') as file:
        reader = csv.reader(file)
        # i = 0
        for row in reader:
            # i += 1
            parent = row[0].strip('Node')
            g.add_node(parent)
            for col in row[1:]:
                child = col.strip(' Node')
                g.add_edge(parent, child)
                # print(f"[{i}] {child}")
            # print()
    return g


def calc_page_rank(g):
    pr_dict = nx.pagerank(g)
    pr_list = []
    i = 0
    for val in pr_dict.values():
        i += 1
        print(f"Page [{i}]: {val}")
        pr_list.append(val * 3000)
    return pr_list


def main():
    path = "graph.csv"
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    g = create_graph(path)
    pr_list = calc_page_rank(g)

    plt.title("Figure 1")
    nx.draw(g, node_color="red", node_size=pr_list, with_labels=True, width=1)
    plt.show()


if __name__ == "__main__":
    main()
