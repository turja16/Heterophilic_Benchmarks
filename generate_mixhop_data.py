import argparse
from pathlib import Path
from typing import NamedTuple, Union

import networkx as nx
import numpy as np
import torch

# python generate_mixhop_data.py --h 0.9 --plot

BASE_DIR = "./mixhop_syn-2000_5/"
# generate syn data dir if not exist
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)


# get a random class
def get_color(class_ratio):
    return np.random.choice(range(1, len(class_ratio) + 1), 1, False, class_ratio)[0]


def color_weight(col1, col2, exponent, class_ratio, opposite_side_class_weight):
    dist = abs(col1 - col2)
    dist = min(dist, len(class_ratio) - dist)
    if dist == (len(class_ratio) / 2):
        return opposite_side_class_weight
    else:
        return (
                       exponent ** ((len(class_ratio) / 2) - dist)
               ) * opposite_side_class_weight


def get_neighbours(G, m, col, h, exponent, class_ratio, opposite_side_class_weight):
    pr = dict()
    for v in G.nodes():
        if G.nodes[v]["color"] == col:
            pr[v] = float(G.degree(v)) * (h)
        else:
            pr[v] = float(G.degree(v)) * (
                    (1 - h) * color_weight(col,
                                           G.nodes[v]["color"],
                                           exponent,
                                           class_ratio,
                                           opposite_side_class_weight)
            )
    norm_pr = float(sum(pr.values()))
    for v in pr.keys():
        pr[v] = float(pr[v]) / norm_pr
    us = np.random.choice(list(pr.keys()), m, False, list(pr.values()))
    return us


# generate_graph(n, 6, 40, h, class_ratio)
def generate_graph(n, m, m0, h, class_ratio, exponent, opposite_side_class_weight):
    if m > n:
        return
    G = nx.Graph()
    # initialize 40 nodes
    for v in range(m0):
        G.add_node(v, color=get_color(class_ratio))
        if v > 1:
            G.add_edge(v, v - 1)
    for v in range(m0, n):
        col = get_color(class_ratio)
        us = get_neighbours(G, m, col, h, exponent, class_ratio, opposite_side_class_weight)
        G.add_node(v, color=col)
        for u in us:
            G.add_edge(v, u)
    return G


# generate splits
def random_disassortative_splits(labels, num_classes):
    """
    0.6 labels for training
    0.2 labels for validation
    0.2 labels for testing
    """
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero()[0]
        index = index[np.random.permutation(index.size)]
        indices.append(index)

    train_index = []
    val_index = []
    test_index = []

    for i in indices:
        train_index.extend(i[: int(len(i) * 0.6)])
        val_index.extend(i[int(len(i) * 0.6): int(len(i) * 0.8)])
        test_index.extend(i[int(len(i) * 0.8):])
    return np.array(train_index), np.array(val_index), np.array(test_index)


# generate feature
def make_x(num_classes, label, n):
    from random import shuffle
    # sample node feature from overlapping multi-gaussian distributions
    # Copyright 2019 Sami Abu-El-Haija. All Rights Reserved.
    # Original code & data: https://github.com/samihaija/mixhop/blob/master/data/synthetic
    variance_factor = 350
    start_cov = np.array([[70.0 * variance_factor, 0.0], [0.0, 20.0 * variance_factor]])
    cov = start_cov
    theta = np.pi * 2 / num_classes
    rotation_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    radius = 300
    allx = np.zeros(shape=[n, 2], dtype="float32")
    for cls, theta in enumerate(np.arange(0, np.pi * 2, np.pi * 2 / num_classes)):
        gaussian_y = radius * np.cos(theta)
        gaussian_x = radius * np.sin(theta)
        num_points = np.sum(label == cls)
        coord_x, coord_y = np.random.multivariate_normal(
            [gaussian_x, gaussian_y], cov, num_points
        ).T
        cov = rotation_mat.T.dot(cov.dot(rotation_mat))
        # Belonging to class cls
        example_indices = np.nonzero(label == cls)[0]
        shuffle(example_indices)
        allx[example_indices, 0] = coord_x
        allx[example_indices, 1] = coord_y
    return allx


def generate_syn_graph(args: Union[NamedTuple, argparse.Namespace]):
    c = args.c  # number of classes
    n = args.n  # total number of nodes
    h = args.h  # homophily
    g_num = args.num_graph
    e = 6  # sample 6 edges for each newly added nodes
    ini = 40  # initialize 40 nodes
    class_ratio = [float(1.0 / c)] * c  # len must be always even number
    # Solving equation of weights of connecting colors
    exponent = 2
    opposite_side_class_weight = 1
    for ind in range(int(len(class_ratio) / 2) - 1):
        opposite_side_class_weight += 2 * (exponent ** (ind + 1))

    opposite_side_class_weight = 1.0 / opposite_side_class_weight

    for g_idx in range(g_num):
        print(f"Generating graph{g_idx}...")
        name = "n{}-h{}-c{}-g{}".format(n, h, len(class_ratio), g_idx)

        G = generate_graph(n, e, ini, h, class_ratio, exponent, opposite_side_class_weight)
        G_d = nx.to_dict_of_lists(G)
        ally = np.zeros((len(G.nodes()), len(class_ratio)))
        for v in G.nodes():
            ally[v][G.nodes[v]["color"] - 1] = 1

        allx = make_x(num_classes=c, label=ally.argmax(1), n=n)

        torch.save(G_d, "{}/ind.{}.graph".format(BASE_DIR, name))  # mixhop style graph
        torch.save(ally, "{}/ind.{}.ally".format(BASE_DIR, name))  # one-hop label
        torch.save(allx, "{}/ind.{}.allx".format(BASE_DIR, name))  # feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic graph dataset following Mixhop"
    )
    parser.add_argument("--h", type=float, default=0.5, help="homophily level")
    parser.add_argument("--n", type=int, default=2000, help="total number of nodes")
    parser.add_argument("--c", type=int, default=5, help="total number of classes")
    parser.add_argument("--plot", action="store_true", help="plot graph and degree dist")
    parser.add_argument(
        "--num_graph",
        type=int,
        default=10,
        help="number of graphs generated for each setting",
    )
    args = parser.parse_args()

    generate_syn_graph(args)
