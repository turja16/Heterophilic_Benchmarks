import argparse
import sys
import os
import random
import numpy as np
import torch
import gencat
from utils_gencat import config_diagonal, feature_extraction
from converter import convert_to_planetoid

datasets_to_convert = [
    "cornell",
    "texas",
    "wisconsin",
    'blogcatalog',
    'wiki',
    'flickr',
    'actor',
    'chameleon',
    'squirrel']


def main(args):
    from utils import load_data, save_graph_2, get_path_to_top_dir
    from models.dataset_utils import DataLoader
    _ = DataLoader(args.dataset, data_dir="./data/")  # download dataset if not exist

    if args.dataset in datasets_to_convert:
        convert_to_planetoid(args.dataset)
    adj, features, labels = load_data(args.dataset)
    print(type(adj))  # <class 'scipy.sparse.csr.csr_matrix'>
    print(adj.shape)  # (n_data, n_data)
    print(type(features))  # <class 'numpy.ndarray'>
    print(features.shape)  # (n_data, n_feature)
    print(type(labels))  # <class 'list'>
    print(len(labels))  # n_data

    num_classes = len(set(labels))
    print("class:", num_classes)

    top_dir = get_path_to_top_dir()
    n_iter = args.n_iter

    M, D, class_size, H, node_degree = feature_extraction(adj, features, labels)

    # if args.exp == "hetero_homo":
    k = num_classes
    M_diag = 0  # initialization
    for i in range(k):
        M_diag += M[i, i]
    M_diag /= k  # calculate average of intra-class connections
    # Integer representation of the percentage of the original data to be adjusted to 10 steps
    base_value = int(M_diag * 10)

    # Generate parameters according to the percentage of in-class connections
    # in the original data
    params = list(range(base_value - 9, base_value + 1))
    print(params)

    data_dir = f"../GenCAT_Exp_hetero_homo"
    os.makedirs(data_dir, exist_ok=True)

    for x_ in params:
        for i in range(n_iter):
            M_config, D_config = config_diagonal(M, D, x_)
            adj, features, labels = gencat.gencat(
                M_config, D_config, H, class_size=class_size, theta=node_degree)
            print(type(adj))  # <class 'scipy.sparse.dok.dok_matrix'>
            print(adj.shape)  # (n_data, n_data)
            graphdata = {
                'adj': adj,
                'feature': features,
                'labels': labels
            }
            torch.save(graphdata, "{}/GenCAT_{}_{}_{}.pt".format(
                data_dir, args.dataset, x_, i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--n_iter', type=int, default=10)
    args = parser.parse_args()
    main(args)
