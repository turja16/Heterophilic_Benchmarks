import argparse
import random
from os import path
from typing import NamedTuple, Union

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops

from FAGCN.FAGCN_models import FAGCN
from FAGCN.FAGCN_training import train

BASE_DIR = f"{path.dirname(path.abspath(__file__))}/../PathNet/other_data"


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean()


def normalize_tensor_sparse(mx, symmetric=0):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 1e-12
    if symmetric == 0:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        mx = r_mat_inv.dot(mx)
        return mx
    else:
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def get_order(ratio: list, masked_index: torch.Tensor, total_node_num: int, seed: int = 1234567):
    random.seed(seed)
    masked_node_num = len(masked_index)
    shuffle_criterion = list(range(masked_node_num))
    random.shuffle(shuffle_criterion)
    #
    train_val_test_list = ratio
    tvt_sum = sum(train_val_test_list)
    tvt_ratio_list = [i / tvt_sum for i in train_val_test_list]
    train_end_index = int(tvt_ratio_list[0] * masked_node_num)
    val_end_index = train_end_index + int(tvt_ratio_list[1] * masked_node_num)
    #
    train_mask_index = shuffle_criterion[:train_end_index]
    val_mask_index = shuffle_criterion[train_end_index:val_end_index]
    test_mask_index = shuffle_criterion[val_end_index:]
    #
    train_index = masked_index[train_mask_index]
    val_index = masked_index[val_mask_index]
    test_index = masked_index[test_mask_index]
    # assert that there are no duplicates in sets
    assert len(set(train_index)) == len(train_index)
    assert len(set(val_index)) == len(val_index)
    assert len(set(test_index)) == len(test_index)
    # assert sets are mutually exclusive
    assert len(set(train_index) - set(val_index)) == len(set(train_index))
    assert len(set(train_index) - set(test_index)) == len(set(train_index))
    assert len(set(val_index) - set(test_index)) == len(set(val_index))
    return (train_index, val_index, test_index)


def random_splits_with_unlabel(labels, ratio: list = [60, 20, 20], seed: int = 1234567):
    labels = labels.cpu()
    y_have_label_mask = labels != -1
    total_node_num = len(labels)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    masked_index = y_index_tensor[y_have_label_mask]
    (train_index, val_index, test_index) = get_order(
        ratio, masked_index, total_node_num, seed)
    return (train_index, val_index, test_index)


def train_pathnetdata_fagcn(device: torch.device,
                            args: Union[NamedTuple, argparse.Namespace]):
    # load pathnet data
    x = np.load(BASE_DIR + '/' + args.dataset + '/x.npy')
    y = np.load(BASE_DIR + '/' + args.dataset + '/y.npy')
    numpy_edge_index = np.load(BASE_DIR + '/' + args.dataset + '/edge_index.npy')
    edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)
    edge_index = remove_self_loops(edge_index)[0]
    del numpy_edge_index
    # get stat
    num_nodes = x.shape[0]
    # adj = to_scipy_sparse_matrix(edge_index)

    lbl_set = []
    for lbl in y:
        if lbl not in lbl_set:
            lbl_set.append(lbl)

    if -1 in lbl_set:
        print('have unlabeled data; will be excluded in train/valid/test set')
        num_classes = len(lbl_set) - 1
    else:
        num_classes = len(lbl_set)

    labels_th = torch.LongTensor(y).to(device)
    feat_data_sparse = sp.coo_matrix(x)
    del y, x
    # normalize
    feat_data_sparse = normalize_tensor_sparse(feat_data_sparse, symmetric=0)
    feat_data_th = torch.tensor(feat_data_sparse.toarray(), dtype=torch.float32).to(device)
    del feat_data_sparse
    # adj.setdiag(1)
    # adj = normalize_tensor_sparse(adj, symmetric=1) # csc_matrix
    # adj = sp.csr_matrix(adj)

    n, c, d = feat_data_th.shape[0], num_classes, feat_data_th.shape[1]
    labels = labels_th
    num_features = d

    edge_index = edge_index.T.numpy()
    # format data for fagcn
    valid_ids = (np.arange(len(labels)), np.unique(edge_index[:, 1]))[args.remove_zero_in_degree_nodes]
    select = np.isin(edge_index[:, 0], valid_ids)
    U = edge_index[:, 0][select]
    V = edge_index[:, 1][select]
    # U = [e[0] for e in edge_index if e[0] in valid_ids]
    # V = [e[1] for e in edge_index if e[0] in valid_ids]
    g = dgl.graph((U, V))
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    #
    g = g.to(device)
    deg = g.in_degrees().cuda().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm
    del U, V, select, valid_ids, norm, deg

    num_targets = num_classes
    loss_fn = F.cross_entropy
    metric = accuracy

    acc_list = []
    torch.manual_seed(0)
    split_seed = 1234567
    for graph_idx in range(args.run):
        print(f'Split [{graph_idx + 1}/{args.run}]')
        # generate split
        idx_train, idx_val, idx_test = random_splits_with_unlabel(
            labels_th, ratio=[60, 20, 20], seed=split_seed)
        split_seed += 1
        # initialize
        net = FAGCN(g, num_features, args.n_hid, num_classes,
                    args.dropout, args.eps, args.layer_num).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        test_acc = train(
            args, feat_data_th, labels_th, num_targets,
            idx_train, idx_val, idx_test, net, optimizer)
        acc_list.append(test_acc)

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = f'./pathnet.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.dataset}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test graph dataset used in PathNet")
    parser.add_argument('--dataset', type=str, default='Bgp', help='dataset name')
    # model training parameters
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--method', type=str, default='FAGCN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--n_hid', type=int, default=64, help='Number of hidden dim')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    # FAGCN
    parser.add_argument('--remove_zero_in_degree_nodes', action='store_true')
    parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
    parser.add_argument('--patience', type=int, default=10000, help='Patience')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
