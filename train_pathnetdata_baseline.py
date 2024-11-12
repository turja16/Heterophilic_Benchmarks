import argparse
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix
from typing import NamedTuple, Union

from models.model import GCN, MLP1, MLP2, SGC1
from utils.data_loader import DataLoader
from utils.train_helper import train


BASE_DIR = "./PathNet/other_data"


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


def normalize_tensor_sparse(mx, symmetric=0):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
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


def train_pathnetdata_basline(device: torch.device,
                              args: Union[NamedTuple, argparse.Namespace]):
    # load data
    x = np.load(BASE_DIR + '/' + args.dataset + '/x.npy')
    y = np.load(BASE_DIR + '/' + args.dataset + '/y.npy')
    numpy_edge_index = np.load(BASE_DIR + '/' + args.dataset + '/edge_index.npy')
    edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)
    edge_index = remove_self_loops(edge_index)[0]
    del numpy_edge_index
    # get stat
    num_nodes = x.shape[0]
    adj = to_scipy_sparse_matrix(edge_index)

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
    adj.setdiag(1)
    adj = normalize_tensor_sparse(adj, symmetric=1)  # csc_matrix
    adj = sp.csr_matrix(adj)

    criterion = nn.CrossEntropyLoss()
    acc_list = []
    train_time = []
    torch.manual_seed(0)
    split_seed = 1234567
    for graph_idx in range(args.run):
        # generate split
        train_nodes, valid_nodes, test_nodes = random_splits_with_unlabel(
            labels_th, ratio=[60, 20, 20], seed=split_seed)
        split_seed += 1
        data_loader = DataLoader(adj, train_nodes, valid_nodes, test_nodes, device)
        # select model
        if args.method == 'GCN':
            model = GCN(n_feat=feat_data_th.shape[1], n_hid=args.n_hid, n_classes=num_classes, dropout=args.dropout,
                        criterion=criterion).to(device)
        elif args.method == 'MLP2':
            model = MLP2(n_feat=feat_data_th.shape[1], n_hid=args.n_hid, n_classes=num_classes, dropout=args.dropout,
                         criterion=criterion).to(device)
        elif args.method == 'MLP1':
            model = MLP1(n_feat=feat_data_th.shape[1], n_classes=num_classes, criterion=criterion).to(device)
        elif args.method == 'SGC1':
            model = SGC1(n_feat=feat_data_th.shape[1], n_classes=num_classes, criterion=criterion).to(device)
        else:
            raise ValueError('model does not exist')
        # print(model)
        # train
        results, acc, avg_time = train(
            model,
            args, feat_data_th, labels_th,
            data_loader, device, note="%s" % (args.method)
        )
        eval_model = results.best_model.to(device)
        output = eval_model(feat_data_th, data_loader.lap_tensor)
        output = output.argmax(dim=1)
        acc = f1_score(
            output[test_nodes].detach().cpu(),
            labels_th[test_nodes].detach().cpu(),
            average="micro")
        acc_list.append(acc)
        train_time.append(avg_time)

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = f'./exp_res/pathnet.csv'
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
    parser.add_argument('--method', type=str, default='GCN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_hid', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    train_pathnetdata_basline(device, args)
