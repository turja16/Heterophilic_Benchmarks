import argparse
from os import path
from typing import NamedTuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from FSGCN.FSGCN_models import FSGNN
from FSGCN.FSGCN_training import run_on_split


# only for accuracy


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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.DoubleTensor(indices, values, shape)


def train_criticaldata_fsgcn(device: torch.device,
                             args: Union[NamedTuple, argparse.Namespace]):
    dataset_str = f'{args.dataset.replace("-", "_")}'
    # load critical data
    npz_data = np.load(f'{path.dirname(path.abspath(__file__))}/../critical_look_utils/data/{dataset_str}.npz')
    if 'directed' not in dataset_str:
        edge = np.concatenate((npz_data['edges'], npz_data['edges'][:, ::-1]), axis=0)
    else:
        edge = npz_data['edges']
    #
    labels = npz_data['node_labels']
    features = npz_data['node_features']
    features = normalize_tensor_sparse(features, symmetric=0)
    features = torch.FloatTensor(features)
    features = features.to(device)

    if len(labels.shape) == 1:
        labels = torch.from_numpy(labels)
    else:
        labels = torch.from_numpy(labels).argmax(dim=-1)

    labels = labels.to(device)

    n, c, d = features.shape[0], len(labels.unique()), features.shape[1]

    num_features = d
    num_labels = c

    # get adjacency matrix and its powers (FSGNN)
    adj = to_scipy_sparse_matrix(torch.from_numpy(edge.T))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to(device)

    adj_i = to_scipy_sparse_matrix(add_self_loops(torch.from_numpy(edge.T))[0])
    adj_i = sparse_mx_to_torch_sparse_tensor(adj_i)
    adj_i = adj_i.to(device)

    list_mat = []
    list_mat.append(features)
    no_loop_mat = features
    loop_mat = features

    for ii in range(args.num_layers):
        no_loop_mat = torch.spmm(adj, no_loop_mat)
        loop_mat = torch.spmm(adj_i, loop_mat)
        list_mat.append(no_loop_mat)
        list_mat.append(loop_mat)

    # Select X and self-looped features
    if args.feat_type == "homophily":
        select_idx = [0] + [2 * ll for ll in range(1, args.num_layers + 1)]
        list_mat = [list_mat[ll] for ll in select_idx]
    # Select X and no-loop features
    elif args.feat_type == "heterophily":
        select_idx = [0] + [2 * ll - 1 for ll in range(1, args.num_layers + 1)]
        list_mat = [list_mat[ll] for ll in select_idx]

    # Otherwise all hop features are selected

    acc_list = []
    torch.manual_seed(0)
    num_splits = args.run
    for i in range(num_splits):
        print(f'Split [{i + 1}/{num_splits}]')
        train_mask = npz_data['train_masks'][i]
        val_mask = npz_data['val_masks'][i]
        test_mask = npz_data['test_masks'][i]
        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        model = FSGNN(
            nfeat=num_features,
            nlayers=len(list_mat),
            nhidden=args.n_hid,
            nclass=num_labels,
            dropout=args.dropout,
            layer_norm=args.layer_norm,
        ).to(device)
        #
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #
        test_acc = run_on_split(
            model, optimizer, features, labels, list_mat, train_mask, val_mask, test_mask, device, args)
        #
        print(f'Test accuracy {test_acc:.4f}')
        acc_list.append(test_acc)

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = f'./critical.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.dataset}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test graph dataset used in PathNet")
    parser.add_argument('--dataset', type=str, default='squirrel-filtered', help='dataset name')
    # model training parameters
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--method', type=str, default='FSGCN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--n_hid', type=int, default=512, help='Number of hidden dim')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-7)
    # fsgcn
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers (hops).')
    parser.add_argument('--feat_type', type=str, default='all', help='Type of features to be used')
    parser.add_argument('--layer_norm', type=int, default=1, help='layer norm')
    parser.add_argument('--patience', type=int, default=10000, help='Patience')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    train_criticaldata_fsgcn(device, args)
