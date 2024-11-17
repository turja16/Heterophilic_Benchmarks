import argparse
from os import path
from typing import NamedTuple, Union

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from FAGCN.FAGCN_models import FAGCN
from FAGCN.FAGCN_training import train


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean()


@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits.cpu().numpy())


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


def train_criticaldata_fagcn(device: torch.device,
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
    #
    features = normalize_tensor_sparse(features, symmetric=0)
    features = torch.FloatTensor(features)
    if len(labels.shape) == 1:
        labels = torch.from_numpy(labels)
    else:
        labels = torch.from_numpy(labels).argmax(dim=-1)

    features = features.to(device)
    labels = labels.to(device)
    edge_index = torch.tensor(edge.T).to(device)

    n, c, d = features.shape[0], labels.max().item() + 1, features.shape[1]
    num_features = d
    num_targets = 1 if c == 2 else c
    loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
    metric = accuracy if c > 2 else roc_auc
    if num_targets == 1:
        labels = labels.float()

    edge_index = edge_index.cpu().T.numpy()
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

    acc_list = []
    torch.manual_seed(0)
    # split_seed = 1234567
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
        net = FAGCN(g, num_features, args.n_hid, num_targets,
                    args.dropout, args.eps, args.layer_num).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        test_acc = train(
            args, features, labels, num_targets,
            idx_train, idx_val, idx_test, net, optimizer)
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
    parser.add_argument('--method', type=str, default='FAGCN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--n_hid', type=int, default=512, help='Number of hidden dim')
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
    train_criticaldata_fagcn(device, args)
