import argparse
from os import path
from typing import NamedTuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from GPRGNN.GPRGNN_models import GPRGNN
from GPRGNN.GPRGNN_training import RunExp


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean()


@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy())


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


def train_criticaldata_gprgnn(device: torch.device,
                              args: Union[NamedTuple, argparse.Namespace]):
    dataset_str = f'{args.dataset.replace("-", "_")}'
    # load critical data
    npz_data = np.load(f'{path.dirname(path.abspath(__file__))}/../critical_look_utils/data/{dataset_str}.npz')
    if 'directed' not in dataset_str:
        edge = np.concatenate((npz_data['edges'], npz_data['edges'][:, ::-1]), axis=0)
    else:
        edge = npz_data['edges']

    labels = npz_data['node_labels']
    labels_th = torch.LongTensor(labels).to(device)
    features = npz_data['node_features']
    features = normalize_tensor_sparse(features, symmetric=0)
    feat_data_th = torch.FloatTensor(features).to(device)
    edge_index = torch.as_tensor(edge.T).to(device)

    c = labels.max().item() + 1
    num_classes = c
    num_features = features.shape[1]
    num_targets = 1 if c == 2 else c
    loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
    metric = accuracy if c > 2 else roc_auc

    # training
    acc_list = []
    torch.manual_seed(0)
    split_seed = 1234567
    #
    Init = args.Init
    Gamma_0 = None
    alpha = args.alpha

    for graph_idx in range(args.run):
        # load a split for critical look
        train_mask = npz_data['train_masks'][graph_idx]
        val_mask = npz_data['val_masks'][graph_idx]
        test_mask = npz_data['test_masks'][graph_idx]
        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]
        train_nodes = torch.LongTensor(idx_train)
        valid_nodes = torch.LongTensor(idx_val)
        test_nodes = torch.LongTensor(idx_test)
        #
        Net = GPRGNN(num_features, num_classes, Gamma_0, args)
        # train
        test_acc, best_val_acc, Gamma_0, = RunExp(
            args, Net, feat_data_th, edge_index, labels_th, train_nodes, valid_nodes, test_nodes, device,
            loss_fn, metric)
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
    parser.add_argument('--method', type=str, default='GPRGNN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--n_hid', type=int, default=512, help='Number of hidden dim')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # GPRGNN parameter
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    # https://github.com/jianhao2016/GPRGNN/blob/master/src/train_model.py
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--ppnp',
                        default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--early_stopping', type=int, default=10000)

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    train_criticaldata_gprgnn(device, args)
