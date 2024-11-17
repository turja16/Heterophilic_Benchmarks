import argparse
import sys
from copy import deepcopy
from os import path
from typing import NamedTuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils.convert import to_scipy_sparse_matrix

sys.path.append("./Heterophilic_Benchmarks/GloGNN_repo/")

from acm import GCN
from parse import parser_add_main_args


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


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean()


@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits.cpu().numpy())


def train(model,
          epoch_num,
          early_stopping,
          optimizer,
          features,
          adj_low,
          adj_high,
          adj_low_unnormalized,
          loss_fn,
          metric,
          labels,
          idx_train,
          idx_val,
          idx_test):
    best_metric = 0
    patience = 0
    best_params = None
    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        # output = model(features, adj)
        # output = model(data).squeeze(1)
        output = model(features, adj_low, adj_high, adj_low_unnormalized).squeeze(1)
        loss_train = loss_fn(output[idx_train], labels[idx_train])
        metric_train = metric(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        # 
        model.eval()
        # output = model(data).squeeze(1)
        output = model(features, adj_low, adj_high, adj_low_unnormalized).squeeze(1)
        loss_val = loss_fn(output[idx_val], labels[idx_val])
        metric_val = metric(output[idx_val], labels[idx_val])
        if (epoch + 1) % 100 == 0:
            print("Train loss= {:.4f}".format(loss_train.item()),
                  "Val metric= {:.4f}".format(metric_val.item()))
        if metric_val > best_metric:
            best_metric = metric_val
            best_params = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        #
        if patience >= early_stopping:
            break
    # test
    model.load_state_dict(best_params)
    # Testing
    model.eval()
    # output = model(data).squeeze(1)
    output = model(features, adj_low, adj_high, adj_low_unnormalized).squeeze(1)
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    metric_test = metric(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "metric= {:.4f}".format(metric_test.item()))
    return metric_test.item()


def train_criticaldata_acm(device: torch.device,
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
    adj = nx.adj_matrix(nx.from_edgelist(edge))
    #
    features = normalize_tensor_sparse(features, symmetric=0)
    features = torch.FloatTensor(features)
    if len(labels.shape) == 1:
        labels = torch.from_numpy(labels)
    else:
        labels = torch.from_numpy(labels).argmax(dim=-1)

    n, c, d = adj.shape[0], labels.max().item() + 1, features.shape[1]

    # acm
    edge_index = torch.tensor(edge.T).to(device)
    adj_low_unnormalized = to_scipy_sparse_matrix(edge_index)
    adj_low = normalize_tensor_sparse(sp.identity(n) + adj_low_unnormalized, symmetric=1)
    adj_high = sp.identity(n) - adj_low
    adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(device)
    adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(device)
    adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj_low_unnormalized).to(device)

    if not args.structure_info:
        adj_low_unnormalized = None

    features = features.to(device)
    labels = labels.to(device)

    num_targets = 1 if c == 2 else c
    loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
    metric = accuracy if c > 2 else roc_auc
    if num_targets == 1:
        labels = labels.float()

    acc_list = []
    torch.manual_seed(0)
    for split in range(args.run):
        # load a split for critical look
        train_mask = npz_data['train_masks'][split]
        val_mask = npz_data['val_masks'][split]
        test_mask = npz_data['test_masks'][split]
        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        # model: ACM GMM
        model = GCN(nfeat=d,
                    nhid=args.hidden_channels,
                    nclass=num_targets,
                    nlayers=args.num_layers,
                    nnodes=n,
                    dropout=args.dropout,
                    model_type=args.acm_method,
                    structure_info=args.structure_info,
                    variant=args.variant).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay)
        test_metric = train(model,
                            args.epoch_num,
                            args.early_stopping,
                            optimizer,
                            features,
                            adj_low,
                            adj_high,
                            adj_low_unnormalized,
                            loss_fn,
                            metric,
                            labels,
                            idx_train,
                            idx_val,
                            idx_test)
        acc_list.append(test_metric)

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = f'./{args.method.lower()}_critical.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.dataset}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.cuda))

    train_criticaldata_acm(device, args)
