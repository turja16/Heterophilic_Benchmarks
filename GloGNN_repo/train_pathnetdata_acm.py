import argparse
import random
from copy import deepcopy
from os import path
from typing import NamedTuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from acm import GCN
from parse import parser_add_main_args

BASE_DIR = f"{path.dirname(path.abspath(__file__))}/../PathNet/other_data"


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


def train_pathnetdata_acm(device: torch.device,
                          args: Union[NamedTuple, argparse.Namespace]):
    x = np.load(BASE_DIR + '/' + args.dataset + '/x.npy')
    y = np.load(BASE_DIR + '/' + args.dataset + '/y.npy')
    numpy_edge_index = np.load(BASE_DIR + '/' + args.dataset + '/edge_index.npy')
    edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)
    edge_index = remove_self_loops(edge_index)[0]
    del numpy_edge_index
    # get stat
    num_nodes = x.shape[0]

    lbl_set = []
    for lbl in y:
        if lbl not in lbl_set:
            lbl_set.append(lbl)

    if -1 in lbl_set:
        print('have unlabeled data; will be excluded in train/valid/test set')
        num_classes = len(lbl_set) - 1
    else:
        num_classes = len(lbl_set)

    labels = torch.LongTensor(y).to(device)
    feat_data_sparse = sp.coo_matrix(x)
    del y, x

    # normalize
    feat_data_sparse = normalize_tensor_sparse(feat_data_sparse, symmetric=0)
    features = torch.tensor(feat_data_sparse.toarray(), dtype=torch.float32).to(device)
    del feat_data_sparse

    n, c, d = features.shape[0], num_classes, features.shape[1]

    # acm
    edge_index = edge_index.to(device)
    adj_low_unnormalized = to_scipy_sparse_matrix(edge_index)
    adj_low = normalize_tensor_sparse(sp.identity(n) + adj_low_unnormalized, symmetric=1)
    adj_high = sp.identity(n) - adj_low
    adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(device)
    adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(device)
    adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj_low_unnormalized).to(device)

    if not args.structure_info:
        adj_low_unnormalized = None

    num_targets = c
    loss_fn = F.cross_entropy
    metric = accuracy

    acc_list = []
    torch.manual_seed(0)
    split_seed = 1234567
    for graph_idx in range(args.run):
        # generate split
        idx_train, idx_val, idx_test = random_splits_with_unlabel(
            labels, ratio=[60, 20, 20], seed=split_seed)
        split_seed += 1
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
    filename = f'./{args.method.lower()}_pathnet.csv'
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
