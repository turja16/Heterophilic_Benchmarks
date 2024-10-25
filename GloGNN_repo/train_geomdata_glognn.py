import argparse
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from parse import parse_method, parser_add_main_args
# from helper import NCDataset
from torch_geometric.utils.convert import to_scipy_sparse_matrix

import sys
import random
sys.path.append("/home/xsslnc/scratch/hetero_metric_win_ver")
from large_scale_data_utils.dataset import load_nc_dataset
from torch_geometric.utils import to_dense_adj, remove_self_loops, dense_to_sparse

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
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy()) 

def train(model, optimizer, features, adj):
    best_metric = 0
    patience = 0
    best_params = None
    for epoch in range(args.epoch_num):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        # output = model(data)
        loss_train = loss_fn(output[idx_train], labels[idx_train])
        metric_train = metric(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        # 
        model.eval()
        output = model(features, adj)
        # output = model(data)
        loss_val = loss_fn(output[idx_val], labels[idx_val])
        metric_val = metric(output[idx_val], labels[idx_val])
        print("Train loss= {:.4f}".format(loss_train.item()),
            "Val metric= {:.4f}".format(metric_val.item()))
        if metric_val > best_metric:
            best_metric = metric_val
            best_params = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        #
        if patience >= args.early_stopping:
            break
    # test
    model.load_state_dict(best_params)
    # Testing
    model.eval()
    output = model(features, adj)
    # output = model(data)
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    metric_test = metric(output[idx_test], labels[idx_test])
    print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "metric= {:.4f}".format(metric_test.item()))
    return metric_test.item()


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

def random_splits(labels, ratio: list = [60, 20, 20], seed: int = 1234567):
    labels = labels.cpu()
    total_node_num = len(labels)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    (train_index, val_index, test_index) = get_order(
        ratio, y_index_tensor, total_node_num, seed)
    return (train_index, val_index, test_index)


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)
device = torch.device("cuda:" + str(args.cuda))

# large scale + geom
if args.dataset in ['deezer-europe', 'genius', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamers', 'Cora', 'CiteSeer', 'PubMed', 'chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
    if args.dataset == 'penn94':
        dataname = 'fb100'
        sub_dataname = 'Penn94'
    else:
        dataname = args.dataset
        sub_dataname = ''
    dataset = load_nc_dataset(dataname, sub_dataname)
    edge_index, feat, label = dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.label
    edge_index = remove_self_loops(edge_index)[0]
else:
    raise ValueError('Invalid data name')

features = normalize_tensor_sparse(feat, symmetric=0)
features = torch.FloatTensor(features).to(device)
n, c, d = feat.shape[0], len(label.unique()), feat.shape[1]
labels = label.to(device)

adj = to_scipy_sparse_matrix(edge_index)
adj.setdiag(1) # A + I
adj = normalize_tensor_sparse(adj, symmetric=1) # csc_matrix
adj = sp.csr_matrix(adj) 
adj = sparse_mx_to_torch_sparse_tensor(adj)
adj = adj.to(device)

# # format
# data = NCDataset(f'{args.dataset.replace("-", "_")}.npz')
# data.graph = {
#     'edge_index': edge_index.to(device),
#     'node_feat': features,
#     'edge_feat': None,
#     'num_nodes': n}

num_targets = 1 if c == 2 else c
loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
metric = accuracy if c > 2 else roc_auc

acc_list = []
torch.manual_seed(0)
split_seed = 1234567
for graph_idx in range(args.run):
    # generate split
    idx_train, idx_val, idx_test = random_splits(
        labels, ratio = [60, 20, 20], seed = split_seed)
    split_seed += 1
    # model
    model = parse_method(args, n, c, d, device, edge_index=None)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    test_metric = train(model, optimizer, features, adj)
    acc_list.append(test_metric)

test_mean = np.mean(acc_list)
test_std = np.std(acc_list)
filename = f'./{args.method.lower()}_geom.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(f"{args.method.lower()}, " +
                    f"{args.dataset}, " +
                    f"{test_mean:.4f}, " +
                    f"{test_std:.4f}, " +
                    f"{args}\n")


