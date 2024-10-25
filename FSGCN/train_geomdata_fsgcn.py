import torch
import random
import argparse
import numpy as np
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops
from FSGCN_models import FSGNN
from FSGCN_training import run_on_split

import sys
import random
import scipy.sparse as sp
from torch_geometric.utils.convert import to_scipy_sparse_matrix
sys.path.append("/home/xsslnc/scratch/hetero_metric_win_ver")
from large_scale_data_utils.dataset import load_nc_dataset
from torch_geometric.utils import to_dense_adj, remove_self_loops, dense_to_sparse

parser = argparse.ArgumentParser(description="Test graph dataset used in PathNet")
parser.add_argument('--dataset', type=str, default='cornell', help='dataset name')
# model training parameters
parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
parser.add_argument('--method', type=str, default='FSGCN', help='which model to use')
parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
parser.add_argument('--n_hid', type=int, default=64, help='Number of hidden dim')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-7)
# fsgcn
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers (hops).')
parser.add_argument('--feat_type',type=str, default='all', help='Type of features to be used')
parser.add_argument('--layer_norm', type=int, default=1, help='layer norm')
parser.add_argument('--patience', type=int, default=10000, help='Patience')
args = parser.parse_args()

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

def random_splits(labels, ratio: list = [60, 20, 20], seed: int = 1234567):
    labels = labels.cpu()
    total_node_num = len(labels)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    (train_index, val_index, test_index) = get_order(
        ratio, y_index_tensor, total_node_num, seed)
    return (train_index, val_index, test_index)

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

# adj = to_scipy_sparse_matrix(edge_index)
# adj.setdiag(1) # A + I
# adj = normalize_tensor_sparse(adj, symmetric=1) # csc_matrix
# adj = sp.csr_matrix(adj) 
# adj = sparse_mx_to_torch_sparse_tensor(adj)
# adj = adj.to(device)

num_features = d
num_labels = c
# get adjacency matrix and its powers (FSGNN)
adj = to_dense_adj(edge_index)[0].to(device)
adj_i = to_dense_adj(add_self_loops(edge_index)[0])[0].to(device)

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
#Select X and no-loop features
elif args.feat_type == "heterophily":
    select_idx = [0] + [2 * ll - 1 for ll in range(1, args.num_layers + 1)]
    list_mat = [list_mat[ll] for ll in select_idx]
#Otherwise all hop features are selected

acc_list = []
torch.manual_seed(0)
split_seed = 1234567
num_splits = args.run
for i in range(num_splits):
    print(f'Split [{i+1}/{num_splits}]')
    idx_train, idx_val, idx_test = random_splits(
        labels, ratio = [60, 20, 20], seed = split_seed)
    split_seed += 1
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
        model, optimizer, features, labels, 
        list_mat, idx_train, idx_val, idx_test, device, args)
    #
    print(f'Test accuracy {test_acc:.4f}')
    acc_list.append(test_acc)

test_mean = np.mean(acc_list)
test_std = np.std(acc_list)
filename = f'./geom.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(f"{args.method.lower()}, " +
                    f"{args.dataset}, " +
                    f"{test_mean:.4f}, " +
                    f"{test_std:.4f}, " +
                    f"{args}\n")

