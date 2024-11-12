import argparse
import random

import torch

import numpy as np
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix
from typing import NamedTuple, Union

from models.model import GCN, MLP1, MLP2, SGC1
from utils.data_loader import DataLoader
from utils.train_helper import train

BASE_DIR = './Opengsl'


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


def random_splits(labels, ratio: list = [60, 20, 20], seed: int = 1234567):
    labels = labels.cpu()
    total_node_num = len(labels)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    (train_index, val_index, test_index) = get_order(
        ratio, y_index_tensor, total_node_num, seed)
    return (train_index, val_index, test_index)


def train_opengsldata_baseline(device: torch.device,
                               args: Union[NamedTuple, argparse.Namespace]):
    if args.dataset == 'wiki-cooc':
        # load
        file_name = f'{args.dataset.replace("-", "_")}.npz'
        data = np.load(os.path.join(BASE_DIR, file_name))
        feats = torch.tensor(data['node_features'])  #
        labels = torch.tensor(data['node_labels'])  #
        edges = torch.tensor(data['edges'])
        # get all fixed splits
        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])
        train_indices = [torch.nonzero(x, as_tuple=False).squeeze() for x in train_masks]
        val_indices = [torch.nonzero(x, as_tuple=False).squeeze() for x in val_masks]
        test_indices = [torch.nonzero(x, as_tuple=False).squeeze() for x in test_masks]
        print('has fixed {} splits'.format(len(train_indices)))
        # get essential
        num_nodes = feats.shape[0]  #
        num_classes = len(labels.unique())  #
        edge_index = remove_self_loops(edges.T)[0]
        adj = to_scipy_sparse_matrix(edge_index)  #
    elif args.dataset in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=BASE_DIR, name=args.dataset)
        g = dataset[0]
        feats = g.x  # unnormalized #
        if args.dataset == 'flickr':
            feats = feats.to_dense()
        num_nodes = feats.shape[0]  #
        num_classes = dataset.num_classes  #
        labels = g.y  #
        edge_index = remove_self_loops(g.edge_index)[0]
        adj = to_scipy_sparse_matrix(edge_index)  #
    else:
        raise ValueError('dataset does not exist')

    # format
    labels_th = torch.LongTensor(labels).to(device)
    feat_data_sparse = sp.coo_matrix(feats)
    # normalize
    feat_data_sparse = normalize_tensor_sparse(feat_data_sparse, symmetric=0)
    feat_data_th = torch.tensor(feat_data_sparse.toarray(), dtype=torch.float32).to(device)
    del feat_data_sparse
    adj.setdiag(1)  # A + I
    adj = normalize_tensor_sparse(adj, symmetric=1)  # csc_matrix
    adj = sp.csr_matrix(adj)

    criterion = nn.CrossEntropyLoss()
    acc_list = []
    train_time = []
    torch.manual_seed(0)
    split_seed = 1234567
    for graph_idx in range(args.run):
        if args.dataset == 'wiki-cooc':
            train_nodes = train_indices[graph_idx]
            valid_nodes = val_indices[graph_idx]
            test_nodes = test_indices[graph_idx]
        elif args.dataset in ['blogcatalog', 'flickr']:
            # generate split
            train_nodes, valid_nodes, test_nodes = random_splits(
                labels_th, ratio=[60, 20, 20], seed=split_seed)
            split_seed += 1
        #
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
    filename = f'./exp_res/opengsl.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.dataset}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test graph dataset used in PathNet")
    parser.add_argument('--dataset', type=str, default='wiki-cooc', help='dataset name')
    # model training parameters
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--method', type=str, default='GCN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--n_hid', type=int, default=64, help='Number of hidden dim')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-7)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
