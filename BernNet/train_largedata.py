import argparse
import random
import sys
from copy import deepcopy
from os import path
from typing import NamedTuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from BernNet.models import BernNet

sys.path.append("/home/xsslnc/scratch/hetero_metric_win_ver")
from large_scale_data_utils.dataset import load_nc_dataset
from torch_geometric.utils import remove_self_loops


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
    # y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    y_have_label_mask = labels != -1  # ignore unlabeled nodes
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
          edge_index,
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
        output = model(features, edge_index).squeeze(1)
        loss_train = loss_fn(output[idx_train], labels[idx_train])
        metric_train = metric(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        #
        model.eval()
        output = model(features, edge_index).squeeze(1)
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
    output = model(features, edge_index).squeeze(1)
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    metric_test = metric(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "metric= {:.4f}".format(metric_test.item()))
    return metric_test.item()


def train_largedata(device: torch.device,
                    args: Union[NamedTuple, argparse.Namespace]):
    # large scale + geom
    if args.dataset in ['deezer-europe', 'genius', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer',
                        'Cora',
                        'CiteSeer', 'PubMed', 'chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        if args.dataset == 'penn94':
            dataname = 'fb100'
            sub_dataname = 'Penn94'
        else:
            dataname = args.dataset
            sub_dataname = ''
        dataset = load_nc_dataset(dataname, sub_dataname)
        edge_index, feat, label = dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.label
        edge_index = remove_self_loops(edge_index)[0]
        if args.dataset in ['genius', 'deezer-europe', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer']:
            name = args.dataset
            if sub_dataname != '':
                name = f'{dataname}-{sub_dataname}'
            #
            BASE_DIR = f"{path.dirname(path.abspath(__file__))}/../splits"
            splits_lst = np.load(f'{BASE_DIR}/{name}-splits.npy', allow_pickle=True)
            for i in range(len(splits_lst)):
                for key in splits_lst[i]:
                    if not torch.is_tensor(splits_lst[i][key]):
                        splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        else:
            splits_lst = None
    else:
        raise ValueError('Invalid data name')

    features = normalize_tensor_sparse(feat, symmetric=0)
    features = torch.FloatTensor(features).to(device)
    n, d = feat.shape[0], feat.shape[1]
    c = len(label.unique())
    if args.dataset == 'arxiv-year':
        label = label.squeeze()

    if args.dataset == 'penn94':
        assert -1 in label
        c = c - 1  # omit unlabled

    labels = label.to(device)
    num_targets = c
    edge_index = edge_index.to(device)

    if args.dataset == 'genius':
        loss_fn = F.binary_cross_entropy_with_logits
        metric = roc_auc
        labels = labels.float()
        num_targets = 1  # num of targets
    else:
        loss_fn = F.cross_entropy
        metric = accuracy

    # training
    acc_list = []
    torch.manual_seed(0)
    split_seed = 1234567

    if args.dataset in ['genius', 'deezer-europe', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer']:
        num_splits = len(splits_lst)
    else:
        num_splits = args.run

    for i in range(num_splits):
        print(f'Split [{i + 1}/{num_splits}]')
        # load a split
        if args.dataset in ['genius', 'deezer-europe', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer']:
            print('load fixed split')
            idx_train, idx_val, idx_test = splits_lst[i]['train'], splits_lst[i]['valid'], splits_lst[i]['test']
        else:
            idx_train, idx_val, idx_test = random_splits(
                labels, ratio=[60, 20, 20], seed=split_seed)
            split_seed += 1
        #
        model = BernNet(d, args.n_hid, num_targets, args).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
        test_metric = train(model,
                            args.epoch_num,
                            args.early_stopping,
                            optimizer,
                            features,
                            edge_index,
                            loss_fn,
                            metric,
                            labels,
                            idx_train,
                            idx_val,
                            idx_test)
        acc_list.append(test_metric)

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = f'./large.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.dataset}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test graph dataset used in PathNet")
    parser.add_argument('--dataset', type=str, default='penn94', help='dataset name')
    # model training parameters
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--method', type=str, default='BernNet', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--n_hid', type=int, default=64, help='Number of hidden dim')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=10000,
                        help='Early stopping')
    # bernet
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
    train_largedata(device, args)
