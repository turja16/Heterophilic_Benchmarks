import os
import random
import sys
from os import path

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor, LongTensor
from torch_geometric.utils import is_undirected, to_undirected

sys.path.append("/home/xsslnc/scratch/hetero_metric_win_ver")
from large_scale_data_utils.dataset import load_nc_dataset
from torch_geometric.utils import remove_self_loops
from torch_geometric.datasets import AttributedGraphDataset


class BaseGraph:
    '''
        A general format for datasets.
        Args:
            x (Tensor): node feature, of shape (number of node, F).
            edge_index (LongTensor): of shape (2, number of edge)
            edge_weight (Tensor): of shape (number of edge)
            # mask: a node mask to show a training/valid/test dataset split, of shape (number of node). mask[i]=0, 1, 2 means the i-th node in train, valid, test dataset respectively.
    '''

    def __init__(self, x: Tensor, edge_index: LongTensor, edge_weight: Tensor,
                 y: Tensor, num_targets):  # , mask: LongTensor):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_weight
        self.y = y
        self.num_classes = torch.unique(y).shape[0]
        self.num_nodes = x.shape[0]
        # self.mask = mask
        self.to_undirected()
        self.num_targets = num_targets

    # def get_split(self, split: str):
    #     tar_mask = {"train": 0, "valid": 1, "test": 2}[split]
    #     tmask = (self.mask == tar_mask) # bool
    #     return self.edge_index, self.edge_attr, tmask, self.y[tmask]
    def get_split(self, split_mask):
        # tar_mask = {"train": 0, "valid": 1, "test": 2}[split]
        # tmask = (self.mask == tar_mask) # bool
        return self.edge_index, self.edge_attr, split_mask, self.y[split_mask]

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            self.edge_index, self.edge_attr = to_undirected(
                self.edge_index, self.edge_attr)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.y = self.y.to(device)
        # self.mask = self.mask.to(device)
        return self


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


def load_geom(name):
    if name == 'penn94':
        dataname = 'fb100'
        sub_dataname = 'Penn94'
    else:
        dataname = name
        sub_dataname = ''
    #
    dataset = load_nc_dataset(dataname, sub_dataname)
    edge_index, feat, labels = dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.label
    edge_index = remove_self_loops(edge_index)[0]
    features = normalize_tensor_sparse(feat, symmetric=0)
    features = torch.FloatTensor(features)
    if name == 'arxiv-year':
        labels = labels.squeeze()
    ##
    lbl_set = labels.unique()
    c = len(lbl_set) - 1 if -1 in lbl_set else len(
        lbl_set)  # penn94 has unlabeled -1, will be excluded using fixed splits
    # set number of targets
    num_targets = 1 if name == 'genius' else c
    ##################################
    # get all splits
    num_node = features.shape[0]
    if name in ['genius', 'deezer-europe', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer']:
        if sub_dataname != '':
            name = f'{dataname}-{sub_dataname}'

        BASE_DIR = f"{path.dirname(path.abspath(__file__))}/../splits"
        splits_lst = np.load(f'{BASE_DIR}/{name}-splits.npy', allow_pickle=True)
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
                splits_lst[i][key] = index_to_mask(splits_lst[i][key], num_node)
    else:
        splits_lst = []
    return features, edge_index, labels, num_targets, splits_lst


def load_critical(name):
    npz_data = np.load(f'../critical_look_utils/data/{name}.npz', )
    edges = np.concatenate((npz_data['edges'], npz_data['edges'][:, ::-1]), axis=0)
    features = torch.from_numpy(npz_data['node_features'])
    features = normalize_tensor_sparse(features, symmetric=0)
    features = torch.FloatTensor(features)
    labels = torch.from_numpy(npz_data['node_labels'])
    edge_index = torch.from_numpy(edges).T
    edge_index = remove_self_loops(edge_index)[0]
    c = labels.max().item() + 1
    num_targets = 1 if c == 2 else c
    # fixed 10 split
    train_mask = torch.from_numpy(npz_data['train_masks'])  # 10 by *
    val_mask = torch.from_numpy(npz_data['val_masks'])
    test_mask = torch.from_numpy(npz_data['test_masks'])
    # format split
    splits_lst = []
    for i in range(train_mask.shape[0]):  # splits
        splits_lst.append({'train': train_mask[i], 'valid': val_mask[i], 'test': test_mask[i]})
    return features, edge_index, labels, num_targets, splits_lst


def load_opengsl(name):
    BASE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../Opengsl"
    if name == 'wiki_cooc':
        # load
        file_name = f'{name.replace("-", "_")}.npz'
        data = np.load(os.path.join(BASE_DIR, file_name))
        feats = torch.tensor(data['node_features'])  #
        labels = torch.tensor(data['node_labels'])  #
        edges = torch.tensor(data['edges'])
        edge_index = remove_self_loops(edges.T)[0]
        # normalize
        features = normalize_tensor_sparse(feats, symmetric=0)
        features = torch.FloatTensor(features)
        # get fixed splits
        train_mask = torch.tensor(data['train_masks'])
        val_mask = torch.tensor(data['val_masks'])
        test_mask = torch.tensor(data['test_masks'])
        del edges, data
        # format split
        splits_lst = []
        for i in range(train_mask.shape[0]):
            splits_lst.append({'train': train_mask[i], 'valid': val_mask[i], 'test': test_mask[i]})
            ##################################
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=BASE_DIR, name=name)
        g = dataset[0]
        feats = g.x  # unnormalized #
        if name == 'flickr':
            feats = feats.to_dense()
        labels = g.y  #
        edge_index = remove_self_loops(g.edge_index)[0]
        # normalize
        features = normalize_tensor_sparse(feats, symmetric=0)
        features = torch.FloatTensor(features)
        del dataset
        splits_lst = []
    ##################################
    num_targets = len(labels.unique())
    return features, edge_index, labels, num_targets, splits_lst


def load_pathnet(name):
    # load pathnet data
    BASE_DIR = f"{path.dirname(path.abspath(__file__))}/../PathNet/other_data"
    x = np.load(BASE_DIR + '/' + name + '/x.npy')
    y = np.load(BASE_DIR + '/' + name + '/y.npy')
    numpy_edge_index = np.load(BASE_DIR + '/' + name + '/edge_index.npy')
    edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)
    edge_index = remove_self_loops(edge_index)[0]
    del numpy_edge_index
    labels = torch.LongTensor(y)
    feat_data_sparse = sp.coo_matrix(x)
    del y, x
    # normalize
    feat_data_sparse = normalize_tensor_sparse(feat_data_sparse, symmetric=0)
    features = torch.tensor(feat_data_sparse.toarray(), dtype=torch.float32)
    del feat_data_sparse
    # have unlabeled data
    lbl_set = labels.unique()
    num_targets = len(lbl_set) - 1 if -1 in lbl_set else len(lbl_set)
    splits_lst = []
    return features, edge_index, labels, num_targets, splits_lst


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


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_dataset(name: str):
    # large scale + geom
    if name in ['deezer-europe', 'genius', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer', 'Cora',
                'CiteSeer', 'PubMed', 'chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        features, edge_index, labels, num_targets, splits_lst = load_geom(name)
    elif name in ['squirrel_filtered', 'chameleon_filtered', 'roman_empire', 'minesweeper', 'questions',
                  'amazon_ratings', 'tolokers']:
        features, edge_index, labels, num_targets, splits_lst = load_critical(name)
    elif name in ['wiki_cooc', 'blogcatalog', 'flickr']:
        features, edge_index, labels, num_targets, splits_lst = load_opengsl(name)
    elif name in ['Bgp', ]:
        features, edge_index, labels, num_targets, splits_lst = load_pathnet(name)
    else:
        raise ValueError('Invalid data name')
    ##############################
    if splits_lst == []:
        # generate random split
        split_seed = 1234567
        # 10 splits
        splits_lst = []
        num_node = features.shape[0]
        for i in range(10):
            idx_train, idx_val, idx_test = random_splits_with_unlabel(
                labels, ratio=[60, 20, 20], seed=split_seed)
            split_seed += 1
            splits_lst.append({
                'train': index_to_mask(idx_train, num_node),
                'valid': index_to_mask(idx_val, num_node),
                'test': index_to_mask(idx_test, num_node)})
    #####################################
    ea = torch.ones(edge_index.shape[1])
    # formatting
    bg = BaseGraph(features, edge_index, ea, labels, num_targets)
    return bg, splits_lst
