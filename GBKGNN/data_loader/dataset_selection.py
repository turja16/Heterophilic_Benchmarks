import torch
import numpy as np
from os import path
import scipy.sparse as sp

from torch_geometric.data import Data
# from torch_geometric.datasets import Actor, WikipediaNetwork, WebKB

# from utils.transform import zero_in_degree_removal
from GBKGNN.utils.statistic import compute_smoothness #, split_dataset

import sys
sys.path.append("/home/xsslnc/scratch/hetero_metric_win_ver")
from large_scale_data_utils.dataset import load_nc_dataset
from torch_geometric.utils import to_dense_adj, remove_self_loops, dense_to_sparse

import os
from torch_geometric.datasets import AttributedGraphDataset

# critical look
DATASET_LIST = [
    'squirrel_filtered', 'chameleon_filtered',
    'roman_empire', 'minesweeper', 'questions', 'amazon_ratings', 'tolokers'
]

# geom and large scale
GEOMDATASET_LIST = [
    'deezer-europe', 'genius', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 
    'twitch-gamers', 'Cora', 'CiteSeer', 'PubMed', 'chameleon', 'cornell', 'film', 
    'squirrel', 'texas', 'wisconsin'    
]

OPENGSL_LIST = [
    'wiki_cooc', 'blogcatalog', 'flickr'
]

PATHNET_LIST = [
    'Bgp'
]

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

def load_custom_data(data_path, to_undirected: bool = True):
    npz_data = np.load(data_path)
    # convert graph to bidirectional
    if to_undirected:
        edges = np.concatenate((npz_data['edges'], npz_data['edges'][:, ::-1]), axis=0)
    else:
        edges = npz_data['edges']
    feat = torch.from_numpy(npz_data['node_features'])
    features = normalize_tensor_sparse(feat, symmetric=0)
    features = torch.FloatTensor(features)
    data = Data(
        #x=torch.from_numpy(npz_data['node_features']),
        x=features,
        y=torch.from_numpy(npz_data['node_labels']),
        edge_index=torch.from_numpy(edges).T,
        # train_mask=torch.from_numpy(npz_data['train_masks']).T, # n by 10
        # val_mask=torch.from_numpy(npz_data['val_masks']).T,
        # test_mask=torch.from_numpy(npz_data['test_masks']).T,
    )
    del npz_data
    return [data]

def load_geom_data(dataname):
    # large scale + geom
    if dataname in ['deezer-europe', 'genius', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamers', 'Cora', 'CiteSeer', 'PubMed', 'chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        if dataname == 'penn94':
            dataname = 'fb100'
            sub_dataname = 'Penn94'
        else:
            dataname = dataname
            sub_dataname = ''
        dataset = load_nc_dataset(dataname, sub_dataname)
        edge_index, feat, label = dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.label
        edge_index = remove_self_loops(edge_index)[0]
    else:
        raise ValueError('Invalid data name')
    #
    features = normalize_tensor_sparse(feat, symmetric=0)
    features = torch.FloatTensor(features)
    data = Data(
        x=features,
        y=label,
        edge_index=edge_index,
    )
    return [data]

def load_opengsl_data(dataname):
    # load opengsl data
    BASE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../Opengsl"
    if dataname == 'wiki_cooc':
        # load
        file_name = f'{dataname.replace("-", "_")}.npz'
        data = np.load(os.path.join(BASE_DIR, file_name))
        feats = torch.tensor(data['node_features']) #
        labels = torch.tensor(data['node_labels']) #
        edges = torch.tensor(data['edges'])
        edge_index = remove_self_loops(edges.T)[0]
        del edges, data
        # normalize
        features = normalize_tensor_sparse(feats, symmetric=0)
        features = torch.FloatTensor(features)
        data = Data(
            x=features,
            y=labels,
            edge_index=edge_index
        )
    elif dataname in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=BASE_DIR, name=dataname)
        g = dataset[0]
        feats = g.x  # unnormalized #
        if dataname == 'flickr':
            feats = feats.to_dense()
        labels = g.y #
        edge_index = remove_self_loops(g.edge_index)[0]
        # normalize
        features = normalize_tensor_sparse(feats, symmetric=0)
        features = torch.FloatTensor(features)        
        data = Data(
            x=features,
            y=labels,
            edge_index=edge_index
        )
        del dataset
    else:
        raise ValueError('dataset does not exist')
    return [data]

def load_pathnet_data(dataname):
    # load pathnet data
    BASE_DIR = f"{path.dirname(path.abspath(__file__))}/../../PathNet/other_data"
    x = np.load(BASE_DIR + '/' + dataname + '/x.npy')
    y = np.load(BASE_DIR + '/' + dataname + '/y.npy')
    numpy_edge_index = np.load(BASE_DIR+'/'+ dataname +'/edge_index.npy')
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
    data = Data(
        x=features,
        y=labels,
        edge_index=edge_index
    )
    return [data]

# add data loading here
def get_dataset(dataset):
    if dataset in DATASET_LIST:
        print('load critical look dataset')
        return load_custom_data(
            f'../critical_look_utils/data/{dataset}.npz', 
            to_undirected='directed' not in dataset
        )
    elif dataset in GEOMDATASET_LIST:
        print('load geom dataset')
        return load_geom_data(dataset)
    elif dataset in OPENGSL_LIST:
        print('load opengsl dataset')
        return load_opengsl_data(dataset)
    elif dataset in PATHNET_LIST:
        return load_pathnet_data(dataset)
    raise ValueError("Unknown dataset")

class DatasetSelection:
    def __init__(
        self, 
        dataset_name, 
        # split, 
        task_type="NodeClasification",
        # remove_zero_degree_nodes: bool = False
    ):
        task2str = {
            "NodeClasification": "node_",
            "EdgeClasification": "edge_",
            "GraphClasification": "graph_"
        }
     
        dataset = get_dataset(dataset_name)      

        self.dataset = {"graph": []}
        smoothness = num_class = num_node = num_edge = 0
        for i in range(len(dataset)):
            num_node += dataset[i].x.shape[0]
            num_edge += dataset[i].edge_index.shape[1]
            if (dataset[i].y.shape == torch.Size([1]) and task_type == "NodeClasification"):
                dataset[i].y.data = dataset[i].x.argmax(dim=1)
                num_class = max(dataset[i].x.shape[1], num_class)
            else:
                if (len(dataset[i].y.shape) != 1):
                    num_class = max(dataset[i].y.shape[1], num_class)
                    dataset[i].y.data = dataset[i].y.argmax(dim=1)
                else:
                    num_class = max(max(dataset[i].y + 1), num_class)
            # if not hasattr(dataset[i], 'train_mask'):
            #     data_tmp = dataset[i]
            #     data_tmp.train_mask, data_tmp.test_mask, data_tmp.val_mask = split_dataset(
            #         dataset[i].x.shape[0])
            #     self.dataset["graph"].append(data_tmp)
                
            self.dataset["graph"].append(dataset[i])
            smoothness += compute_smoothness(dataset[i]) * \
                dataset[i].x.shape[0]


        if (type(num_class) != type(1)):
            num_class = num_class.numpy()

        smoothness /= num_node
        self.dataset['num_node'] = num_node
        self.dataset['num_edge'] = num_edge
        self.dataset['num_node_features'] = dataset[0].x.shape[1]
        self.dataset['smoothness'] = smoothness
        self.dataset['num_' + task2str[task_type] + 'classes'] = num_class
        self.dataset['num_classes'] = num_class

    def get_dataset(self):
        return self.dataset
