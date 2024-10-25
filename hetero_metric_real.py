import torch
from collections import defaultdict
from torch_geometric.utils import to_dense_adj, remove_self_loops, dense_to_sparse
from tqdm import tqdm
import argparse
import os
import numpy as np
import math
from itertools import combinations
import networkx as nx
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix
from classifer_based_utils import *
import scipy.sparse as sp
from large_scale_data_utils.dataset import load_nc_dataset
from torch_geometric.datasets import AttributedGraphDataset
from critical_look_utils.datasets import Dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()
    return args

torch.manual_seed(0)
args = get_args()
device = torch.device(args.device)

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
# opengsl
elif args.dataset in ['blogcatalog', 'flickr', 'wiki-cooc']:
    BASE_DIR = './Opengsl'
    if args.dataset == 'wiki-cooc':      
        file_name = f'{args.dataset.replace("-", "_")}.npz'
        data = np.load(os.path.join(BASE_DIR, file_name))
        feat = torch.tensor(data['node_features']) #
        label = torch.tensor(data['node_labels']) #
        edges = torch.tensor(data['edges'])
        edge_index = remove_self_loops(edges.T)[0]
    elif args.dataset in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=BASE_DIR, name=args.dataset)
        g = dataset[0]
        feat = g.x  # unnormalized #
        if args.dataset == 'flickr':
            feat = feat.to_dense()
        label = g.y #
        edge_index = remove_self_loops(g.edge_index)[0]
    else:
        raise ValueError('dataset does not exist')
# pathnet
elif args.dataset in ['Bgp']:
    BASE_DIR = "./PathNet/other_data"
    x = np.load(BASE_DIR + '/' + args.dataset + '/x.npy')
    y = np.load(BASE_DIR + '/' + args.dataset + '/y.npy')
    # might have unlabeled data
    label = torch.tensor(y)
    feat = torch.tensor(x)
    numpy_edge_index = np.load(BASE_DIR+'/'+ args.dataset+'/edge_index.npy')
    edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)
    edge_index = remove_self_loops(edge_index)[0]
# critical look
elif args.dataset in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'squirrel-filtered', 'chameleon-filtered']:
    dataset = Dataset(name=args.dataset,
        add_self_loops=False,
        device=device,
        use_sgc_features=False,
        use_identity_features=False,
        use_adjacency_features=False,
        do_not_use_original_features=False)
    graph = dataset.graph
    feat = dataset.node_features
    label = dataset.labels
    edge_index = graph.adj().coalesce().indices()
    edge_index = remove_self_loops(edge_index)[0]

nnodes = feat.shape[0]
C = len(label.unique())
num_edge = int(edge_index.shape[1]/2)
num_fea = num_fea = feat.shape[-1]
# adj = to_dense_adj(edge_index.to(dtype=torch.int64))[0].numpy()
adj = to_scipy_sparse_matrix(edge_index)

# H edge
h_edge = 0
for i in range(edge_index.shape[1]):
    src, dst = edge_index[:, i]
    if label[src] == label[dst]:
        h_edge += 1

h_edge /= edge_index.shape[1]
#############

# # h node
# h_node = 0
# for v in range(nnodes):
#     h_v = 0
#     # N_v = adj[v, :]
#     N_v = (adj.getrow(v)).toarray()[0]
#     d_v = N_v.sum()
#     if d_v !=0:
#         u_list = N_v.nonzero()[0]
#         for u in u_list:
#             if label[v] == label[u]:
#                 h_v += 1
#         h_v = h_v / d_v
#         h_node = h_node + h_v

# h_node = h_node / nnodes

print('check sym: {}'.format((adj - adj.T).sum() == 0))
# print(f'dataset {args.dataset}; num node {nnodes}; num edge {num_edge}; num feature {num_fea}; classes {C}; H edge {h_edge: .4f}; H node {h_node: .4f}')
print(f'dataset {args.dataset}; num node {nnodes}; num edge {num_edge}; num feature {num_fea}; classes {C}; H edge {h_edge: .4f}')