import argparse
import os

from itertools import combinations
# from typing import NamedTuple, Union

from torch_geometric.utils import to_dense_adj

from metric_function import *

# evaluate heterophily metric on synthetic data
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='PA', choices=['RG', 'PA', 'Gencat'])
    parser.add_argument('--metric', type=str, default='edge', choices=[
        'edge', 'node', 'class', 'li', 'adjust', 'ge', 'agg', 'ne', 
        'kernel_reg0', 'kernel_reg1', 'gnb'])
    # graph id: each level 10 graphs
    parser.add_argument('--graph_id', type=int, default=0)
    # preferential attachment
    parser.add_argument('--mixhop_h', type=float, default=0.0)
    # gencat
    parser.add_argument('--base_dataset_gencat', type=str, default='cora')
    parser.add_argument('--beta', type=int, default=0)
    # RG
    parser.add_argument('--num_edge_same', type=int, default=800)
    parser.add_argument('--homo_lvl', type=float, default=0.15)
    parser.add_argument('--base_dataset_rg', type=str, default='cora')
    args = parser.parse_args()
    return args

# for PA
def edge_mixhop_to_edge_list(edge_mixhop):
    adj_indices = []
    for node, neighbors in edge_mixhop.items():
        for n in neighbors:
            adj_indices.append([node, n])
    return np.transpose(adj_indices)

def load_RG(args, device):
    print('load regular syn graph')
    features = torch.tensor(preprocess_features(
        torch.load(
            f"./data_synthesis/features/{args.base_dataset_rg}/{args.base_dataset_rg}_{args.graph_id}.pt").clone().detach().float())).clone().detach()
    # Path(f"./When-Do-GNNs-Help/data_synthesis/{num_edge_same}/{homo_lvl}").mkdir(parents=True, exist_ok=True)
    adj = torch.load((
        f"./data_synthesis/{args.num_edge_same}/{args.homo_lvl}/adj_{args.homo_lvl}_{args.graph_id}.pt")).to_dense().clone().detach().float()
    label = ((torch.load((
        f"./data_synthesis/{args.num_edge_same}/{args.homo_lvl}/label_{args.homo_lvl}_{args.graph_id}.pt")).to_dense().clone().detach().float())).clone().detach()
    adj_tensor = adj.to(device)
    label_long = label.to(device) # one-hot encoding
    features = features.to(device)
    return adj_tensor, label_long, features

def load_PA(args, device):
    BASE_DIR = "./mixhop_syn-2000_5/"
    feat = torch.load(os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.allx".format(args.mixhop_h, args.graph_id)))
    label = torch.load(
        os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.ally".format(args.mixhop_h, args.graph_id)))  # one-hot label
    label = label.argmax(1)
    label = torch.LongTensor(label).to(device)
    edge_mixhop = torch.load(
        os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.graph".format(args.mixhop_h, args.graph_id)))
    # edge_mixhop = pickle.load(
    #     open(os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.graph".format(args.mixhop_h, args.graph_id)), 'rb'),
    #     encoding='latin1')
    nnodes = len(edge_mixhop)
    edge_list = edge_mixhop_to_edge_list(edge_mixhop)
    edge_index = torch.tensor(edge_list).to(device)
    adj = to_dense_adj(edge_index.cpu().to(dtype=torch.int64))[0].numpy()
    label_long = np.zeros((nnodes, int(label.max().item() + 1)))
    label_long[np.arange(nnodes), label.cpu().numpy().astype(int)] = 1
    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
    features = torch.tensor(feat, dtype=torch.float32).to(device)
    label_long = torch.tensor(label_long, dtype=torch.float32).to(device)
    return adj_tensor, label_long, features

def load_Gencat(args, device):
    BASE_DIR = "./GenCAT_Exp_hetero_homo"
    data = torch.load("{}/GenCAT_{}_{}_{}.pt".format(
        BASE_DIR, args.base_dataset_gencat, int(args.beta*10), args.graph_id))
    adj = data['adj']
    features = data['feature']
    label = data['labels']
    adj = sp.dok_matrix.toarray(adj)
    label = np.array(label)
    nnodes = len(label)
    label_long = np.zeros((nnodes, int(label.max().item() + 1)))
    label_long[np.arange(nnodes), label] = 1
    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
    features = torch.tensor(features, dtype=torch.float32).to(device)
    label_long = torch.tensor(label_long, dtype=torch.float32).to(device)
    return adj_tensor, label_long, features

def load_data(args, device):
    if args.mode == 'PA':
        adj_tensor, label_long, features = load_PA(args, device)
    elif args.mode == 'Gencat':
        adj_tensor, label_long, features = load_Gencat(args, device)
    elif args.mode == 'RG':
        adj_tensor, label_long, features = load_RG(args, device)
    else:
        raise ValueError('Invalid mode')
    return adj_tensor, label_long, features
        
def compute_metrics_on_syn_graph(args, device):
    torch.manual_seed(0)
    adj_tensor, label_long, features = load_data(args, device)
    nnodes = adj_tensor.shape[0]
    label_short = label_long.argmax(1)
    num_class = int(label_short.max().item() + 1) # number of class
    sample_max = 500
    result = None

    features = features.to(device)
    adj_tensor = adj_tensor.to(device)   

    # choose the metric
    if args.metric == 'edge':
        # edge homophily
        result = edge_homophily(adj_tensor, label_long).item()
    elif args.metric == 'node':
        # node homophily
        result = node_homophily(adj_tensor, torch.argmax(label_long, 1)).item()
    elif args.metric == 'class':
        result = our_measure(adj_tensor, torch.argmax(label_long, 1)).item()
    elif args.metric == 'li':
        # label informative
        result = label_informativeness(adj_tensor, label_long).item()
    elif args.metric == 'adjust':
        # adjust edge homophily
        result = adjusted_homo(adj_tensor, label_long).item()
    elif args.metric == 'ge':
        result = generalized_edge_homophily(adj_tensor, features, label_long).item()
    elif args.metric == 'agg':
        adj_hat_np = adj_tensor.cpu().numpy() + np.eye(nnodes)
        result = agg_h(adj_hat_np, label_long.cpu().numpy(), label_short, nnodes)
    elif args.metric == 'ne':
        result = N_ident(adj_tensor.cpu().numpy(), label_short, nnodes, num_class)
    elif args.metric == 'kernel_reg0':
        labels = torch.argmax(label_long, 1)
        labels = labels.to(device)
        result = classifier_based_performance_metric(
            features,
            adj_tensor,
            torch.argmax(label_long, 1).cpu(),
            sample_max= sample_max,
            base_classifier='kernel_reg0',
            epochs=100)   
    elif args.metric == 'kernel_reg1':
        labels = torch.argmax(label_long, 1)
        labels = labels.to(device)
        result = classifier_based_performance_metric(
            features,
            adj_tensor,
            torch.argmax(label_long, 1).cpu(),
            sample_max= sample_max,
            base_classifier='kernel_reg1',
            epochs=100)
    elif args.metric == 'gnb':
        labels = torch.argmax(label_long, 1)
        labels = labels.to(device)
        result = classifier_based_performance_metric(
            features,
            adj_tensor,
            labels,
            sample_max= sample_max,
            base_classifier='gnb',
            epochs=100)        
    else:
        raise ValueError('Invalid metric')
    return result

def compte_metrics_on_all_graph_samples_and_save(args, device, num_graph=10, save=False):
    all_results = []
    for i in range(num_graph):
        args.graph_id = i
        result = compute_metrics_on_syn_graph(args, device)
        all_results.append(result)
    if save:
        # save computed results
        if args.mode == 'PA':
            homo_level = args.mixhop_h
        elif args.mode == 'Gencat':
            homo_level = args.beta
        elif args.mode == 'RG':
            homo_level = args.homo_lvl
        save_dir = f'./metrics_results/{args.mode}/{args.metric}_{homo_level}.pt'
        torch.save(all_results, save_dir)
    return all_results

if __name__ == "__main__":
    args = get_args()
    # device = torch.device(args.device)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    # compute a certain metric for 10 graphs of a certain homophily level
    all_results = compte_metrics_on_all_graph_samples_and_save(args, device, num_graph=10, save=True)
