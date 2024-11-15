import argparse
from itertools import combinations
from typing import NamedTuple, Union

from torch_geometric.utils import to_dense_adj

from classifer_based_utils import *


# evaluate heterophily metric on synthetic data
def get_args():
    parser = argparse.ArgumentParser()
    # preferential attachment
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mixhop_h', type=float, default=None)
    parser.add_argument('--mixhop_id', type=int, default=None)
    # gencat
    parser.add_argument('--base_dataset', type=str, default=None)
    parser.add_argument('--beta', type=int, default=None)
    parser.add_argument('--gen_id', type=int, default=None)
    return args


def normalize_tensor(mx, symmetric=0):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, 1)
    if symmetric == 0:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx
    else:
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(torch.mm(r_mat_inv, mx), r_mat_inv)
        return mx


# x = dataset.node_features.cpu().numpy()
def agg_h(adj_hat, z, label, nnodes):
    agg = adj_hat @ z
    post_agg = agg @ agg.T
    s_agg = 0
    for v in range(nnodes):
        c_v = label[v].item()
        node_same = np.argwhere(label.cpu().numpy() == c_v).flatten()
        node_diff = np.argwhere(label.cpu().numpy() != c_v).flatten()
        check_v = post_agg[v, node_same].mean() >= post_agg[v, node_diff].mean()
        s_agg += int(check_v)
    s_agg = s_agg / nnodes
    h_m_agg = max(2 * s_agg - 1, 0)
    return h_m_agg


def edge_mixhop_to_edge_list(edge_mixhop):
    adj_indices = []
    for node, neighbors in edge_mixhop.items():
        for n in neighbors:
            adj_indices.append([node, n])
    return np.transpose(adj_indices)


def compute_metrics_on_syn_graph(device: torch.device,
                                 args: Union[NamedTuple, argparse.Namespace]):
    torch.manual_seed(0)
    if args.mixhop_h is not None and args.mixhop_id is not None:
        print('load mixhop syn data (preferential attachment)')
        BASE_DIR = "./mixhop_syn-2000_5/"

        feat = torch.load(os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.allx".format(args.mixhop_h, args.mixhop_id)))
        label = torch.load(
            os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.ally".format(args.mixhop_h, args.mixhop_id)))  # one-hot label
        label = label.argmax(1)
        label = torch.LongTensor(label).to(device)
        edge_mixhop = torch.load(
            os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.graph".format(args.mixhop_h, args.mixhop_id)))
        # edge_mixhop = pickle.load(
        #     open(os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.graph".format(args.mixhop_h, args.mixhop_id)), 'rb'),
        #     encoding='latin1')
        nnodes = len(edge_mixhop)
        edge_list = edge_mixhop_to_edge_list(edge_mixhop)
        edge_index = torch.tensor(edge_list).to(device)
        adj = to_dense_adj(edge_index.cpu().to(dtype=torch.int64))[0].numpy()
        num_fea = feat.shape[-1]
        num_edge = int(edge_index.shape[1] / 2)
        C = int(label.max().item() + 1)
        D = adj @ np.array([1] * nnodes)
        adj_hat = adj + np.eye(nnodes)
        z = np.zeros((nnodes, C))
        z[np.arange(nnodes), label.cpu().numpy().astype(int)] = 1
        feat = torch.tensor(
            feat, dtype=torch.float32, device=device
        )
        print('check sym: {}'.format((adj - adj.T).sum() == 0))
    else:
        BASE_DIR = "./GenCAT_Exp_hetero_homo"
        data = torch.load("{}/GenCAT_{}_{}_{}.pt".format(
            BASE_DIR, args.base_dataset, args.beta, args.gen_id))
        adj = data['adj']
        feature = data['feature']
        label = data['labels']
        adj = sp.dok_matrix.toarray(adj)
        label = torch.LongTensor(label).to(device)
        nnodes = len(label)
        edge_index = np.nonzero(torch.tensor(adj)).T.to(device)
        D = adj @ np.array([1] * nnodes)
        adj_hat = adj + np.eye(nnodes)
        num_edge = int(edge_index.shape[1] / 2)
        C = int(label.max().item() + 1)
        z = np.zeros((nnodes, C))
        z[np.arange(nnodes), label.cpu().numpy().astype(int)] = 1
        feat = torch.tensor(
            feature, dtype=torch.float32, device=device
        )
        num_fea = feat.shape[-1]
        print('check sym: {}'.format((adj - adj.T).sum() == 0))

    # apply normalization
    feat_norm = normalize_tensor(feat)
    adj_hat_norm_rw = normalize_tensor(torch.tensor(adj_hat, dtype=torch.float32, device=device))
    adj_hat_norm_sym = normalize_tensor(torch.tensor(adj_hat, dtype=torch.float32, device=device), symmetric=1)

    # H edge
    h_edge = 0
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        if label[src] == label[dst]:
            h_edge += 1

    h_edge /= edge_index.shape[1]
    #############

    # h node
    h_node = 0
    for v in range(nnodes):
        h_v = 0
        N_v = adj[v, :]
        d_v = N_v.sum()
        if d_v != 0:
            u_list = N_v.nonzero()[0]
            for u in u_list:
                if label[v] == label[u]:
                    h_v += 1
            h_v = h_v / d_v
            h_node = h_node + h_v

    h_node = h_node / nnodes
    #############

    # h class
    h_c = 0
    for c in range(C):
        # find nodes belong to class c
        node_c = np.argwhere(label.cpu().numpy() == c).flatten()
        h_k_numerator = 0
        h_k_denominator = 0
        for v in node_c:
            N_v = adj[v, :]
            h_k_denominator += N_v.sum()
            u_list = N_v.nonzero()[0]
            u_count = 0
            # count the number of node in N_v with same label as v
            for u in u_list:
                if label[u] == label[v]:
                    u_count += 1
            h_k_numerator += u_count
        h_k = h_k_numerator / h_k_denominator
        h_c += max(h_k - len(node_c) / nnodes, 0)

    h_c = h_c / (C - 1)

    #############

    # h agg mean
    h_m_agg = agg_h(adj_hat, z, label, nnodes)
    h_m_agg_norm_rw = agg_h(adj_hat_norm_rw.cpu().numpy(), z, label, nnodes)
    h_m_agg_norm_sym = agg_h(adj_hat_norm_sym.cpu().numpy(), z, label, nnodes)

    # adjusted H edge
    h_adj_c = 0
    for c in range(C):
        node_c = np.argwhere(label.cpu().numpy() == c).flatten()
        Dk = D[node_c].sum()
        h_adj_c += (Dk) ** 2 / (2 * num_edge) ** 2

    h_edge_adj = (h_edge - h_adj_c) / (1 - h_adj_c)

    # label informativeness
    LI_denominator = 0
    dst = []
    for c in range(C):
        node_c = np.argwhere(label.cpu().numpy() == c).flatten()
        Dk = D[node_c].sum()
        pk = Dk / (2 * num_edge)
        LI_denominator += pk * math.log(pk)
        dst.append(pk)  # should be a distribution

    c_list = [i for i in range(C)]
    combinations_list = list(combinations(c_list, 2))
    numerator = 0
    count_c1_c2 = {}
    for c1, c2 in combinations_list:
        count_c1_c2[(c1, c2)] = 0
        count_c1_c2[(c2, c1)] = 0

    for c in range(C):
        count_c1_c2[(c, c)] = 0

    for i in range(num_edge * 2):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        u_c = label[u].item()
        v_c = label[v].item()
        count_c1_c2[(u_c, v_c)] += 1

    LI_numerator = 0
    dst = []
    for c1_c2, p in count_c1_c2.items():
        p_c1_c2 = p / (2 * num_edge)
        dst.append(p_c1_c2)
        if p != 0:
            LI_numerator += p_c1_c2 * math.log(p_c1_c2)

    LI = 2 - LI_numerator / LI_denominator

    # generalized edge homophily
    ge = 0
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        xi = feat_norm[src]
        xj = feat_norm[dst]
        if torch.linalg.vector_norm(xi, ord=2).item() == 0 or torch.linalg.vector_norm(xj, ord=2).item() == 0:
            cos = 0
        else:
            cos = (xi @ xj) / (torch.linalg.vector_norm(xi, ord=2) * torch.linalg.vector_norm(xj, ord=2))
        ge = ge + cos

    ge = (ge / edge_index.shape[1]).item()

    # neighborhood identifiability
    ne = 0
    node_id = np.arange(nnodes)
    for k in range(C):
        # each class k
        node_class_k = node_id[(label == k).cpu().numpy()]
        ne_dist = []
        # iter through every nodes in class k
        # compute neighborhood label dist
        for v in node_class_k:
            N_v = adj[v, :]
            d_v = N_v.sum()
            if d_v != 0:
                u_list = N_v.nonzero()[0]
                class_dist = [0] * C
                for u in u_list:
                    u_class = label[u].item()
                    class_dist[u_class] += 1
                class_dist = np.array(class_dist)
                class_dist = 1.0 / len(u_list) * class_dist
                ne_dist.append(list(class_dist))
            else:
                ne_dist.append([0] * C)
        n_k = len(node_class_k)
        Ak = np.array(ne_dist)
        U, S, V = np.linalg.svd(Ak)
        S = S[S != 0]
        S = S / S.sum()
        log_S = np.log(S)
        ne_k = -1.0 * (S * log_S).sum() / math.log(C)
        ne = ne + (n_k / nnodes) * ne_k

    # classifier-based
    ## rw norm
    kernel_reg0_norm_rw, time = classifier_based_performance_metric(
        feat_norm, adj_hat_norm_rw, label,
        sample_max=500,
        base_classifier="kernel_reg0",
        epochs=100)

    kernel_reg1_norm_rw, time = classifier_based_performance_metric(
        feat_norm, adj_hat_norm_rw, label,
        sample_max=500,
        base_classifier="kernel_reg1",
        epochs=100)

    gnb_norm_rw, time = classifier_based_performance_metric(
        feat_norm, adj_hat_norm_rw, label,
        sample_max=500,
        base_classifier="gnb",
        epochs=100)

    ## sym norm
    kernel_reg0_norm_sym, time = classifier_based_performance_metric(
        feat_norm, adj_hat_norm_sym, label,
        sample_max=500,
        base_classifier="kernel_reg0",
        epochs=100)

    kernel_reg1_norm_sym, time = classifier_based_performance_metric(
        feat_norm, adj_hat_norm_sym, label,
        sample_max=500,
        base_classifier="kernel_reg1",
        epochs=100)

    gnb_norm_sym, time = classifier_based_performance_metric(
        feat_norm, adj_hat_norm_sym, label,
        sample_max=500,
        base_classifier="gnb",
        epochs=100)

    if args.mixhop_h is not None:
        name = 'mixhop_h{}_g{}'.format(args.mixhop_h, args.mixhop_id)
    elif args.base_dataset is not None:
        name = 'gencat_{}_{}_{}'.format(args.base_dataset, args.beta, args.gen_id)

    print('dataset {}; classes {}; num edge {}; num feature {}; num node {}'.format(
        name, C, num_edge, num_fea, nnodes))

    print(f'H node {h_node:.2f}; H edge {h_edge:.2f}; H class {h_c:.2f}')

    print(f'H_m_AGG {h_m_agg:.2f}')

    print(f'H_m_AGG (rw) {h_m_agg_norm_rw:.2f}')

    print(f'H_m_AGG (sym) {h_m_agg_norm_sym:.2f}')

    print(f'adjust H edge {h_edge_adj:.2f}; label informativeness {LI:.2f}')

    print(f'generalized edge homophily {ge:.2f}; neighborhood identifiability {ne:.2f}')

    print(f'(rw) kernel_reg0 {kernel_reg0_norm_rw:.2f}; kernel_reg1 {kernel_reg1_norm_rw:.2f}; gnb {gnb_norm_rw:.2f}')

    print(
        f'(sym) kernel_reg0 {kernel_reg0_norm_sym:.2f}; kernel_reg1 {kernel_reg1_norm_sym:.2f}; gnb {gnb_norm_sym:.2f}')

    res = {'name': name,
           'classes': C,
           'num_edge': num_edge,
           'num_feat': num_fea,
           'num_node': nnodes,
           'H node': h_node,
           'H edge': h_edge,
           'H class': h_c,
           'H_ge': ge,
           'H_m_AGG': h_m_agg,
           'H_m_AGG_rw': h_m_agg_norm_rw,
           'H_m_AGG_sym': h_m_agg_norm_sym,
           'H_edge_adj': h_edge_adj,
           'LI': LI,
           'H_nei': ne,
           'kernel_reg0_rw': kernel_reg0_norm_rw,
           'kernel_reg1_rw': kernel_reg1_norm_rw,
           'gnb_rw': gnb_norm_rw,
           'kernel_reg0_sym': kernel_reg0_norm_sym,
           'kernel_reg1_sym': kernel_reg1_norm_sym,
           'gnb_sym': gnb_norm_sym
           }

    torch.save(res, './stat/{}.pt'.format(name))
    return res


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)

    compute_metrics_on_syn_graph(device, args)
