import torch
import numpy as np
import scipy.sparse as sp
import scipy
import random
import math
import time

from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
pi = math.pi   

def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# preprocess_features from When-Do-GNNs-Help.utils.util_funcs
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = (1 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

############################################
# When-Do-GNNs-Help.utils.homophily_plot
############################################

def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr

def edge_homophily(adj, label):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    """
    adj = (adj > 0).float()
    adj = adj - torch.diag(torch.diag(adj))
    label_adj = torch.mm(label, label.transpose(0, 1))
    edge_hom = torch.sum(label_adj * adj) / torch.sum(adj)
    return edge_hom

def node_homophily_edge_idx(edge_index, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    hs = torch.zeros(num_nodes).to(device)
    degs = torch.bincount(edge_index[0, :]).float().to(device)
    matches = (labels[edge_index[0, :]] == labels[edge_index[1, :]]).float().to(device)
    hs = hs.scatter_add(0, edge_index[0, :], matches) / degs
    return hs[degs != 0].mean()

def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node, targ_node = A.nonzero()[:, 0], A.nonzero()[:, 1]
    edge_idx = torch.tensor(np.vstack((src_node.cpu(), targ_node.cpu())), dtype=torch.long).contiguous().to(device)
    labels = labels.clone().detach()
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)

def our_measure(A, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    A = A - torch.diag(torch.diag(A))
    A = A + torch.diag((torch.sum(A, 1) == 0).float())
    edge_index = A.nonzero()
    label = label.squeeze()
    c = label.max() + 1
    H = compact_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c - 1
    return val


def compact_matrix_edge_idx(edge_index, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    src_node, targ_node = edge_index[:, 0], edge_index[:, 1]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max() + 1
    H = torch.zeros((c, c)).to(device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        H[k, :].scatter_add_(src=torch.ones_like(add_idx).to(H.dtype), dim=-1, index=add_idx)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H


def class_distribution(A, labels):
    edge_index = A.to_sparse().coalesce().indices()
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    deg = torch.sum(A, dim=0)
    # replace the following code as it does not consider the case where degi=0
    # here we all assume the A does not contain self-loop
    # deg = src_node.unique(return_counts=True)[1]
    # deg = deg.to(device)
    # remove self-loop
    # deg = deg - 1
    # edge_index = remove_self_loops(A.to_sparse().coalesce().indices())[0]
    # src_node, targ_node = edge_index[0, :], edge_index[1, :]
    labels = labels.squeeze()
    p = labels.unique(return_counts=True)[1] / labels.shape[0]
    p_bar = torch.zeros(labels.max() + 1, device=labels.device)
    pc = torch.zeros((labels.max() + 1, labels.max() + 1), device=labels.device)
    for i in range(labels.max() + 1):
        p_bar[i] = torch.sum(deg[torch.where(labels == i)])
        for j in range(labels.max() + 1):
            pc[i, j] = torch.sum(labels[targ_node[torch.where(labels[src_node] == i)]] == j)
    p_bar, pc = p_bar / torch.sum(deg), pc / torch.sum(deg)
    p_bar[torch.where(p_bar == 0)], pc[torch.where(pc == 0)] = 1e-8, 1e-8
    return p, p_bar, pc

def label_informativeness(A, label):
    p, p_bar, pc = class_distribution(A, torch.argmax(label, 1))
    LI = 2 - torch.sum(pc * torch.log(pc)) / torch.sum(p_bar * torch.log(p_bar))
    return LI


def adjusted_homo(A, label):
    p, p_bar, pc = class_distribution(A, torch.argmax(label, 1))
    edge_homo = edge_homophily(A, label)
    adj_homo = (edge_homo - torch.sum(p_bar ** 2)) / (1 - torch.sum(p_bar ** 2))
    return adj_homo


def generalized_edge_homophily(adj, features, label, sample_max=20000, iteration=100):
    adj = (adj > 0).float()
    nnodes = label.shape[0]
    if nnodes < sample_max:
        sim = torch.tensor(cosine_similarity(features.cpu(), features.cpu())).to(device)
        sim[torch.isnan(sim)] = 0
        adj = adj - torch.diag(torch.diag(adj))
        g_edge_homo = torch.sum(sim * adj) / torch.sum(adj)
        return g_edge_homo
    else:
        edge_homo = np.zeros(iteration)
        for i in range(iteration):
            val_sample = torch.tensor(random.sample(list(np.arange(nnodes)), int(sample_max)))
            sim = torch.tensor(cosine_similarity(features[val_sample].cpu(), features[val_sample].cpu())).to(device)
            sim[torch.isnan(sim)] = 0
            adj = adj.to_dense()[val_sample, :][:, val_sample]
            adj = adj - torch.diag(torch.diag(adj))
            edge_homo[i] = torch.sum(sim * adj) / torch.sum(adj)
        return np.mean(edge_homo)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_disassortative_splits(labels, num_classes, training_percentage=0.6):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(training_percentage * (labels.size()[0] / num_classes)))
    val_lb = int(round(0.2 * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])
    return train_mask.to(device), val_mask.to(device), test_mask.to(device)

 
def gntk_homophily_(features, adj, sample, n_layers):
    if features.device != sample.device:
        features = features.to(sample.device)
    if adj.device != sample.device:
        adj = adj.to(sample.device)
        
    eps = 1e-8
    nnodes = features.shape[0]
    G_gram = torch.mm(torch.spmm(adj, features)[sample, :], torch.transpose(torch.spmm(adj, features)[sample, :], 0, 1))
    G_norm = torch.sqrt(torch.diag(G_gram)).reshape(-1, 1) * torch.sqrt(torch.diag(G_gram)).reshape(1, -1)
    G_norm = (G_norm > eps) * G_norm + eps * (G_norm <= eps)
    if n_layers == 1:
        arccos = torch.acos(torch.div(G_gram, G_norm))
        sqrt = torch.sqrt(torch.square(G_norm) - torch.square(G_gram))
        arccos[torch.isnan(arccos)], sqrt[torch.isnan(sqrt)] = 0, 0
        K_G = 1 / pi * (G_gram * (pi - arccos) + sqrt)
    else:
        K_G = G_gram

    gram = torch.mm(features[sample, :], torch.transpose(features[sample, :], 0, 1))
    norm = torch.sqrt(torch.diag(gram)).reshape(-1, 1) * torch.sqrt(torch.diag(gram)).reshape(1, -1)
    norm = (norm > eps) * norm + eps * (norm <= eps)
    if n_layers == 1:
        arccos = torch.acos(torch.div(gram, norm))
        sqrt = torch.sqrt(torch.square(norm) - torch.square(gram))
        arccos[torch.isnan(arccos)], sqrt[torch.isnan(sqrt)] = 0, 0
        K_X = 1 / pi * (gram * (pi - arccos) + sqrt)
    else:
        K_X = gram

    return K_G / 2, K_X / 2


def classifier_based_performance_metric(features, adj, labels, sample_max, rcond=1e-15, base_classifier='kernel_reg1',
                                        epochs=100):
    if labels.device != device:
        labels = labels.to(device)

    nnodes = (labels.shape[0])
    if labels.dim() > 1:
        labels = labels.flatten()

    G_results, X_results, diff_results, G_good_p_results, X_good_p_results = torch.zeros(epochs), torch.zeros(
        epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs)
    t_time = time.time()
    for j in range(epochs):
        if nnodes <= sample_max:
            sample = np.arange(nnodes)
            label_onehot = torch.eye(labels.max() + 1, device=device)[labels].cpu()
            labels_sample = labels
        else:
            sample, _, _ = random_disassortative_splits(labels, labels.max() + 1, sample_max / nnodes)
            label_onehot = torch.eye(labels.max() + 1, device=device)[labels][sample, :].cpu()
            labels_sample = labels[sample]
        
        idx_train, idx_val, idx_test = random_disassortative_splits(labels_sample, labels_sample.max() + 1)
        idx_val = idx_val + idx_test
        # Kernel Regression based p-values
        if base_classifier in {'kernel_reg0', 'kernel_reg1'}:
            nlayers = 0 if base_classifier == 'kernel_reg0' else 1
            K_graph, K = gntk_homophily_(features, adj, sample, nlayers)
            K_graph_train_train, K_train_train = K_graph[idx_train, :][:, idx_train], K[idx_train, :][:, idx_train]
            K_graph_val_train, K_val_train = K_graph[idx_val, :][:, idx_train], K[idx_val, :][:, idx_train]
            Kreg_G, Kreg_X = K_graph_val_train.cpu() @ (
                    torch.tensor(np.linalg.pinv(K_graph_train_train.cpu().numpy())) @ label_onehot[
                idx_train.to(label_onehot.device)]), K_val_train.cpu() @ (
                                     torch.tensor(np.linalg.pinv(K_train_train.cpu().numpy())) @ label_onehot.cpu()[
                                 idx_train.to(label_onehot.device)])

            diff_results[j] = (accuracy(labels_sample[idx_val], Kreg_G) > accuracy(labels_sample[idx_val], Kreg_X))
            G_results[j] = accuracy(labels_sample[idx_val], Kreg_G)
            X_results[j] = accuracy(labels_sample[idx_val], Kreg_X)
        elif base_classifier == 'gnb':
            #  Gaussian Naive Bayes model
            X = features[sample].cpu()
            X_agg = torch.spmm(adj, features)[sample].cpu()

            X_gnb, G_gnb = GaussianNB(), GaussianNB()
            X_gnb.fit(X[idx_train.cpu()], labels_sample[idx_train].cpu().numpy())
            G_gnb.fit(X_agg[idx_train.cpu()], labels_sample[idx_train].cpu().numpy())

            X_pred = torch.tensor(X_gnb.predict(X[idx_val.cpu()]))
            G_pred = torch.tensor(G_gnb.predict(X_agg[idx_val.cpu()]))

            diff_results[j] = (torch.mean(G_pred.eq(labels_sample[idx_val].cpu()).float()) > torch.mean(
                X_pred.eq(labels_sample[idx_val].cpu()).float()))
            # G_results[j] = torch.mean(G_pred.eq(labels_sample[idx_val]).float())
            G_results[j] = torch.mean(G_pred.eq(labels_sample[idx_val].to(G_pred.device)).float())
            X_results[j] = torch.mean(X_pred.eq(labels_sample[idx_val].to(G_pred.device)).float())
        else:
            #  SVM based p-values
            X = features[sample].cpu()
            X_agg = torch.spmm(adj, features)[sample].cpu()
            if base_classifier == 'svm_rbf':
                G_svm = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X[idx_train], labels_sample[idx_train])
            elif base_classifier == 'svm_poly':
                G_svm = svm.SVC(kernel='poly', degree=3, C=1).fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='poly', degree=3, C=1).fit(X[idx_train], labels_sample[idx_train])
            elif base_classifier == 'svm_linear':
                G_svm = svm.SVC(kernel='linear').fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='linear').fit(X[idx_train], labels_sample[idx_train])

            G_pred = torch.tensor(G_svm.predict(X_agg[idx_val]))
            X_pred = torch.tensor(X_svm.predict(X[idx_val]))
            diff_results[j] = (torch.mean(G_pred.eq(labels_sample[idx_val]).float()) > torch.mean(
                X_pred.eq(labels_sample[idx_val]).float()))
            G_results[j] = torch.mean(G_pred.eq(labels_sample[
                                                    idx_val]).float())
            X_results[j] = torch.mean(X_pred.eq(labels_sample[
                                                    idx_val]).float())

    if scipy.__version__ == '1.4.1':
        g_aware_good_stats, g_aware_good_p = ttest_ind(X_results.detach().cpu(), G_results.detach().cpu(), axis=0,
                                                       equal_var=False,
                                                       nan_policy='propagate')  # ttest_1samp(diff_results,0.5)
    else:
        g_aware_good_stats, g_aware_good_p = ttest_ind(X_results.detach().cpu(), G_results.detach().cpu(), axis=0,
                                                       equal_var=False, nan_policy='propagate')

    if torch.mean(diff_results) <= 0.5:
        g_aware_good_p = g_aware_good_p / 2

    else:
        g_aware_good_p = 1 - g_aware_good_p / 2

    return g_aware_good_p

    
### more functions

# post aggregation based
def agg_h(adj_hat_np, z, label_short, nnodes):
    # ACM-GCN
    agg = adj_hat_np @ z
    post_agg = agg @ agg.T
    s_agg = 0
    for v in range(nnodes):
        c_v = label_short[v].item()
        node_same = np.argwhere(label_short.cpu().numpy() == c_v).flatten()
        node_diff = np.argwhere(label_short.cpu().numpy() != c_v).flatten()
        check_v = post_agg[v, node_same].mean() >= post_agg[v, node_diff].mean()
        s_agg += int(check_v)
    s_agg = s_agg / nnodes
    h_m_agg = max(2 * s_agg - 1, 0)
    return h_m_agg

# neighbourhood identificability
def N_ident(adj, label, nnodes, C):
# adj: array; label: 1D tensor, nnodes: #nodes; C: #classes
    ne = 0
    node_id = np.arange(nnodes)
    for k in range(C): # each class k
        # nodes id that belong to class k
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
                # get the dist of v's neighbours' label
                for u in u_list:
                    u_class = label[u].item()
                    class_dist[u_class] += 1
                class_dist = np.array(class_dist)
                class_dist = 1.0 / len(u_list) * class_dist
                ne_dist.append(list(class_dist))
            else:
                ne_dist.append([0] * C)
        n_k = len(node_class_k)
        Ak = np.array(ne_dist) # n_k by C
        _, sv, __ = np.linalg.svd(Ak)
        sv = sv[sv != 0]
        sv = sv / sv.sum() # normalized singular value of Ak
        log_sv = np.log(sv)
        ne_k = -1.0 * (sv * log_sv).sum() / math.log(C)
        ne = ne + (n_k / nnodes) * ne_k
    return ne
