import math
import random
import time
import numpy as np
import scipy
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from torch_scatter import scatter_add

import os
import pickle as pkl
import sys
from os import path

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import normalize as sk_normalize

if torch.cuda.is_available():
    from torch_geometric.utils import to_dense_adj, contains_self_loops, remove_self_loops, \
        to_dense_adj

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

pi = math.pi
# function for classifer-based
# https://github.com/SitaoLuan/When-Do-GNNs-Help/blob/main/utils/homophily_metrics.py
def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

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

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def gntk_homophily_(features, adj, sample, n_layers):
    eps = 1e-8
    G_gram = torch.mm(torch.spmm(adj, features)[sample, :],
                      torch.transpose(torch.spmm(adj, features)[sample, :], 0, 1))
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


def classifier_based_performance_metric(features, adj, labels, sample_max, base_classifier='kernel_reg1', epochs=100):
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
                                                       nan_policy='propagate')
    else:
        g_aware_good_stats, g_aware_good_p = ttest_ind(X_results.detach().cpu(), G_results.detach().cpu(), axis=0,
                                                       equal_var=False, nan_policy='propagate')

    if torch.mean(diff_results) <= 0.5:
        g_aware_good_p = g_aware_good_p / 2

    else:
        g_aware_good_p = 1 - g_aware_good_p / 2

    return g_aware_good_p, time.time() - t_time
