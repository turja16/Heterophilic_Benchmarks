from torch_geometric.utils.convert import to_networkx
from typing import List
import numpy as np
import operator
from torch.nn import CosineSimilarity
import torch
import random

def compute_cosine_similarity(dataset, edge_index, attri):
    cos = CosineSimilarity(dim=0, eps=1e-6)
    similaity_list = []
    for i in range(len(dataset['graph'])):
        if attri == "label":
            attri = dataset['graph'][i].y
        elif attri == "feature":
            attri = dataset['graph'][i].x
        for item in edge_index.transpose(1, 0):
            similarity = cos(attri[item[0]].float(), attri[item[1]].float())
            similaity_list.append(float(similarity))
    return similaity_list


def compute_parameter(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')


def compute_label_percentage(li: List):
    dict = {}
    for key in li:
        dict[key] = dict.get(key, 0) + 1
    return dict


def compute_smoothness(dataset):
    smooth_edges = 0
    G = to_networkx(dataset)
    adj_dict = dict(G.adj.items())
    for i in range(len(G.nodes)):
        if len(adj_dict[i]) == 0:
            continue
        node_labels = []
        for key in dict(adj_dict[i]):
            node_labels.append(dataset.y[key])
        precent_dict = compute_label_percentage(node_labels)
        prop_max_label = max(precent_dict.items(),
                             key=operator.itemgetter(1))[0]
        if dataset.y[i].equal(prop_max_label):
            smooth_edges += 1

    return smooth_edges / len(G.nodes)


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

# for dataset with unlabeled data
def random_splits_with_unlabel(labels, ratio: list = [60, 20, 20], seed: int = 1234567):
    labels = labels.cpu()
    y_have_label_mask = labels != -1
    total_node_num = len(labels)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    masked_index = y_index_tensor[y_have_label_mask]
    (train_index, val_index, test_index) = get_order(
        ratio, masked_index, total_node_num, seed)
    return (train_index, val_index, test_index)

def index_to_mask(num_nodes, index):
    mask = [False] * num_nodes
    mask = torch.tensor(np.array(mask))
    mask[index] = True
    return mask


def split_dataset(num_nodes, p=[0.6, 0.2, 0.2]):
    train_mask_list = [True] + [False] * (num_nodes - 1)  # at least one sample
    test_mask_list = [True] + [False] * (num_nodes - 1)
    val_mask_list = [True] + [False] * (num_nodes - 1)
    for i in range(num_nodes - 3):
        p_now = random.uniform(0, 1)
        for j in range(len(p)):
            if (p_now <= p[j]):
                if (j == 0):
                    train_mask_list[i + 1] = True
                elif (j == 1):
                    test_mask_list[i + 1] = True
                elif (j == 2):
                    val_mask_list[i + 1] = True
                break
            p_now -= p[j]
    return torch.tensor(train_mask_list), torch.tensor(test_mask_list), torch.tensor(val_mask_list)
