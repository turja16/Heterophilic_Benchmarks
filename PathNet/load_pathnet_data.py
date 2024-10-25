import torch 
import numpy as np

def get_whole_mask(y, ratio: list = [48, 32, 20], seed: int = 1234567):
    '''
    work for "load_data", random_spilt at [48, 32, 20] ratio
    '''
    y_have_label_mask = y != -1
    total_node_num = len(y)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    masked_index = y_index_tensor[y_have_label_mask]
    while True:
        (train_mask, val_mask, test_mask) = get_order(
            ratio, masked_index, total_node_num, seed)
        # if check_train_containing(train_mask,y):
        return (train_mask, val_mask, test_mask)
        # else:
        #     seed+=1


def load_data(dataset_name, round, data_root="./other_data"):
    '''
    Load data for Nba, Electronics, Bgp
    '''
    numpy_x = np.load(data_root + '/' + dataset_name + '/x.npy')
    x = torch.from_numpy(numpy_x).to(torch.float)
    numpy_y = np.load(data_root + '/' + dataset_name + '/y.npy')
    y = torch.from_numpy(numpy_y).to(torch.long)
    numpy_edge_index = np.load(data_root+'/'+dataset_name+'/edge_index.npy')
    edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)

    # (train_mask, val_mask, test_mask) = get_whole_mask(y, seed=round + 1)

    lbl_set = []
    for lbl in y:
        if lbl not in lbl_set:
            lbl_set.append(lbl)
    num_classes = len(lbl_set)

    return x, y, num_classes, train_mask, val_mask, test_mask