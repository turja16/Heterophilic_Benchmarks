import argparse
from collections import defaultdict as ddt
from typing import NamedTuple, Union

from torch_geometric.utils import add_remaining_self_loops

from GBKGNN.GBKGNN_training import *
from GBKGNN.data_loader import data_loaders
# from models import dnn, gat, gcn, gin, sage, gcn2
from GBKGNN.models import sage
from GBKGNN.utils.statistic import *

# MODEL_CLASSES = {'DNN': dnn.DNN, 'GraphSage': sage.GraphSage,
#                  'GIN': gin.GIN, 'GCN2': gcn2.GCN2,
#                  'GCN': gcn.GCN, 'GAT': gat.GAT, }
BASE_DIR = "../PathNet/other_data"
MODEL_CLASSES = {'GraphSage': sage.GraphSage}


def train_pathnetdata_gbkgnn(device: torch.device,
                             args: Union[NamedTuple, argparse.Namespace]):
    name = f'{args.name.replace("-", "_")}'
    experiment_ans = ddt(lambda: [])
    args.dataset_name = name
    model_name = args.model_type
    args.dim_size = args.n_hid
    args.aug = True  # use gbkgnn message passing method

    y = np.load(BASE_DIR + '/' + args.dataset_name + '/y.npy')

    lbl_set = []
    for lbl in y:
        if lbl not in lbl_set:
            lbl_set.append(lbl)

    if -1 in lbl_set:
        print('have unlabeled data; will be excluded in train/valid/test set')
        num_classes = len(lbl_set) - 1
    else:
        num_classes = len(lbl_set)

    acc_list = []
    torch.manual_seed(0)
    split_seed = 1234567
    for split_id in range(args.run):
        print('{}/{}'.format(split_id, args.run))
        args.dataset = data_loaders.DataLoader(args).dataset
        experiment_ans = ddt(lambda: [])
        experiment_ans["datasetName"].append(name)
        experiment_ans["nodeNum"].append(args.dataset["num_node"])
        experiment_ans["edgeNum"].append(args.dataset["num_edge"])
        experiment_ans["nodeFeaturesDim"].append(
            args.dataset["num_node_features"])
        experiment_ans["nodeClassification"].append(
            args.dataset["num_node_classes"])
        experiment_ans["smoothness"].append(
            compute_smoothness(args.dataset["graph"][0]))
        #
        if model_name != "GraphSage":
            edge_index, _ = add_remaining_self_loops(
                args.dataset["graph"][0].edge_index, None, 1, args.dataset["num_node"])
        else:
            edge_index = args.dataset["graph"][0].edge_index
        #
        n = args.dataset["num_node"]
        # c = args.dataset['num_classes']
        c = num_classes
        args.dataset['num_classes'] = num_classes
        # split
        args.similarity = compute_cosine_similarity(
            args.dataset, edge_index, "label")
        # data split
        train_index, val_index, test_index = random_splits_with_unlabel(
            args.dataset["graph"][0].y, ratio=[60, 20, 20], seed=split_seed)
        # index to mask
        args.dataset["graph"][0].train_mask = index_to_mask(n, train_index)
        args.dataset["graph"][0].val_mask = index_to_mask(n, val_index)
        args.dataset["graph"][0].test_mask = index_to_mask(n, test_index)
        split_seed += 1
        #
        model = MODEL_CLASSES[args.model_type](args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # training
        test_acc = training(args, device, model, optimizer)
        acc_list.append(test_acc)

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = f'./pathnet.csv'
    print(f"Saving results to {filename}")
    # delete something before saving
    args.similarity = None
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.name}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test graph dataset used in PathNet")
    parser.add_argument('--name', type=str, default='Bgp', help='dataset name')
    # model training parameters
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--method', type=str, default='GBKGCN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--n_hid', type=int, default=64, help='Number of hidden dim')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-7)
    # gbkgnn
    parser.add_argument("--split", nargs="+",
                        default=[0.6, 0.2, 0.2], type=float)
    parser.add_argument("--model_type", default="GraphSage", type=str)
    parser.add_argument('--aug', dest='aug', default=False, action='store_true',
                        help="Whether use our message passing method.")
    parser.add_argument('--lamda', type=float, default=30,
                        help="The hypereparameter of regularization term.")
    parser.add_argument('--patience', type=int, default=10000,
                        help="number of epochs without improvement for early stopping.")
    parser.add_argument('--log_interval', type=int, default=100,
                        help="log interval while training.")
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    train_pathnetdata_gbkgnn(device, args)
