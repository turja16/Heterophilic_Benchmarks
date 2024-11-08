from typing import NamedTuple, Union

from scipy import sparse
from torch_geometric.utils import to_dense_adj

from models.model import *
from utils.data_loader import DataLoader
from utils.train_helper import *

BASE_DIR = "./mixhop_syn-2000_5/"


def edge_mixhop_to_edge_list(edge_mixhop):
    adj_indices = []
    for node, neighbors in edge_mixhop.items():
        for n in neighbors:
            adj_indices.append([node, n])
    return np.transpose(adj_indices)


def random_disassortative_splits(labels, num_classes, training_percentage=0.6):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels = labels.cpu()
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
    val_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]
    return train_index, val_index, test_index


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


def train_syn_basline(device: torch.device,
                      args: Union[NamedTuple, argparse.Namespace]):
    criterion = nn.CrossEntropyLoss()
    acc_list = []
    train_time = []
    torch.manual_seed(0)
    for graph_idx in range(args.run):
        # load data
        feat = torch.load(os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.allx".format(args.h, graph_idx)))
        label = torch.load(
            os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.ally".format(args.h, graph_idx)))  # one-hot label
        edge_mixhop = torch.load(os.path.join(BASE_DIR, "ind.n2000-h{}-c5-g{}.graph".format(args.h, graph_idx)))
        num_nodes = len(edge_mixhop)
        edge_list = edge_mixhop_to_edge_list(edge_mixhop)
        adj = to_dense_adj(torch.tensor(edge_list).to(dtype=torch.int64))[0].numpy()
        label = label.argmax(1)
        num_classes = label.max() + 1
        labels_th = torch.LongTensor(label).to(device)
        feat_data_th = torch.FloatTensor(feat).to(device)
        # normalize
        feat_data_th = normalize_tensor(feat_data_th, symmetric=0)
        adj = normalize_tensor(torch.tensor(adj + np.eye(num_nodes), dtype=torch.float32), symmetric=1)
        adj = sparse.csr_matrix(adj.numpy())
        # adj = sparse.csr_matrix(adj)
        # generate split
        train_nodes, valid_nodes, test_nodes = random_disassortative_splits(
            labels_th, num_classes, training_percentage=0.6)
        data_loader = DataLoader(adj, train_nodes, valid_nodes, test_nodes, device)
        # select model
        if args.method == 'GCN':
            model = GCN(n_feat=feat_data_th.shape[1], n_hid=64, n_classes=num_classes, dropout=args.dropout,
                        criterion=criterion).to(device)
        elif args.method == 'MLP2':
            model = MLP2(n_feat=feat_data_th.shape[1], n_hid=64, n_classes=num_classes, dropout=args.dropout,
                         criterion=criterion).to(device)
        elif args.method == 'MLP1':
            model = MLP1(n_feat=feat_data_th.shape[1], n_classes=num_classes, criterion=criterion).to(device)
        elif args.method == 'SGC1':
            model = SGC1(n_feat=feat_data_th.shape[1], n_classes=num_classes, criterion=criterion).to(device)
        else:
            raise ValueError('model does not exist')
        # print(model)
        # train
        results, acc, avg_time = train(
            model,
            args, feat_data_th, labels_th,
            data_loader, device, note="%s (layers = 2)" % (args.method)
        )
        eval_model = results.best_model.to(device)
        output = eval_model(feat_data_th, data_loader.lap_tensor)
        output = output.argmax(dim=1)
        acc = f1_score(
            output[test_nodes].detach().cpu(),
            labels_th[test_nodes].detach().cpu(),
            average="micro")
        acc_list.append(acc)
        train_time.append(avg_time)

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = './mixhop.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.h}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    # syn data parameters
    parser = argparse.ArgumentParser(description="Test synthetic graph dataset generated by Mixhop")
    parser.add_argument("--h", type=float, default=0.0, help="number of nodes per class in the synthetic dataset", )
    # model training parameters
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--method', type=str, default='GCN', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    train_syn_basline(device, args)
