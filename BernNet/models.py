import torch
import torch.nn.functional as F
from torch.nn import Linear

# from torch_geometric.nn import GATConv, GCNConv, ChebConv
# from torch_geometric.nn import MessagePassing, APPNP
from BernNet.Bernpro import Bern_prop


class BernNet(torch.nn.Module):
    # def __init__(self, dataset, args):
    def __init__(self, in_dim, n_hid, n_classes, args):
        super(BernNet, self).__init__()
        # self.lin1 = Linear(dataset.num_features, args.hidden)
        # self.lin2 = Linear(args.hidden, dataset.num_classes)
        # self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.lin1 = Linear(in_dim, n_hid)
        self.lin2 = Linear(n_hid, n_classes)
        self.m = torch.nn.BatchNorm1d(n_classes)
        self.prop1 = Bern_prop(args.K)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    # def forward(self, data):
    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            # return F.log_softmax(x, dim=1)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            # return F.log_softmax(x, dim=1)
            return x
