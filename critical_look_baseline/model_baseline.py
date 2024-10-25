from torch import nn
from dgl import ops

# do not use critical think's setting
class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        layer = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        layer.dropout = nn.Dropout(p=dropout)
        layer.act = nn.ReLU()
        self.layers.append(layer)
        layer2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.layers.append(layer2)

    def forward(self, graph, x):
        x = self.layers[0](x)
        x = self.layers[0].dropout(x)
        x = self.layers[0].act(x)
        x = self.layers[1](x)
        return x

class MLP1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):

        super().__init__()
        self.layers = nn.ModuleList()
        layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.layers.append(layer)

    def forward(self, graph, x):
        x = self.layers[0](x)
        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feed_forward_module = nn.Linear(
            in_features=input_dim, out_features=output_dim)

    def forward(self, graph, x):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5
        x = ops.u_mul_e_sum(graph, x, norm_coefs)
        x = self.feed_forward_module(x)

        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):

        super().__init__()
        self.layers = nn.ModuleList()
        layer = GCNLayer(input_dim=input_dim, output_dim=hidden_dim)
        layer.dropout = nn.Dropout(p=dropout)
        layer.act = nn.ReLU()
        self.layers.append(layer)
        layer2 = GCNLayer(input_dim=hidden_dim, output_dim=output_dim)
        self.layers.append(layer2)

    def forward(self, graph, x):
        x = self.layers[0](graph, x)
        x = self.layers[0].dropout(x)
        x = self.layers[0].act(x)
        x = self.layers[1](graph, x)
        return x
   
class SGC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):

        super().__init__()
        self.layers = nn.ModuleList()
        layer = GCNLayer(input_dim=input_dim, output_dim=output_dim)
        self.layers.append(layer)

    def forward(self, graph, x):
        x = self.layers[0](graph, x)
        return x