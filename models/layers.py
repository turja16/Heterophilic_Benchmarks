import torch
import torch.nn as nn
import math

class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, bias=False):
        super(GraphConv, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)

        self.uniform(n_in, self.linear.weight)
        self.zeros(self.linear.bias)
    
    def uniform(self, size, tensor):
        if tensor is not None:
            bound = 1.0 / math.sqrt(size)
            tensor.data.uniform_(-bound, bound)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)


    def forward(self, x, adj):
        x = torch.spmm(adj, x)
        x = self.linear(x)
        return x