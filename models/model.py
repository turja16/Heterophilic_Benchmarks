import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.layers import GraphConv

#########################################################
#########################################################
#########################################################
class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, dropout, criterion):
        super(GCN, self).__init__()
        self.n_hid = n_hid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConv(n_feat,  n_hid))
        self.gcs.append(GraphConv(n_hid,  n_classes))
        self.relu = nn.ReLU()
        self.dropout = dropout
        self.criterion = criterion

    def forward(self, x, adj):
        x = self.gcs[0](x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(x)
        x = self.gcs[1](x, adj)
        return x

#########################################################

class MLP2(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, dropout, criterion):
        super(MLP2, self).__init__()
        self.conv1 = nn.Linear(n_feat, n_hid)
        self.conv2 = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.criterion = criterion
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(x)
        x = self.conv2(x)
        return x

#########################################################
    
class MLP1(nn.Module):
    def __init__(self, n_feat, n_classes, criterion):
        super(MLP1, self).__init__()
        self.conv1 = nn.Linear(n_feat, n_classes)
        self.criterion = criterion

    def forward(self, x, adj):
        x = self.conv1(x)
        return x

#########################################################
    
class SGC1(nn.Module):
    def __init__(self, n_feat, n_classes, criterion):
        super(SGC1, self).__init__()
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConv(n_feat,  n_classes))
        self.criterion = criterion

    def forward(self, x, adj):
        x = self.gcs[0](x, adj)
        return x