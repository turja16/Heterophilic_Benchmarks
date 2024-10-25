# GloGNN
# https://github.com/RecklessRonan/GloGNN/blob/master/small-scale/main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

class MLP_NORM(nn.Module):
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta, norm_func_id, norm_layers, orders, orders_func_id, cuda):
        super(MLP_NORM, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.fc3 = nn.Linear(nnodes, nhid)
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(gamma)
        self.delta = torch.tensor(delta)
        self.norm_layers = norm_layers
        self.orders = orders
        self.class_eye = torch.eye(nclass)
        self.nodes_eye = torch.eye(nnodes)

        if cuda:
            self.orders_weight = Parameter(
                torch.ones(orders, 1) / orders, requires_grad=True
            ).to('cuda')
            # use kaiming_normal to initialize the weight matrix in Orders3
            self.orders_weight_matrix = Parameter(
                torch.DoubleTensor(nclass, orders), requires_grad=True
            ).to('cuda')
            self.orders_weight_matrix2 = Parameter(
                torch.DoubleTensor(orders, orders), requires_grad=True
            ).to('cuda')
            # use diag matirx to initialize the second norm layer
            self.diag_weight = Parameter(
                torch.ones(nclass, 1) / nclass, requires_grad=True
            ).to('cuda')
            self.alpha = self.alpha.cuda()
            self.beta = self.beta.cuda()
            self.gamma = self.gamma.cuda()
            self.delta = self.delta.cuda()
            self.class_eye = self.class_eye.cuda()
            self.nodes_eye = self.nodes_eye.cuda()
        else:
            self.orders_weight = Parameter(
                torch.ones(orders, 1) / orders, requires_grad=True
            )
            # use kaiming_normal to initialize the weight matrix in Orders3
            self.orders_weight_matrix = Parameter(
                torch.DoubleTensor(nclass, orders), requires_grad=True
            )
            self.orders_weight_matrix2 = Parameter(
                torch.DoubleTensor(orders, orders), requires_grad=True
            )
            # use diag matirx to initialize the second norm layer
            self.diag_weight = Parameter(
                torch.ones(nclass, 1) / nclass, requires_grad=True
            )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()

        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def forward(self, x, adj):
        xX = F.dropout(x, self.dropout, training=self.training)
        xX = self.fc1(x)
        xA = self.fc3(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        for _ in range(self.norm_layers):
            # adj_drop = F.dropout(adj, self.dropout, training=self.training)
            x = self.norm(x, h0, adj)
        # return F.log_softmax(x, dim=1)
        return x

    def norm_func1(self, x, h0, adj):
        # print('norm_func1 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        # print('norm_func2 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0

        # calculate z
        xx = torch.mm(x, x.t())
        hx = torch.mm(h0, x.t())
        # print('adj', adj.shape)
        # print('orders_weight', self.orders_weight[0].shape)
        adj = adj.to_dense()
        adjk = adj
        a_sum = adjk * self.orders_weight[0]
        for i in range(1, self.orders):
            adjk = torch.mm(adjk, adj)
            a_sum += adjk * self.orders_weight[i]
        z = torch.mm(coe1 * xx + self.beta * a_sum - self.gamma * coe1 * hx,
                     torch.inverse(coe1 * coe1 * xx + (self.alpha + self.beta) * self.nodes_eye))
        # print(z.shape)
        # print(z)
        return res

    def order_func1(self, x, res, adj):
        # Orders1
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # Orders2
        tmp_orders = torch.spmm(adj, res)
        # print('tmp_orders', tmp_orders.shape)
        # print('orders_weight', self.orders_weight[0].shape)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        # Orders3
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        # orders_para = torch.mm(x, self.orders_weight_matrix)
        orders_para = torch.transpose(orders_para, 0, 1)
        tmp_orders = torch.spmm(adj, res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders
