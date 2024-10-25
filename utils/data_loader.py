import numpy as np
import scipy.sparse as sp
import torch
import copy

class DataLoader(object):
    def __init__(self, adj_mat, train_nodes, valid_nodes, test_nodes, device):
        # have removed the sym normalized part
        self.adj_mat = adj_mat
        self.train_nodes = train_nodes
        self.valid_nodes = valid_nodes
        self.test_nodes = test_nodes
        self.device = device
        self.num_nodes = adj_mat.shape[0]
        self.num_train_nodes = len(self.train_nodes)
        self.lap_tensor = self.sparse_mx_to_torch_sparse_tensor(adj_mat)
        self.lap_tensor = torch.sparse.FloatTensor(self.lap_tensor[0],
                                                   self.lap_tensor[1], 
                                                   self.lap_tensor[2]).to(device)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
            indices = torch.LongTensor([[], []])
        else:
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return indices, values, shape
            
    def get_train_batch(self, ):
        return self.train_nodes, self.lap_tensor

    def get_valid_batch(self,):
        return self.valid_nodes, self.lap_tensor

    def get_test_batch(self,):
        return self.test_nodes, self.lap_tensor

class ResultRecorder:
    def __init__(self, note):
        self.train_loss_record = []
        self.train_acc_record = []
        self.loss_record = []
        self.acc_record = []
        self.best_acc = None
        self.best_model = None
        self.note = note
        self.sample_time = []
        self.compute_time = []
        self.state_dicts = []
        self.grad_norms = []
        
    def update(self, train_loss, train_acc, loss, acc, model, sample_time=0, compute_time=0):
        self.sample_time += [sample_time]
        self.compute_time += [compute_time]

        self.train_loss_record += [train_loss]
        self.train_acc_record += [train_acc]
        
        self.loss_record += [loss]
        self.acc_record += [acc]
            
        if self.best_acc is None:
            self.best_acc = acc
            self.best_model = copy.deepcopy(model).cpu()
        elif self.best_acc < acc:
            self.best_acc = acc
            self.best_model = copy.deepcopy(model).cpu()
