import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import optim

class CFGT(nn.Module):
    def __init__(self, h_dim, input_dim, adj):
        super(CFGT, self).__init__()
        self.h_dim = h_dim
        # A
        self.base_gcn = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.pred_a = nn.Sequential(nn.Linear(h_dim, adj.shape[1]))

        # S
        self.sf = nn.Sequential(nn.Linear(1, 1))  # n x 1, parameter: 1

    def encode_A(self, X):
        mask_X = X
        hidden = self.base_gcn(mask_X)
        mean = self.gcn_mean(hidden)
        sampled_z = mean
        return sampled_z

    def pred_adj(self, Z, S):
        A_pred = self.pred_a(Z)  # n x n
        S_rep_f = self.sf(S)
        S_rep_cf = self.sf(1 - S)

        s_match = (torch.matmul(S_rep_f, S_rep_f.t()) + torch.matmul(S_rep_cf, S_rep_cf.t())) / 2
        A_pred = F.sigmoid(A_pred + s_match)
        return A_pred

    def encode(self, X):
        Z_a = self.encode_A(X)
        return Z_a

    def pred_graph(self, Z_a, S):
        A_pred = self.pred_adj(Z_a, S)
        return A_pred

    def forward(self, X, sen_idx):
        # encoder: X\S, adj -> Z
        # decoder: Z + S' -> A'
        S = X[:, sen_idx].view(-1, 1)
        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this dim

        Z_a = self.encode(X_ns)
        A_pred = self.pred_graph(Z_a, S)
        return A_pred

    def loss_function(self, adj, A_pred):
        # loss_reconst
        weighted = True
        if weighted:
            weights_0 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
            weights_1 = 1 - weights_0
            assert (weights_0 > 0 and weights_1 > 0)
            weight = torch.ones_like(A_pred).reshape(-1) * weights_0  # (n x n), weight 0
            idx_1 = adj.to_dense().reshape(-1) == 1
            weight[idx_1] = weights_1

            loss_bce = nn.BCELoss(weight=weight, reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))
        else:
            loss_bce = nn.BCELoss(reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))

        loss_result = {'loss_reconst_a': loss_reconst_a}
        return loss_result

    def train_model(self, X, adj, sen_idx, dataset, model_path='', lr=0.0001, weight_decay=1e-5):
        rate_1 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
        print('adj=1: ', rate_1)

        optimizer = optim.Adam([{'params': self.parameters(), 'lr': lr}], weight_decay=weight_decay)

        self.train()
        n = X.shape[0]

        print("start training counterfactual augmentation module!")
        for epoch in range(2000):
            optimizer.zero_grad()

            A_pred = self.forward(X, sen_idx)
            loss_result = self.loss_function(adj, A_pred)

            # backward propagation
            loss_reconst_a = loss_result['loss_reconst_a']
            loss_reconst_a.backward()
            optimizer.step()

            if epoch % 100 == 0:
                self.eval()
                eval_result = self.test(X, adj, sen_idx)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_reconst_a: {:.4f}'.format(loss_reconst_a.item()),
                      'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                      'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                      'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                      )
                # save model
                save_model = True
                if save_model and epoch > 0:
                    save_model_path = model_path + f'weights_CFGT_{dataset}' + '.pt'
                    torch.save(self.state_dict(), save_model_path)
                    print('saved model weight in: ', save_model_path)
                self.train()
        return

    def test(self, X, adj, sen_idx):
        self.eval()
        A_pred = self.forward(X, sen_idx)
        loss_result = self.loss_function(adj, A_pred)
        eval_result = loss_result

        A_pred_binary = (A_pred > 0.5).float()  # binary
        adj_size = A_pred_binary.shape[0] * A_pred_binary.shape[1]

        sum_1 = torch.sparse.sum(adj)
        correct_num_1 = torch.sparse.sum(sparse_dense_mul(adj, A_pred_binary))  # 1
        correct_num_0 = (adj_size - (A_pred_binary + adj).sum() + correct_num_1)
        acc_a_pred = (correct_num_1 + correct_num_0) / adj_size
        acc_a_pred_0 = correct_num_0 / (adj_size - sum_1)
        acc_a_pred_1 = correct_num_1 / sum_1

        eval_result['acc_a_pred'] = acc_a_pred
        eval_result['acc_a_pred_0'] = acc_a_pred_0
        eval_result['acc_a_pred_1'] = acc_a_pred_1
        return eval_result

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())