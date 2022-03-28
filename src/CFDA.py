import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import optim

class CFDA(nn.Module):
    def __init__(self, h_dim, input_dim, adj):
        super(CFDA, self).__init__()
        self.type = 'GAE'
        self.h_dim = h_dim
        self.s_num = 4
        # A
        self.base_gcn = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.pred_a = nn.Sequential(nn.Linear(h_dim+1, adj.shape[1]), nn.Sigmoid())
        # X
        self.base_gcn_x = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean_x = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.gcn_logstddev_x = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)

        # reconst_X
        self.reconst_X = nn.Sequential(nn.Linear(h_dim+1, input_dim))
        # pred_S
        self.pred_s = nn.Sequential(nn.Linear(h_dim + h_dim, self.s_num), nn.Softmax())

    def encode_A(self, X):
        mask_X = X
        hidden = self.base_gcn(mask_X)
        mean = self.gcn_mean(hidden)
        logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.h_dim)
        if self.training and self.type == 'VGAE':
            sampled_z = gaussian_noise * torch.exp(logstd) + mean
        else:
            sampled_z = mean
        return sampled_z

    def encode_X(self, X):
        hidden = self.base_gcn_x(X)
        mean = self.gcn_mean_x(hidden)
        logstd = self.gcn_logstddev_x(hidden)
        gaussian_noise = torch.randn(X.size(0), self.h_dim)
        if self.training and self.type == 'VGAE':
            sampled_z = gaussian_noise * torch.exp(logstd) + mean
        else:
            sampled_z = mean
        return sampled_z

    def pred_adj(self, Z, S):
        ZS = torch.cat([Z, S], dim=1)
        A_pred = self.pred_a(ZS)
        # A_pred = F.sigmoid(self.pred_a(ZS, ZS))
        # A_pred = torch.sigmoid(torch.matmul(ZS, ZS.t()))
        return A_pred

    def pred_features(self, Z, S):
        ZS = torch.cat([Z, S], dim=1)
        X_pred = self.reconst_X(ZS)
        return X_pred

    def pred_S_agg(self, Z):
        S_pred = self.pred_s(Z)
        return S_pred

    def encode(self, X):
        Z_a = self.encode_A(X)
        Z_x = self.encode_X(X)
        return Z_a, Z_x

    def pred_graph(self, Z_a, Z_x, S):
        A_pred = self.pred_adj(Z_a, S)
        X_pred = self.pred_features(Z_x, S)
        return A_pred, X_pred

    def forward(self, X, sen_idx):
        # encoder: X\S, adj -> Z
        # decoder: Z + S' -> X', A'
        S = X[:, sen_idx].view(-1, 1)
        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this dim
        Z_a, Z_x = self.encode(X_ns)
        A_pred, X_pred = self.pred_graph(Z_a, Z_x, S)
        S_agg_pred = self.pred_S_agg(torch.cat([Z_a, Z_x], dim=1))
        return A_pred, X_pred, S_agg_pred

    def loss_function(self, adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred):
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

        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this sensitive dim
        loss_mse = nn.MSELoss(reduction='mean')
        loss_reconst_x = loss_mse(X_pred, X_ns)

        loss_ce = nn.CrossEntropyLoss()
        loss_s = loss_ce(S_agg_pred, S_agg_cat.view(-1))  # S_agg_pred: n x K, S_agg: n
        loss_result = {'loss_reconst_a': loss_reconst_a, 'loss_reconst_x': loss_reconst_x, 'loss_s': loss_s}
        return loss_result

    def train_model(self, X, adj, sen_idx, dataset, model_path='', lr=0.0001, weight_decay=1e-5):
        rate_1 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
        print('adj=1: ', rate_1)

        par_s = list(self.pred_s.parameters())
        par_other = list(self.base_gcn.parameters()) + list(self.gcn_mean.parameters()) + list(self.gcn_logstddev.parameters()) + list(self.pred_a.parameters()) + \
                    list(self.base_gcn_x.parameters()) + list(self.gcn_mean_x.parameters()) + list(self.gcn_logstddev_x.parameters()) + list(self.reconst_X.parameters())
        optimizer_1 = optim.Adam([{'params': par_s, 'lr': lr}], weight_decay=weight_decay)  #
        optimizer_2 = optim.Adam([{'params': par_other, 'lr': lr}], weight_decay=weight_decay)  #

        self.train()
        n = X.shape[0]

        S = X[:, sen_idx].view(-1, 1)  # n x 1
        S_agg = torch.mm(adj, S) / n  # n x 1
        S_agg_max = S_agg.max()
        S_agg_min = S_agg.min()
        S_agg_cat = torch.floor(S_agg / ((S_agg_max + 0.000001 - S_agg_min) / self.s_num)).long()  # n x 1

        print("start training counterfactual augmentation module!")
        for epoch in range(2000):
            for i in range(3):
                optimizer_1.zero_grad()

                A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
                loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)

                # backward propagation
                loss_s = loss_result['loss_s']
                loss_s.backward()
                optimizer_1.step()

            for i in range(5):
                optimizer_2.zero_grad()

                A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
                loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)

                # backward propagation
                loss_s = loss_result['loss_s']
                loss_reconst_x = loss_result['loss_reconst_x']
                loss_reconst_a = loss_result['loss_reconst_a']
                #loss_reconst_a.backward()
                (-loss_s + loss_reconst_a + loss_reconst_x).backward()
                optimizer_2.step()

            if epoch % 100 == 0:
                self.eval()
                eval_result = self.test(adj, X, sen_idx, S_agg_cat)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_reconst_a: {:.4f}'.format(loss_reconst_a.item()),
                      'loss_reconst_x: {:.4f}'.format(loss_reconst_x.item()),
                      'loss_s: {:.4f}'.format(loss_s.item()),
                      'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                      'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                      'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                      )
                # save model
                save_model = True
                if save_model and epoch > 0:
                    save_model_path = model_path + f'weights_CFDA_{dataset}' + '.pt'
                    torch.save(self.state_dict(), save_model_path)
                    print('saved model weight in: ', save_model_path)
                self.train()
        return

    def test(self, adj, X, sen_idx, S_agg_cat):
        self.eval()
        A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
        loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)
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

        eval_result = loss_result
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
