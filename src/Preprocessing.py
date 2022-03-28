'''
This file is for data preprocessing and data analysis
'''

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from matplotlib import pyplot
import numpy as np
from numpy import cov
from scipy.stats import pearsonr
from scipy import spatial
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import scipy.io as scio
from CFGT import CFGT

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def stats_cov(data1, data2):
    '''
    :param data1: np, d_1 x n
    :param data2: np, d_2 x n
    :return:
    '''
    cov = np.cov(data1, data2)  # (d_1 + d_2) x (d_1 + d_2)

    # only when data1 and data 2 are both shaped as (n,)
    corr_pear, p_value = pearsonr(data1, data2)
    result = {
        'cov': cov,
        'pearson': corr_pear,
        'pear_p_value': p_value
    }
    return result

def pre_analysis(adj, labels, sens):
    '''
    :param
        labels: n
        sens: n
        adj: csr_sparse, n x n
    :return: this function analyze:
        1. the correlation between the sensitive attributes of neighbors and the labels
        2. the correlation between the sensitive attributes of itself and the labels
        S_N(i), Y_i | S_i  not independent ->
    '''
    adj_noself = adj.copy()
    adj_noself.setdiag(0)  # remove edge v -> v
    if (adj_noself != adj_noself.T).nnz == 0:
        print("symmetric!")
    else:
        print("not symmetric!")
    adj_degree = adj.sum(axis=1)
    ave_degree = adj_degree.sum()/len(adj_degree)
    print('averaged degree: ', ave_degree, ' max degree: ', adj_degree.max(), ' min degree: ', adj_degree.min())

    # inter- and intra- connections
    node_num = adj.shape[0]
    edge_num = (len(adj.nonzero()[0]) - node_num) / 2
    intra_num = 0
    for u, v in zip(*adj.nonzero()):  # u -> v
        if u >= v:
            continue
        if sens[u] == sens[v]:
            intra_num += 1
    print("edge num: ", edge_num, " intra-group edge: ", intra_num, " inter-group edge: ", edge_num - intra_num)

    # row-normalize
    adj_noself = normalize(adj_noself, norm='l1', axis=1)
    nb_sens_ave = adj_noself @ sens  # n x 1, average sens of 1-hop neighbors

    # Y_i, S_i
    #pyplot.scatter(labels, sens)
    #pyplot.show()

    cov_results = stats_cov(labels, sens)
    print('correlation between Y and S:', cov_results)

    # S_N(i), Y_i | S_i
    cov_nb_results = stats_cov(labels, nb_sens_ave)
    print('correlation between Y and neighbors (not include self)\' S:', cov_nb_results)

    # R^2
    X = sens.reshape(node_num, -1)
    reg = LinearRegression().fit(X, labels)
    y_pred = reg.predict(X)
    R2 = r2_score(labels, y_pred)
    print('R2 - self: ', R2, ' ', reg.score(X, labels))

    X = nb_sens_ave.reshape(node_num, -1)
    reg = LinearRegression().fit(X, labels)
    y_pred = reg.predict(X)
    R2 = r2_score(labels, y_pred)
    print('R2 - neighbor: ', R2, ' ', reg.score(X, labels))

    return

def generate_cf_true(data, dataset, sens_rate_list, sens_idx, save_path, save_file=True, raw_data_info=None, mode=1):
    n = data.x.shape[0]
    if dataset == 'synthetic':
        generate_cf_true_synthetic(data, dataset, sens_rate_list, sens_idx, save_path, save_file=save_file, raw_data_info=raw_data_info)
        return
    else:
        generate_cf_true_rw(data, dataset, sens_rate_list, sens_idx, save_path, save_file=save_file, raw_data_info=raw_data_info)

    return



# Load data
# print(args.dataset)
def load_data(path_root, dataset):
    # Load credit_scoring dataset
    raw_data_info = None
    if dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = path_root + "./dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset, sens_attr,
                                                                                predict_attr, path=path_credit,
                                                                                label_number=label_number
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    # Load german dataset
    elif dataset == 'german':
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "GoodCustomer"
        label_number = 100
        path_german = path_root + "./dataset/german"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )
    # Load bail dataset
    elif dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = path_root + "./dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(dataset, sens_attr,
                                                                                predict_attr, path=path_bail,
                                                                                label_number=label_number,
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    elif dataset == 'synthetic':
        sens_idx = 0
        label_number = 1000
        path_sythetic = path_root + './dataset/synthetic.mat'
        adj, features, labels, idx_train, idx_val, idx_test, sens, raw_data_info = load_synthetic(path=path_sythetic,
                                                                              label_number=label_number)
    else:
        print('Invalid dataset name!!')
        exit(0)

    print("loaded dataset: ", dataset, "num of node: ", len(features), ' feature dim: ', features.shape[1])

    num_class = labels.unique().shape[0]-1
    return adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, raw_data_info


# get true cf for real-world datasets
def generate_cf_true_rw(data, dataset, sens_rate_list, sens_idx, save_path, save_file=True, train='test', raw_data_info=None):
    n = data.x.shape[0]
    input_dim = data.x.shape[1]
    h_dim = 32
    w_hd_x = 0.95
    thresh_a = 0.9
    adj_orin = raw_data_info['adj']
    adj = adj_orin.tocoo()
    indices_adj = torch.LongTensor([adj.row, adj.col])
    adj = torch.sparse_coo_tensor(indices_adj, adj.data, size=(adj.shape[0], adj.shape[1])).float()

    for i in range(len(sens_rate_list)):
        sens_rate = sens_rate_list[i]
        sampled_idx = random.sample(range(n), int(sens_rate * n))
        data_cf = data.clone()

        sens_new = torch.zeros_like(data_cf.x[:, sens_idx])
        sens_new[sampled_idx] = 1

        # X
        sens = data.x[:, sens_idx]
        idx_1 = (sens == 1)
        idx_0 = (sens == 0)
        x_mean_1 = data.x[idx_1, :]  # n1 x d
        x_mean_0 = data.x[idx_0, :]
        x_mean_diff = x_mean_1.mean(dim=0) - x_mean_0.mean(dim=0)  # d
        x_update = ((sens_new - sens).view(-1, 1).tile(1, x_mean_1.shape[1]) * x_mean_diff.view(1, -1).tile(n, 1))
        data_cf.x = w_hd_x * data.x + (1-w_hd_x) * x_update

        # S
        data_cf.x[:, sens_idx] = sens_new

        # adj
        model_GT = CFGT(h_dim, input_dim, adj.cuda()).cuda()
        if train == 'test':  # train or load existing model
            model_GT.load_state_dict(torch.load(save_path + f'weights_CFGT_{dataset}' + '.pt'))
            # test?
            test_model = False
            if test_model:
                eval_result = model_GT.test(data.x.cuda(), adj.cuda(), sens_idx)
                print(
                    'loss_reconst_a: {:.4f}'.format(eval_result['loss_reconst_a'].item()),
                    'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                    'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                    'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                )
        else:
            model_GT.train_model(data.x.cuda(), adj.cuda(), sens_idx, dataset, model_path=save_path, lr=0.0001, weight_decay=1e-5)

        # generate cf for whole graph to achieve better efficiency
        Z_a = model_GT.encode(data_cf.x.cuda())
        adj_update = model_GT.pred_graph(Z_a, sens_new.view(-1, 1).cuda())
        adj_cf = adj.to_dense().clone()
        adj_cf[adj_update > thresh_a] = 1  # to binary
        adj_cf[adj_update < 1 - thresh_a] = 0
        rate_na = (adj.to_dense() != adj_cf).sum() / (n * n)
        print('rate of A change: ', rate_na)

        edge_index_cf = adj_cf.to_sparse().indices().cpu()
        data_cf.edge_index = edge_index_cf

        # skip y, as y is not used

        # data_cf
        data_results = {'data': data_cf}
        # save in files
        if save_file:
            with open(save_path + '/' + dataset + '_cf_' + str(sens_rate) + '.pkl', 'wb') as f:
                pickle.dump(data_results, f)
                print('saved counterfactual data: ', dataset + '_cf_' + str(sens_rate) + '.pkl')

    return

def generate_cf_true_synthetic(data, dataset, sens_rate_list, sens_idx, save_path, save_file=True, raw_data_info=None):
    # generate graphs in sens_rate_list
    embedding = raw_data_info['z']
    v = raw_data_info['v']
    feat_idxs = raw_data_info['feat_idxs']
    w = raw_data_info['w']
    w_s = raw_data_info['w_s']
    n = data.x.shape[0]
    adj_orin = raw_data_info['adj']
    alpha = raw_data_info['alpha']
    oa = 0.9

    for i in range(len(sens_rate_list)):
        sens_rate = sens_rate_list[i]
        sampled_idx = random.sample(range(n), int(sens_rate * n))
        data_cf = data.clone()
        data_cf.x[:, sens_idx] = 0
        data_cf.x[sampled_idx, sens_idx] = 1

        sens = data_cf.x[:, sens_idx].numpy()

        # x\s
        features_cf = embedding[:, feat_idxs] + (np.dot(sens.reshape(-1, 1), v))  # (n x dim) + (1 x dim) -> n x dim
        features_cf = torch.FloatTensor(features_cf)
        data_cf.x[:, 1:] = oa * data_cf.x[:, 1:] + (1 - oa) * features_cf

        # adj
        sens_sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):  # i<=j
                if i == j:
                    sens_sim[i][j] = 1
                    continue
                sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])

        similarities = cosine_similarity(embedding)  # n x n
        adj = similarities + alpha * sens_sim

        adj = oa * adj_orin + (1 - oa) * adj

        adj[np.where(adj >= 0.4)] = 1
        adj[np.where(adj < 0.4)] = 0
        adj = sparse.csr_matrix(adj)

        data_cf.edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

        # skip y
        # adj_norm = normalize(adj, norm='l1', axis=1)
        # nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors
        #
        # labels = np.matmul(embedding, w) + w_s * nb_sens_ave.reshape(-1, 1)  # n x 1
        # labels = labels.reshape(-1)
        #
        # labels = oa * data_cf.y.numpy() + (1-oa)* labels
        # labels_mean = np.mean(labels)
        # labels_binary = np.zeros_like(labels)
        # labels_binary[np.where(labels > labels_mean)] = 1.0
        # data_cf.y = torch.FloatTensor(labels_binary)

        data_results = {'data': data_cf}

        # save in files
        if save_file:
            with open(save_path + '/' + dataset + '_cf_' + str(sens_rate) + '.pkl', 'wb') as f:
                pickle.dump(data_results, f)
                print('saved counterfactual data: ', dataset + '_cf_' + str(sens_rate) + '.pkl')
    return

def generate_synthetic_data(path):
    n = 2000
    z_dim = 50
    dim = 25
    p = 0.4
    alpha = 0.01  #
    sens = np.random.binomial(n=1, p=p, size=n)
    embedding = np.random.normal(loc=0, scale=1, size=(n, z_dim))
    feat_idxs = random.sample(range(z_dim), dim)
    v = np.random.normal(0, 1, size=(1, dim))
    features = embedding[:, feat_idxs] + (np.dot(sens.reshape(-1,1), v))  # (n x dim) + (1 x dim) -> n x dim

    adj = np.zeros((n, n))
    sens_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # i<=j
            if i == j:
                sens_sim[i][j] = 1
                continue
            sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])
            # sim_ij = 1 - spatial.distance.cosine(embedding[i], embedding[j])  # [-1, 1]
            # adj[i][j] = adj[j][i] = sim_ij + alpha * (sens[i] == sens[j])

    similarities = cosine_similarity(embedding)  # n x n
    adj = similarities + alpha * sens_sim

    print('adj max: ', adj.max(), ' min: ', adj.min())
    adj[np.where(adj >= 0.4)] = 1
    adj[np.where(adj < 0.4)] = 0
    adj = sparse.csr_matrix(adj)

    w = np.random.normal(0, 1, size=(z_dim, 1))
    w_s = 1

    adj_norm = normalize(adj, norm='l1', axis=1)
    nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors

    dd = np.matmul(embedding, w)
    d2 = nb_sens_ave.reshape(-1,1)
    print('y component: ', np.mean(dd), np.mean(d2))
    labels = np.matmul(embedding, w) + w_s * nb_sens_ave.reshape(-1,1) # n x 1
    labels = labels.reshape(-1)
    labels_mean = np.mean(labels)
    labels_binary = np.zeros_like(labels)
    labels_binary[np.where(labels > labels_mean)] = 1.0

    print('pos labels: ', labels_binary.sum(), ' neg: ', len(labels_binary) - labels_binary.sum())

    # statistics
    pre_analysis(adj, labels, sens)

    data = {'x': features, 'adj': adj, 'labels': labels_binary, 'sens': sens,
            'z': embedding, 'v': v, 'feat_idxs': feat_idxs, 'alpha': alpha, 'w': w, 'w_s': w_s}
    scio.savemat(path, data)
    print('data saved in ', path)
    return data

if __name__ == '__main__':
    dataset = 'credit'
    path_root = './nifty-main/'
    path = path_root+'synthetic.mat'
    generate_synthetic_data(path)
