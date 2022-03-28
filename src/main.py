'''
Graph fairness
'''

import time
import argparse
import numpy as np
import random
import math
import sys
import pickle

from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE as tsn

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import utils
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.data import Data
import torch_geometric.utils as gm_utils

import Preprocessing as dpp

from utils_mp import Subgraph, preprocess
import models
from CFDA import CFDA

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, # 1000
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--proj_hidden', type=int, default=16,
                    help='Number of hidden units in the projection layer of encoder.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--sim_coeff', type=float, default=0.6,
                    help='regularization similarity')
parser.add_argument('--dataset', type=str, default='synthetic',
                    choices=['synthetic','bail', 'credit'])
parser.add_argument('--encoder', type=str, default='sage', choices=['gcn', 'gin', 'sage', 'infomax', 'jk'])
parser.add_argument('--batch_size', type=int, help='batch size', default=100)
parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=30)
parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=10)
parser.add_argument('--hidden_size', type=int, help='hidden size', default=1024)
parser.add_argument('--experiment_type', type=str, default='train', choices=['train', 'cf', 'test'])   # train, cf, test


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_list_in_dict(key, dict, elem):
    if key not in dict:
        dict[key] = [elem]
    else:
        dict[key].append(elem)
    return dict

def stats_cov(data1, data2):
    '''
    :param data1: np, n x d1
    :param data2: np, n x d2
    :return:
    '''
    cov = np.cov(data1, data2)  # (d_1 + d_2) x (d_1 + d_2)

    # only when data1 and data 2 are both shaped as (n,)
    corr_pear, p_value = pearsonr(data1, data2)

    # R^2
    node_num = len(data2)
    X = data2.reshape(node_num, -1)
    reg = LinearRegression().fit(X, data1)
    y_pred = reg.predict(X)
    R2 = r2_score(data1, y_pred)
    print('R-square', R2)

    result = {
        'cov': cov,
        'pearson': corr_pear,
        'pear_p_value': p_value,
        'R-square': R2
    }
    return result

def analyze_dependency(sens, adj, ypred_tst, idx_select, type='mean'):
    if type == 'mean':
        # row-normalize
        adj_norm = normalize(adj, norm='l1', axis=1)
        nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors

        # S_N(i), Y_i | S_i
        cov_nb_results = stats_cov(ypred_tst, nb_sens_ave[idx_select])
        # print('correlation between Y and neighbors (not include self)\' S:', cov_nb_results)

    return cov_nb_results


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def get_all_node_emb(model, mask, subgraph, num_node):
    # Obtain central node embs from subgraphs
    node_list = np.arange(0, num_node, 1)[mask]
    list_size = node_list.size
    z = torch.Tensor(list_size, args.hidden_size).cuda()
    group_nb = math.ceil(list_size / args.batch_size)  # num of batches
    for i in range(group_nb):
        maxx = min(list_size, (i + 1) * args.batch_size)
        minn = i * args.batch_size
        batch, index = subgraph.search(node_list[minn:maxx])
        node = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
        z[minn:maxx] = node
    return z

def get_all_node_pred(model, mask, subgraph, num_node):
    # Obtain central node prediction from subgraphs
    node_list = np.arange(0, num_node, 1)[mask]
    list_size = node_list.size
    # z = torch.Tensor(list_size, args.hidden_size).cuda()
    y_pred = torch.Tensor(list_size).cuda()
    group_nb = math.ceil(list_size / args.batch_size)  # num of batches
    for i in range(group_nb):
        maxx = min(list_size, (i + 1) * args.batch_size)
        minn = i * args.batch_size
        batch, index = subgraph.search(node_list[minn:maxx])
        node = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
        y_pred_cur = model.predict(node)
        y_pred[minn:maxx] = (y_pred_cur.squeeze() > 0).float()
    return y_pred

def evaluate(model, data, subgraph, cf_subgraph_list, labels, sens, idx_select, type='all'):
    loss_result = compute_loss(model, subgraph, cf_subgraph_list, labels, idx_select)
    if type == 'easy':
        eval_results = {'loss': loss_result['loss'], 'loss_c': loss_result['loss_c'], 'loss_s': loss_result['loss_s']}

    elif type == 'all':
        n = len(labels)
        idx_select_mask = (torch.zeros(n).scatter_(0, idx_select, 1) > 0)  # size = n, bool

        # performance
        emb = get_all_node_emb(model, idx_select_mask, subgraph, n)
        output = model.predict(emb)
        output_preds = (output.squeeze() > 0).type_as(labels)

        auc_roc = roc_auc_score(labels.cpu().numpy()[idx_select], output.detach().cpu().numpy())
        f1_s = f1_score(labels[idx_select].cpu().numpy(), output_preds.cpu().numpy())
        acc = accuracy_score(labels[idx_select].cpu().numpy(), output_preds.cpu().numpy())

        # fairness
        parity, equality = fair_metric(output_preds.cpu().numpy(), labels[idx_select].cpu().numpy(),
                                       sens[idx_select].numpy())
        # counterfactual fairness
        cf = 0.0
        for si in range(len(cf_subgraph_list)):
            cf_subgraph = cf_subgraph_list[si]
            emb_cf = get_all_node_emb(model, idx_select_mask, cf_subgraph, n)
            output_cf = model.predict(emb_cf)
            output_preds_cf = (output_cf.squeeze() > 0).type_as(labels)

            cf_si = 1 - (output_preds.eq(output_preds_cf).sum().item() / idx_select.shape[0])
            cf += cf_si
        cf /= len(cf_subgraph_list)

        eval_results = {'acc': acc, 'auc': auc_roc, 'f1': f1_s, 'parity': parity, 'equality': equality, 'cf': cf,
                        'loss': loss_result['loss'], 'loss_c': loss_result['loss_c'], 'loss_s': loss_result['loss_s']}  # counterfactual_fairness
    return eval_results

def compute_loss(model, subgraph, cf_subgraph_list, labels, idx_select):
    idx_select_mask = (torch.zeros(n).scatter_(0, idx_select, 1) > 0)
    z1 = get_all_node_emb(model, idx_select_mask, subgraph, n)

    # classifier
    c1 = model.classifier(z1)

    # Binary Cross-Entropy
    l1 = F.binary_cross_entropy_with_logits(c1, labels[idx_select].unsqueeze(1).float().to(device)) / 2

    loss_c = (1 - args.sim_coeff) * l1

    loss_sim = 0.0
    for si in range(len(cf_subgraph_list)):
        cf_subgraph = cf_subgraph_list[si]
        z2 = get_all_node_emb(model, idx_select_mask, cf_subgraph, n)
        loss_sim_si = compute_loss_sim(model, subgraph, cf_subgraph, idx_select, z1, z2)
        loss_sim += loss_sim_si
    loss_sim /= len(cf_subgraph_list)

    loss_result = {'loss_c': loss_c, 'loss_s': loss_sim, 'loss': loss_sim+loss_c}

    return loss_result


def compute_loss_sim(model, subgraph, cf_subgraph, idx_select, z1=None, z2=None):
    idx_select_mask = (torch.zeros(n).scatter_(0, idx_select, 1) > 0)
    if z1 is None:
        z1 = get_all_node_emb(model, idx_select_mask, subgraph, n)
    if z2 is None:
        z2 = get_all_node_emb(model, idx_select_mask, cf_subgraph, n)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1, p2)/2
    l2 = model.D(h2, p1)/2
    sim_loss = args.sim_coeff*(l1+l2)

    return sim_loss

def evaluate_cf(model, data, subgraph, cf_subgraph, labels, sens, idx_select, type='self'):
    # Obtain COUNTERFACTUAL central node embs from subgraphs
    node_list = np.arange(0, num_node, 1)[mask]
    list_size = node_list.size
    z = torch.Tensor(list_size, args.hidden_size).cuda()
    group_nb = math.ceil(list_size / args.batch_size)  # num of batches
    for i in range(group_nb):
        maxx = min(list_size, (i + 1) * args.batch_size)
        minn = i * args.batch_size
        if type == 'self':  # generate ground truth counterfactual subgraph: change self, fix others
            batch, index = get_cf_self(node_list[minn:maxx]) # subgraph.search(node_list[minn:maxx])
        elif type == 'neighbor':
            batch, index = get_cf_neighbor(node_list[minn:maxx])
        node = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
        z[minn:maxx] = node
    return z

def train(epochs, model, optimizer_1, optimizer_2, data, subgraph, cf_subgraph_list, idx_train, idx_val, idx_test, exp_id):
    print("start training!")
    best_loss = 100
    labels = data.y
    for epoch in range(epochs + 1):
        sim_loss = 0
        cl_loss = 0
        rep = 1
        for _ in range(rep):
            model.train()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # sample central node
            sample_size = min(args.batch_size, len(idx_train))
            sample_idx = random.sample(list(idx_train.cpu().numpy()), sample_size)  # select |batch size| central nodes

            # forward: factual subgraph
            batch, index = subgraph.search(sample_idx)
            z = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda()) # center node rep, subgraph rep

            # projector
            p1 = model.projection(z)
            # predictor
            h1 = model.prediction(p1)

            # forward: counterfactual subgraph
            sim_loss_smps = 0.0
            for si in range(len(cf_subgraph_list)):
                cf_subgraph = cf_subgraph_list[si]
                batch_cf, index_cf = cf_subgraph.search(sample_idx)
                z_cf = model(batch_cf.x.cuda(), batch_cf.edge_index.cuda(), batch_cf.batch.cuda(), index_cf.cuda())  # center node rep, subgraph rep

                # projector
                p2 = model.projection(z_cf)
                # predictor
                h2 = model.prediction(p2)

                l1 = model.D(h1, p2) / 2  # cosine similarity
                l2 = model.D(h2, p1) / 2
                sim_loss_smps += args.sim_coeff * (l1 + l2)  # similarity loss
            sim_loss_smps /= len(cf_subgraph_list)
            sim_loss += sim_loss_smps

        (sim_loss / rep).backward(retain_graph=True)
        optimizer_1.step()

        # classifier
        z = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())  # center node rep, subgraph rep
        c1 = model.classifier(z)

        # Binary Cross-Entropy
        l3 = F.binary_cross_entropy_with_logits(c1, labels[sample_idx].unsqueeze(1).float().to(device)) / 2

        cl_loss = (1 - args.sim_coeff) * (l3)
        cl_loss.backward()
        optimizer_2.step()
        loss = (sim_loss / rep + cl_loss)

        # Validation
        #model.eval()
        #eval_results_trn = evaluate(model, data, subgraph, cf_subgraph, labels, sens, idx_train)
        # eval_results_val = evaluate(model, data, subgraph, cf_subgraph, labels, sens, idx_val)
        #eval_results_tst = evaluate(model, data, subgraph, cf_subgraph, labels, sens, idx_test)
        if epoch % 100 == 0:
            model.eval()
            eval_results_trn = evaluate(model, data, subgraph, cf_subgraph_list, labels, sens, idx_train)
            eval_results_val = evaluate(model, data, subgraph, cf_subgraph_list, labels, sens, idx_val)
            print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss / rep):.4f} | train_c_loss: {cl_loss:.4f} | "
                  f"trn_loss: {eval_results_trn['loss']:.4f} |"
                  f"trn_acc: {eval_results_trn['acc']:.4f} | trn_auc_roc: {eval_results_trn['auc']:.4f} | trn_F1: {eval_results_trn['f1']:.4f} | "
                  f"trn_Parity: {eval_results_trn['parity']:.4f} | trn_Equality: {eval_results_trn['equality']:.4f} | trn_CounterFactual Fairness: {eval_results_trn['cf']:.4f} |"
                  f"val_loss: {eval_results_val['loss']:.4f} |"
                  f"val_acc: {eval_results_val['acc']:.4f} | val_auc_roc: {eval_results_val['auc']:.4f} | val_F1: {eval_results_val['f1']:.4f} | "
                  f"val_Parity: {eval_results_val['parity']:.4f} | val_Equality: {eval_results_val['equality']:.4f} | val_CounterFactual Fairness: {eval_results_val['cf']:.4f} |"
                  #f"tst_loss: {eval_results_tst['loss']:.4f} |"
                  #f"tst_acc: {eval_results_tst['acc']:.4f} | tst_auc_roc: {eval_results_tst['auc']:.4f} | tst_F1: {eval_results_tst['f1']:.4f} | "
                  #f"tst_Parity: {eval_results_tst['parity']:.4f} | tst_Equality: {eval_results_tst['equality']:.4f} | tst_CounterFactual Fairness: {eval_results_tst['cf']:.4f} |"
                  )

            val_c_loss = eval_results_val['loss_c']
            val_s_loss = eval_results_val['loss_s']
            if (val_c_loss + val_s_loss) < best_loss:
                # print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
                best_loss = val_c_loss + val_s_loss
                torch.save(model.state_dict(), f'models_save/weights_graphCF_{args.encoder}_{args.dataset}_exp'+str(exp_id)+'.pt')

    return

def test(model, adj, data, dataset, subgraph, cf_subgraph_pred_list, labels, sens, path_true_cf_data, sens_rate_list, idx_select):
    #
    model.eval()
    eval_results_orin = evaluate(model, data, subgraph, cf_subgraph_pred_list, labels, sens, idx_select)

    n = len(labels)
    idx_select_mask = (torch.zeros(n).scatter_(0, idx_select, 1) > 0)  # size = n, bool
    # performance
    emb = get_all_node_emb(model, idx_select_mask, subgraph, n)
    output = model.predict(emb)
    output_preds = (output.squeeze() > 0).type_as(labels)

    cf_score = []
    # counterfactual fairness -- true
    for i in range(len(sens_rate_list)):
        # load cf-true data
        sens_rate = sens_rate_list[i]
        post_str = dataset+'_cf_'+str(sens_rate)
        file_path = path_true_cf_data+'/'+dataset+'_cf_'+str(sens_rate)+'.pkl'

        with open(file_path, 'rb') as f:
            data_cf = pickle.load(f)['data']
            print('loaded data from: '+file_path)

        cf_subgraph = Subgraph(data_cf.x, data_cf.edge_index, ppr_path, args.subgraph_size, args.n_order)  # true
        cf_subgraph.build(postfix='true_cf'+post_str)  # true_cf_0.3

        n = len(data_cf.y)
        # performance
        emb_cf = get_all_node_emb(model, idx_select_mask, cf_subgraph, n)
        output_cf = model.predict(emb_cf)
        output_preds_cf = (output_cf.squeeze() > 0).type_as(data_cf.y)

        # compute how many labels are changed in counterfactual world
        cf_score_cur = (output_preds != output_preds_cf).sum()
        cf_score_cur = float(cf_score_cur.item()) / len(output_preds)
        cf_score.append(cf_score_cur)

    ave_cf_score = sum(cf_score) / len(cf_score)

    cf_eval_dict = {'ave_cf_score': ave_cf_score, 'cf_score': cf_score}

    # r-square
    col_ypred_s_summary = analyze_dependency(sens.cpu().numpy(), adj, output_preds.cpu().numpy(), idx_select, type='mean')

    eval_results = dict(eval_results_orin, **cf_eval_dict)  # counterfactual_fairness

    eval_results['R-square'] = col_ypred_s_summary['R-square']
    eval_results['pearson'] = col_ypred_s_summary['pearson']
    return eval_results


# the model generates counterfactual data as augmentation(don't have to be true)
def generate_cf_data(data, sens_idx, mode=1, sens_cf=None, adj_raw=None, model_path='', train='test'):
    h_dim = 32
    input_dim = data.x.shape[1]
    adj = adj_raw.tocoo()
    indices_adj = torch.LongTensor([adj.row, adj.col])
    adj = torch.sparse_coo_tensor(indices_adj, adj.data, size=(adj.shape[0], adj.shape[1])).float()

    model_DA = CFDA(h_dim, input_dim, adj.cuda()).to(device)
    if train == 'test':
        model_DA.load_state_dict(torch.load(model_path + f'weights_CFDA_{args.dataset}' + '.pt'))
        # test?
        test_model = True
        if test_model:
            S = data.x[:, sens_idx].view(-1, 1)  # n x 1
            S_agg = torch.mm(adj, S) / n  # n x 1
            S_agg_max = S_agg.max()
            S_agg_min = S_agg.min()
            s_num = 4
            S_agg_cat = torch.floor(S_agg / ((S_agg_max + 0.000001 - S_agg_min) / s_num)).long()  # n x 1

            eval_result = model_DA.test(adj.cuda(), data.x.cuda(), sens_idx, S_agg_cat.cuda())
            print(
                'loss_reconst_a: {:.4f}'.format(eval_result['loss_reconst_a'].item()),
                'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
            )
    else:
        model_DA.train_model(data.x.cuda(), adj.cuda(), sens_idx, args.dataset, model_path=model_path, lr=0.0001, weight_decay=1e-5)

    # generate cf for whole graph to achieve better efficiency
    Z_a, Z_x = model_DA.encode(data.x.cuda())
    adj_update, x_update = model_DA.pred_graph(Z_a, Z_x, sens_cf.view(-1,1).cuda())

    # hybrid
    w_hd_x = 0.99
    thresh_a = 0.9

    data_cf = data.clone()
    data_cf.x = (w_hd_x * data.x + (1 - w_hd_x) * x_update.cpu())
    data_cf.x[:, sens_idx] = sens_cf

    adj_cf = adj.to_dense().clone()
    adj_cf[adj_update > thresh_a] = 1  # to binary
    adj_cf[adj_update < 1 - thresh_a] = 0
    rate_na = (adj.to_dense() != adj_cf).sum() / (n * n)
    print('rate of A change: ', rate_na)

    edge_index_cf = adj_cf.to_sparse().indices().cpu()
    data_cf.edge_index = edge_index_cf

    return data_cf


def show_rep_distri(adj, model, data, subgraph, labels, sens, idx_select):
    model.eval()
    n = len(labels)
    idx_select_mask = (torch.zeros(n).scatter_(0, idx_select, 1) > 0)  # size = n, bool

    adj_norm = normalize(adj, norm='l1', axis=1)
    nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors
    nb_sens_ave[np.where(nb_sens_ave < 0.5)] = 0
    nb_sens_ave[np.where(nb_sens_ave >= 0.5)] = 1
    nb_sens_ave = nb_sens_ave.reshape(-1)

    emb = get_all_node_emb(model, idx_select_mask, subgraph, n).cpu().detach().numpy()
    idx_emb_0 = np.where(nb_sens_ave[idx_select] == 0)
    idx_emb_1 = np.where(nb_sens_ave[idx_select] == 1)

    fig, ax = plt.subplots()
    point_size = 8
    Zt_tsn = tsn(n_components=2).fit_transform(emb)  # m x d => m x 2
    ax.scatter(Zt_tsn[idx_emb_0, 0], Zt_tsn[idx_emb_0, 1], point_size, marker='o', color='r')  # cluster k
    ax.scatter(Zt_tsn[idx_emb_1, 0], Zt_tsn[idx_emb_1, 1], point_size, marker='o', color='b')  # cluster k
    # ax.scatter(Zt_tsn[k - num_cluster, 0], Zt_tsn[k - num_cluster, 1], centroid_size, marker='D',
      #             color=cluster_color[k])  # centroid
        # plt.xlim(-100, 100)

    # plt.show()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    #plt.savefig('./' + args.dataset + '_zt_cluster.tsne.pdf', bbox_inches='tight')
    plt.show()
    return

if __name__ == '__main__':
    experiment_type = args.experiment_type  # train, cf, test
    print('running experiment: ', experiment_type)
    data_path_root = '../'
    model_path = 'models_save/'
    adj, features, labels, idx_train_list, idx_val_list, idx_test_list, sens, sens_idx, raw_data_info = dpp.load_data(data_path_root, args.dataset)
    if raw_data_info is None:
        raw_data_info = {'adj': adj}
    # dpp.pre_analysis(adj, labels, sens)

    results_all_exp = {}
    exp_num = 3
    for exp_i in range(0, exp_num):  # repeated experiments
        idx_train = idx_train_list[exp_i]
        idx_val = idx_val_list[exp_i]
        idx_test = idx_test_list[exp_i]

        # must sorted in ascending order
        idx_train, _ = torch.sort(idx_train)
        idx_val, _ = torch.sort(idx_val)
        idx_test, _ = torch.sort(idx_test)

        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
        num_class = labels.unique().shape[0] - 1

        # preprocess the input
        n = features.shape[0]
        data = Data(x=features, edge_index=edge_index)
        data.y = labels  # n

        # ============== generate counterfactual data (ground-truth) ================
        if experiment_type == 'cf':
            sens_rate_list = [0, 0.5, 1.0]
            path_truecf_data = 'graphFair_subgraph/cf/'
            dpp.generate_cf_true(data, args.dataset, sens_rate_list, sens_idx, path_truecf_data, save_file=True, raw_data_info=raw_data_info)  # generate
            sys.exit()  # stop here

        num_node = data.x.size(0)
        device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

        # Subgraph: Setting up the subgraph extractor
        ppr_path = './graphFair_subgraph/' + args.dataset
        subgraph = Subgraph(data.x, data.edge_index, ppr_path, args.subgraph_size, args.n_order)
        subgraph.build()

        # counterfactual graph generation (may not true)
        cf_subgraph_list = []
        subgraph_load = True
        if subgraph_load:
            path_cf_ag = 'graphFair_subgraph/aug/' + f'{args.dataset}_cf_aug_' + str(0) + '.pkl'
            with open(path_cf_ag, 'rb') as f:
                data_cf = pickle.load(f)['data_cf']
                print('loaded counterfactual augmentation data from: ' + path_cf_ag)
        else:
            sens_cf = 1 - data.x[:, sens_idx]
            data_cf = generate_cf_data(data, sens_idx, mode=1, sens_cf=sens_cf, adj_raw=adj, model_path=model_path)  #
            path_cf_ag = 'graphFair_subgraph/aug/' + f'{args.dataset}_cf_aug_' + str(0) + '.pkl'
            with open(path_cf_ag, 'wb') as f:
                data_cf_save = {'data_cf': data_cf}
                pickle.dump(data_cf_save, f)
                print('saved counterfactual augmentation data in: ', path_cf_ag)

        cf_subgraph = Subgraph(data_cf.x, data_cf.edge_index, ppr_path, args.subgraph_size, args.n_order)
        cf_subgraph.build(postfix='_cf' + str(0))
        cf_subgraph_list.append(cf_subgraph)

        # add more augmentation if wanted
        subgraph_load = True
        sens_rate_list = [0.0, 1.0]
        for si in range(len(sens_rate_list)):
            sens_rate = sens_rate_list[si]
            sampled_idx = random.sample(range(n), int(sens_rate * n))
            sens_cf = torch.zeros(n)
            sens_cf[sampled_idx] = 1.
            if subgraph_load:
                path_cf_ag = 'graphFair_subgraph/aug/' + f'{args.dataset}_cf_aug_' + str(si + 1) + '.pkl'
                with open(path_cf_ag, 'rb') as f:
                    data_cf = pickle.load(f)['data_cf']
                    print('loaded counterfactual augmentation data from: ' + path_cf_ag)
            else:
                data_cf = generate_cf_data(data, sens_idx, mode=0, sens_cf=sens_cf, adj_raw=adj,
                                           model_path=model_path)  #
                path_cf_ag = 'graphFair_subgraph/aug/' + f'{args.dataset}_cf_aug_' + str(si + 1) + '.pkl'
                with open(path_cf_ag, 'wb') as f:
                    data_cf_save = {'data_cf': data_cf}
                    pickle.dump(data_cf_save, f)
                    print('saved counterfactual augmentation data in: ', path_cf_ag)
            cf_subgraph = Subgraph(data_cf.x, data_cf.edge_index, ppr_path, args.subgraph_size, args.n_order)
            cf_subgraph.build(postfix='_cf' + str(si + 1))
            cf_subgraph_list.append(cf_subgraph)

        #
        if experiment_type == 'test':
            # evaluate on the best model, we use TRUE CF graphs
            sens_rate_list = [0, 0.5, 1.0]
            path_true_cf_data = 'graphFair_subgraph/cf'  # no '/'

            model = models.GraphCF(encoder=models.Encoder(data.num_features, args.hidden_size, base_model=args.encoder),
                                   args=args, num_class=num_class).to(device)
            model.load_state_dict(torch.load(model_path +f'weights_graphCF_{args.encoder}_{args.dataset}_exp'+str(exp_i)+'.pt'))

            #show_rep_distri(adj, model, data, subgraph, labels, sens, idx_test)
            #sys.exit()

            eval_results = test(model, adj, data, args.dataset, subgraph, cf_subgraph_list, labels, sens, path_true_cf_data, sens_rate_list, idx_test)

            results_all_exp = add_list_in_dict('Accuracy', results_all_exp, eval_results['acc'])
            results_all_exp = add_list_in_dict('F1-score', results_all_exp, eval_results['f1'])
            results_all_exp = add_list_in_dict('auc_roc', results_all_exp, eval_results['auc'])
            results_all_exp = add_list_in_dict('Equality', results_all_exp, eval_results['equality'])
            results_all_exp = add_list_in_dict('Parity', results_all_exp, eval_results['parity'])
            results_all_exp = add_list_in_dict('ave_cf_score', results_all_exp, eval_results['ave_cf_score'])
            results_all_exp = add_list_in_dict('CounterFactual Fairness', results_all_exp, eval_results['cf'])
            results_all_exp = add_list_in_dict('R-square', results_all_exp, eval_results['R-square'])

            print('=========================== Exp ', str(exp_i),' ==================================')
            for k in eval_results:
                if isinstance(eval_results[k], list):
                    print(k, ": ", eval_results[k])
                else:
                    print(k, f": {eval_results[k]:.4f}")

            if exp_i <= exp_num - 1:
                continue
            else:  # last exp
                print('============================= Overall =============================================')
                for k in results_all_exp:
                    results_all_exp[k] = np.array(results_all_exp[k])
                    print(k, f": mean: {np.mean(results_all_exp[k]):.4f} | std: {np.std(results_all_exp[k]):.4f}")
                sys.exit()

        # Setting up the model and optimizer
        model = models.GraphCF(encoder=models.Encoder(data.num_features, args.hidden_size, base_model=args.encoder), args=args, num_class=num_class).to(device)
        par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(
            model.fc3.parameters()) + list(model.fc4.parameters())
        par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
        optimizer_1 = torch.optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_2 = torch.optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)

        # cuda
        if args.cuda:
            model = model.to(device)

        train(args.epochs, model, optimizer_1, optimizer_2, data, subgraph, cf_subgraph_list, idx_train, idx_val, idx_test, exp_i)

        # ========= test all ===========
        # evaluate on the best model, we use TRUE CF graphs
        sens_rate_list = [0, 0.5, 1.0]
        path_true_cf_data = 'graphFair_subgraph/cf'  # no '/'

        model = models.GraphCF(encoder=models.Encoder(data.num_features, args.hidden_size, base_model=args.encoder), args=args, num_class=num_class).to(device)
        model.load_state_dict(torch.load(model_path + f'weights_graphCF_{args.encoder}_{args.dataset}_exp'+str(exp_i)+'.pt'))
        eval_results = test(model, adj, data, args.dataset, subgraph, cf_subgraph_list, labels, sens, path_true_cf_data, sens_rate_list, idx_test)

        results_all_exp = add_list_in_dict('Accuracy', results_all_exp, eval_results['acc'])
        results_all_exp = add_list_in_dict('F1-score', results_all_exp, eval_results['f1'])
        results_all_exp = add_list_in_dict('auc_roc', results_all_exp, eval_results['auc'])
        results_all_exp = add_list_in_dict('Equality', results_all_exp, eval_results['equality'])
        results_all_exp = add_list_in_dict('Parity', results_all_exp, eval_results['parity'])
        results_all_exp = add_list_in_dict('ave_cf_score', results_all_exp, eval_results['ave_cf_score'])
        results_all_exp = add_list_in_dict('CounterFactual Fairness', results_all_exp, eval_results['cf'])
        results_all_exp = add_list_in_dict('R-square', results_all_exp, eval_results['R-square'])

        print('=========================== Exp ', str(exp_i), ' ==================================')
        for k in eval_results:
            if isinstance(eval_results[k], list):
                print(k, ": ", eval_results[k])
            else:
                print(k, f": {eval_results[k]:.4f}")

    print('============================= Overall =============================================')
    for k in results_all_exp:
        results_all_exp[k] = np.array(results_all_exp[k])
        print(k, f": mean: {np.mean(results_all_exp[k]):.4f} | std: {np.std(results_all_exp[k]):.4f}")









