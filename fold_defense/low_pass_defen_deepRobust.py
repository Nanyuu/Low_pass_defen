from config import opts
from fold_data import dataset
from fold_util import F_Info
import torch as t
from fold_util import F_Normalize as F_Nor
from fold_defense import simGraph_init
from fold_defense import low_pass_adj_sym
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
from train import F_accuracy
import numpy as np
import scipy.sparse as sp
from deeprobust.graph import utils

model_path = "../checkpoint"

opt = opts()
opt.data_path = r"../fold_data/Data/Planetoid"
opt.model_path = "../checkpoint"
opt.dataset = 'cora'
opt.model = 'GCN'
neighbor_num = 0

# 读取数据集
data_load = dataset.c_dataset_loader(opt.dataset, opt.data_path)
base_adj, base_feat,_,_,_,_  = data_load.process_data()
if opt.dataset == 'cora':
    attack_info = t.load("../fold_attack/GCN/Nettack/attack_info_1".format(opt.dataset))
else:
    attack_info = t.load("../fold_attack/GCN/Nettack/attack_info_{}_11".format(opt.dataset))


adj_per = attack_info['adj_per'][15].A
model = attack_info['surrogate']

idx_train = attack_info['idx_train']
idx_val = attack_info['idx_val']
idx_test = attack_info['idx_test']

labels = model.labels

# check output
adj_per_nor_t = t.from_numpy(adj_per).float().cuda()

hidden = t.from_numpy(base_feat).float().cuda()
hidden = adj_per_nor_t @ hidden @ model.gc1.weight
if model.with_bias:
    hidden = hidden + model.gc1.bias
if model.with_relu:
    hidden = F.relu(hidden)
hidden = adj_per_nor_t @ hidden @ model.gc2.weight
if model.with_bias:
    hidden = hidden + model.gc2.bias

output = F.log_softmax(hidden, dim=1)
acc_after_attack = utils.accuracy(output[idx_test], labels[idx_test])

# do the low_pass
sim_adj_h0 = simGraph_init(base_feat, p_neighbor_num=neighbor_num, p_layer_id=0)

modified_adj_h0 = low_pass_adj_sym(adj_per, sim_adj_h0, base_feat, p_filter_value=1)

modified_adj_h0_t = t.from_numpy(modified_adj_h0).float().cuda()

hidden_low_pass_0 = t.from_numpy(base_feat).float().cuda()
hidden_low_pass_h0 = modified_adj_h0_t @ hidden_low_pass_0 @ model.gc1.weight
if model.with_bias:
    hidden_low_pass_h0 = hidden_low_pass_h0 + model.gc1.bias
if model.with_relu:
    hidden_low_pass_h0 = F.relu(hidden_low_pass_h0)


sim_adj_h1 = simGraph_init(hidden_low_pass_h0.detach().cpu().numpy(), p_neighbor_num=neighbor_num, p_layer_id=1)

modified_adj_h1 = low_pass_adj_sym(base_adj, sim_adj_h1, hidden_low_pass_h0.detach().cpu().numpy(), p_filter_value=1)
modified_adj_h1_t = t.from_numpy(modified_adj_h1).float().cuda()
hidden_low_pass_h1 = modified_adj_h1_t @ hidden_low_pass_h0 @ model.gc2.weight
if model.with_bias:
    hidden_low_pass_h1 = hidden_low_pass_h1 + model.gc2.bias

h2_softmax = F.log_softmax(hidden_low_pass_h1, dim=1)

acc_after_low_pass = utils.accuracy(h2_softmax[idx_test], labels[idx_test])

