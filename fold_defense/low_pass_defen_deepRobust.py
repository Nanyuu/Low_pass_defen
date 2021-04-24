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

# 读取数据集
data_load = dataset.c_dataset_loader(opt.dataset, opt.data_path)
base_adj, base_feat, label, idx_train, idx_val, idx_test = data_load.process_data()
label_not_one_hot = F_Info.F_one_hot_to_label(label)

# load_metAttack
attack_name = "MetAttack"
model_name = "GCN"

attack_base_root = "../fold_attack/adj_After_Attack/{}/{}".format(attack_name,model_name)
attack_info = t.load("{}/attack_info_1".format(attack_base_root))

adj_nor = F_Nor.normalize_adj_sym(base_adj)


# load_modified_adj
adj_after_attack = attack_info['modified_adj'].A
adj_after_attack_nor = F_Nor.normalize_adj_sym(adj_after_attack)
idx_test_attack = attack_info['idx_test']

# check output
adj_after_attack_nor_t = t.from_numpy(adj_after_attack_nor).float().cuda()

hidden = t.from_numpy(base_feat).float().cuda()
hidden = adj_after_attack_nor_t @ hidden @ attack_info['model_weight'][0]
hidden = adj_after_attack_nor_t @ hidden @ attack_info['model_weight'][1]

output = F.log_softmax(hidden, dim=1)
acc_after_attack = utils.accuracy(output[idx_test_attack], t.from_numpy(label_not_one_hot[idx_test_attack]).cuda())

# do the low_pass

sim_adj_h0 = simGraph_init(base_feat, p_neighbor_num=20, p_layer_id=0)

modified_adj_h0 = low_pass_adj_sym(adj_after_attack, sim_adj_h0, base_feat, p_filter_value=1)

modified_adj_h0_t = t.from_numpy(modified_adj_h0).float().cuda()

hidden_low_pass_0 = t.from_numpy(base_feat).float().cuda()
hidden_low_pass_h0 = modified_adj_h0_t @ hidden_low_pass_0 @ attack_info['model_weight'][0]

sim_adj_h1 = simGraph_init(hidden_low_pass_h0.detach().cpu().numpy(), p_neighbor_num=20, p_layer_id=1)

modified_adj_h1 = low_pass_adj_sym(base_adj, sim_adj_h1, hidden_low_pass_h0.detach().cpu().numpy(), p_filter_value=1)
modified_adj_h1_t = t.from_numpy(modified_adj_h1).float().cuda()
hidden_low_pass_h1 = modified_adj_h1_t @ hidden_low_pass_h0 @ attack_info['model_weight'][1]

h2_softmax = F.log_softmax(hidden_low_pass_h1, dim=1)

acc_after_low_pass = utils.accuracy(h2_softmax[idx_test_attack], t.from_numpy(label_not_one_hot[idx_test_attack]).cuda())
