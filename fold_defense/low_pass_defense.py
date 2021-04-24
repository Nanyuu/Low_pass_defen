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
data_info = F_Info.C_per_info(base_adj, base_feat, label, idx_train, idx_val, idx_test, opts)

adj_nor = F_Nor.normalize_adj_sym(base_adj)
feat_nor = F_Nor.normalize_feat(base_feat)

# 读取模型
model = t.load("{}/{}/GCN.t7".format(opt.model_path, opt.dataset))['model']
label_tensor = t.from_numpy(label_not_one_hot).long()

# load_modified_adj
adj_after_attack = sp.load_npz("../fold_attack/adj_After_Attack/MetaAttack/test0/modified_adj_0.npz").A
adj_after_attack_nor = F_Nor.normalize_adj_sym(adj_after_attack)
attack_test_node = np.load(r"D:\Learning Project\python_project\low_pass_def\fold_attack\adj_After_Attack\MetaAttack\test0\idx_test.npy")


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.gc1.register_forward_hook(get_activation('gc1'))

model.eval()
adj_nor_t = t.from_numpy(adj_nor).float().cuda()
feat_nor_t = t.from_numpy(feat_nor).float().cuda()


output = model(feat_nor_t, adj_nor_t)

print(output[0])

sim_adj_h0 = simGraph_init(base_feat, p_neighbor_num=20, p_layer_id=0)

modified_adj_h0 = low_pass_adj_sym(base_adj, sim_adj_h0, base_feat, p_filter_value=1)

modified_adj_h0_t = t.from_numpy(modified_adj_h0).float().cuda()
gc1 = model.gc1
h1_gc = gc1(feat_nor_t, modified_adj_h0_t)
h1_relu = t.relu(h1_gc)

sim_adj_h1 = simGraph_init(h1_relu.detach().cpu().numpy(), p_neighbor_num=20, p_layer_id=1)

modified_adj_h1 = low_pass_adj_sym(base_adj, sim_adj_h1, h1_relu.detach().cpu().numpy(), p_filter_value=1)
modified_adj_h1_t = t.from_numpy(modified_adj_h1).float().cuda()
gc2 = model.gc2
h2_gc = gc2(h1_relu, modified_adj_h1_t)
h2_softmax = F.log_softmax(h2_gc, dim=1)

test_Acc_modified = F_accuracy(h2_softmax[idx_test], label_tensor[idx_test])
test_Acc_origin = F_accuracy(output[idx_test], label_tensor[idx_test])











