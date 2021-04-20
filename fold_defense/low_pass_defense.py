from config import opts
from fold_data import dataset
from fold_util import F_Info
import torch as t
from fold_util import F_Normalize as F_Nor
from fold_defense import simGraph_init
from fold_defense import low_pass_adj
from sklearn.metrics.pairwise import euclidean_distances

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

sim_adj_l0 = simGraph_init(base_feat, p_neighbor_num=20, p_layer_id=0)

modified_adj = low_pass_adj(base_adj, sim_adj_l0, base_feat, p_filter_value=1)

print(modified_adj)











