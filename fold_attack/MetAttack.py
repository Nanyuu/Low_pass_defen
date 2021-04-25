import numpy as np
import scipy.sparse as sp
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from config import opts
from attack_util import random_select_train_test
from fold_data import dataset
from config import opts
from fold_util import F_Info
import torch as t
import os

opt = opts()
rand_seed = 10

data_load = dataset.c_dataset_loader(opt.dataset, ".{}".format(opt.data_path))
adj, features, label, _,_,_ = data_load.process_data()
labels = F_Info.F_one_hot_to_label(label)

# data = Dataset(root='{}/'.format(opts.data_path_graphRobust), name='cora')
# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# idx_unlabeled = np.union1d(idx_val, idx_test)
'''edit 2021/4/22 random_choose 20 node from each class as the train and validation samples '''
idx_train, idx_val, idx_test = random_select_train_test(labels, 20, p_rand_seed=rand_seed)
idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                     nhid=16, dropout=0, with_relu=False, with_bias=False, device='cuda').cuda()
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, verbose=True)
# Setup Attack Model
model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                       attack_structure=True, attack_features=False, device='cuda', lambda_=0)
# Attack
n_perturbations = (idx_train.shape[0] + idx_val.shape[0])*2
model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=n_perturbations, ll_constraint=False)
modified_adj = model.modified_adj

test_id = rand_seed
attack_name = "MetAttack"
opt.is_save = True
if opt.is_save:


    base_root = "./adj_After_Attack/{}/GCN/".format(attack_name)
    if not os.path.exists(base_root):
        os.mkdir(base_root)
    collect_info = {}
    collect_info['idx_train'] = idx_train
    collect_info['idx_test'] = idx_test
    collect_info['idx_val'] = idx_val
    collect_info['acc_after_attack'] = model.acc_after_attack
    collect_info['rand_seed'] = rand_seed
    collect_info['modified_adj'] = sp.csr_matrix(modified_adj.detach().cpu().numpy())
    collect_info['model_weight'] = model.weights
    collect_info['with_bias'] = False
    collect_info['with_relu'] = False
    collect_info['n_perturbations'] = n_perturbations

    t.save(collect_info, "{}/attack_info_{}".format(base_root, test_id))





