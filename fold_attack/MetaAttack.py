import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from config import opts
from attack_util import random_select_train_test

data = Dataset(root='{}/'.format(opts.data_path_graphRobust), name='cora')
adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# idx_unlabeled = np.union1d(idx_val, idx_test)
'''edit 2021/4/22 random_choose 20 node from each class as the train and validation samples '''
idx_train, idx_val, idx_test = random_select_train_test(labels, 20, 4)
idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                     nhid=16, dropout=0, with_relu=False, with_bias=False, device='cuda:0').cuda()
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, verbose=True)
# Setup Attack Model
model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                       attack_structure=True, attack_features=False, device='cuda:0', lambda_=0).cuda()
# Attack
n_perturbations = (idx_train.shape[0] + idx_val.shape[0])*2
model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=n_perturbations, ll_constraint=False)
modified_adj = model.modified_adj
