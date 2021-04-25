import numpy as np
import torch as t
import scipy.sparse as sp
from attack_util import test_given_adj
from config import opts
from fold_data import dataset
from fold_util import F_Info

per_adj = sp.load_npz("pertAdj.npz")
idx_info = t.load("idx_info")
opt = opts()
rand_seed = 10

data_load = dataset.c_dataset_loader(opt.dataset, ".{}".format(opt.data_path))
adj, features, label, _,_,_ = data_load.process_data()
labels = F_Info.F_one_hot_to_label(label)

test_given_adj(per_adj, features, labels, idx_info['train'],idx_info['val'],idx_info['test'])


