import torch as t
import numpy as np
import scipy.sparse as sp
import networkx as nx
import time

from fold_data import c_dataset_loader
from config import opts
from train import train
from test import test
# from train import Train_PyG
from fold_util import F_Normalize as F_Nor
from fold_util import F_Perturbation as F_Per
from fold_util import F_Info as F_Info


# def opt_test(p_opt:opts):
#     p_opt.is_adj_normed = True
#     return True


if __name__ == '__main__':
    opts.dataset = 'cora'
    data_loader = c_dataset_loader(opts.dataset,opts.data_path)
    adj, feat, label, idx_train, idx_val, idx_test = data_loader.process_data()
    opt = opts()

    train(adj, feat, label, idx_train, idx_val, opts.feature_Nor)
    # Train_PyG(adj, feat, label, idx_train, idx_val)

    # 训练
    # adj_f_np, feat_f_np = F_Per.Per_add_fake_node(adj, feat, backdoor_index=opt.backdoor_node_idx, node_num=opt.fake_node_num)
    # train(adj_f_np, feat_f_np, label, idx_train, idx_val, opts.feature_Nor)

    test(adj, feat, label, idx_test, normalize_feat=True)
    # per_info = F_Info.C_per_info(adj, feat, label, idx_train, idx_val, idx_test)

    # # opt测试
    # opt_test(opt)


