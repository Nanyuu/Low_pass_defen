import numpy as np
import scipy.sparse as sp
import torch as t
from fold_util import F_Normalize
from deeprobust.graph.defense import GCN


def random_select_train_test(p_labels: np.ndarray, p_pick_num_each, p_rand_seed=0):
    """
    在给定label中各类选择一定数量的节点作为训练和验证，其余用作测试
    p_labels : label not one hot
    p_pick_num_each: each
    """
    total_class_num = p_labels.max(axis=0) + 1
    res_train = np.array([])
    res_val = np.array([])
    res_test = np.arange(p_labels.shape[0])
    for i in range(total_class_num):
        candidates = np.where(p_labels == i)[0]
        np.random.seed(p_rand_seed)
        chosen = np.random.choice(candidates, p_pick_num_each * 2, replace=False)
        res_train = np.union1d(res_train, chosen[:p_pick_num_each])
        res_val = np.union1d(res_val, chosen[p_pick_num_each:])

    res_test = np.setdiff1d(res_test, res_train)
    res_test = np.setdiff1d(res_test, res_val)
    return res_train, res_val, res_test

def test_given_adj(p_adj:sp.csr_matrix, p_feat:np.ndarray, p_label:np.ndarray, p_train_idx,p_val_idx,p_test_idx):
    """
    Parameters:
        p_adj : csr_matrix


    """
    use_gpu = t.cuda.is_available()
    adj = p_adj.A
    features = p_feat
    labels = p_label

    # surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=True,
                    with_bias=True, device='cuda:0').cuda()
    # train model
    surrogate.fit(features, adj, labels, p_train_idx, p_val_idx, patience=30)

    # test_model
    surrogate.test(p_test_idx)

    return surrogate

