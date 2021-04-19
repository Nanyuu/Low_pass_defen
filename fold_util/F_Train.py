import numpy as np
import torch as t
import scipy.sparse as sp


def Train_for_attack_single_node(adj, feat, label, idx_train, idx_val, is_normalize=False):
    """
    训练攻击单一目标节点，每攻击一次 就retrain一次
    :param adj:
    :param feat:
    :param label:
    :param idx_train:
    :param idx_val:
    :param is_normalize:
    :return:
    """
