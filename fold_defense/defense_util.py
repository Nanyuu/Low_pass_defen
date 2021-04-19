import numpy as np
import torch as t
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from fold_util import F_Normalize as F_Nor
import scipy.sparse as sp


# 通过相似度计算Sim_Graph
def simGraph_init(p_feat:np.ndarray, p_neighbor_num: int, p_layer_id: int):
    """
    generate simGraph using the l0 (input feature X)
    :param p_layer_id: layer_id
                        for input layer(X), treat every non-zero features as 1
                        for hidden layer(l1), use the layer output
    :param p_feat: normalized or un-normalized features
    :param p_neighbor_num: set the neighbor number
    :return: sim_mat (ndArray)
    """
    # using cosine_similarity
    temp_feat = np.copy(p_feat)

    if p_layer_id == 0:
        temp_feat[temp_feat != 0] = 1
    sim_mat = cosine_similarity(temp_feat)

    sim_mat[(np.arange(len(sim_mat)), np.arange(len(sim_mat)))] = 0

    for i in range(len(sim_mat)):
        arg_sort = np.argsort(sim_mat[i])
        sim_mat[i, arg_sort[:-p_neighbor_num]] = 0

    sim_mat[sim_mat != 0] = 1

    return sim_mat

# 给定原始邻接矩阵和相似度后的矩阵
# 计算特征矩阵的形式
# Step：
# 1、对于每个节点，根据邻接矩阵计算欧氏距离 l2
# 2、对于距离进行如下操作： 1/max(1,distance(u,v))
# 3、对于得到的距离相加和，除以度(d), 多余的部分使用SimAdj的邻接矩阵数值补充
def modified_adj(p_adj_original:np.ndarray, p_adj_sim:np.ndarray,p_feat:np.ndarray, p_filter_value):
    """
    compute the modified_Adj using the distance between two nodes in A
    :param p_adj_original: original adj (perturbed)
    :param p_adj_sim:
    :param p_feat:
    :param p_filter_value:
    :return:
    """
    nor_adj = F_Nor.normalize_adj(p_adj_original)
    nor_adj_sim = F_Nor.normalize_adj(p_adj_sim)
    res_adj = np.zeros([nor_adj.shape[0],nor_adj.shape[0]])
    for i in range(nor_adj.shape[0]):
        # 对每个连接的节点特征计算欧氏距离
        linked_nodes = np.where(nor_adj[i]!=0)[0]
        temp_degree = linked_nodes.shape[0]
        temp_score = 0

        for j in linked_nodes:
            temp_dis = euclidean_distances(p_feat[i], p_feat[j])
            temp_score = temp_score + p_filter_value/max(p_filter_value, temp_dis)
            res_adj[i][j] = temp_score*nor_adj[i][j]

        score_for_i = temp_score/temp_degree
        score_for_sim = 1-score_for_i
        res_adj[i] = res_adj[i] + nor_adj_sim[i]*score_for_sim

    return res_adj





