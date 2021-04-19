import numpy as np


def find_neighbor_idx(p_adj:np.ndarray, p_hops:int, p_node_idx:int):
    """
    寻找K阶邻居
    :param p_adj: 邻接矩阵
    :param p_hops: 阶数
    :param p_node_set: 所需要寻找邻居的节点
    :return: 邻居矩阵 np.ndarray
    """

    neighbor_matrix = np.array([p_node_idx], dtype=np.int16)
    for lp_hop in range(p_hops):
    # 一共hop阶
        for lp_node_id in range(neighbor_matrix.shape[0]):
        # 一共需要为x个节点找neighbor
            temp_neighbors = np.where(p_adj[neighbor_matrix[lp_node_id]]!=0)[0]
            for idx in temp_neighbors:
                if neighbor_matrix.__contains__(idx):
                    continue
                else:
                    neighbor_matrix = np.append(neighbor_matrix, idx)

    return np.sort(neighbor_matrix)

def find_neighbor_np(p_adj: np.ndarray, p_hops: int, p_node_idx_np: np.ndarray):
    """
    根据idx_np找到对应的邻居节点
    :param p_adj: 邻接矩阵
    :param p_hops: 邻居阶数
    :param p_node_idx_np:  需要找邻居的节点集合
    :return:
    """
    neighbor_matrix = np.copy(p_node_idx_np)

    for lp_hop in range(p_hops):
    # 一共有p_hops阶个邻居
        for lp_node_id in range(neighbor_matrix.shape[0]):
            temp_neighbors = np.where(p_adj[neighbor_matrix[lp_node_id]] != 0)[0]
            for idx in temp_neighbors:
                if neighbor_matrix.__contains__(idx):
                    continue
                else:
                    neighbor_matrix = np.append(neighbor_matrix, idx)

    return np.sort(neighbor_matrix)

def construct_sub_graph(p_adj, p_feat, p_node_set:np.ndarray):
    """
    根据节点集合构造子图
    :param p_adj:邻接矩阵
    :param p_feat:特征矩阵
    :param p_node_set:节点集合
    :return: 返回未标准化的邻接矩阵，度矩阵，特征矩阵
    """

    # temp_adj = np.copy(p_adj)
    # 创建映射词典
    proj_o_to_s = {}  # origin to sub
    proj_s_to_o = {}  # sub to origin
    for lp_set_id in range(p_node_set.shape[0]):
        proj_s_to_o[lp_set_id] = p_node_set[lp_set_id]
        proj_o_to_s[p_node_set[lp_set_id]] = lp_set_id

    # 初始化0矩阵
    sub_adj = np.zeros([p_node_set.shape[0], p_node_set.shape[0]])

    # 构造邻接矩阵adj
    for lp_node_i in p_node_set:
        for lp_node_j in p_node_set:
            if p_adj[lp_node_i, lp_node_j] == 1:
                sub_idx_i = proj_o_to_s[lp_node_i]
                sub_idx_j = proj_o_to_s[lp_node_j]
                sub_adj[sub_idx_i, sub_idx_j] = 1

    # 添加对角阵
    # temp_adj = temp_adj + np.eye(temp_adj.shape[0])

    # 根据adj求取节点的度
    sub_d = np.diag(p_adj[p_node_set].sum(1))

    # 子图特征矩阵
    sub_feat = np.copy(p_feat[p_node_set])

    # 返回子图邻接矩阵，度矩阵，特征矩阵
    return sub_adj, sub_d, sub_feat





