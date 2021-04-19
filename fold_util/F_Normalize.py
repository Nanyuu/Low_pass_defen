import numpy as np
import torch as t
import scipy.sparse as sp


def normalize_feat(feat_np: np.ndarray) -> np.ndarray:
    """
    特征标准化
    :param feat_np:
    :return:
    """
    feat_csr = sp.csr_matrix(feat_np)  # NdArray -> Csr_matrix
    rowsum = np.array(feat_csr.sum(1))  # 统计每行有多少个为1的元素
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()  # 取逆，压平
    r_inv[np.isinf(r_inv)] = 0.  # 无穷的元素置0
    r_mat_inv = sp.diags(r_inv)  # 化为对角阵
    feat_nor_np = r_mat_inv.dot(feat_csr)  # 对角阵乘原矩阵 实现标准化
    return feat_nor_np.A  # 返回标准化后的矩阵


def normalize_adj_degree(adj_np: np.ndarray) -> np.ndarray:
    """
    获取 D^-0.5
    :param adj: Adjacency Matrix
    :return: D^-0.5
    """
    adj_csr = sp.csr_matrix(adj_np)
    # adj_eye_csr = adj_csr    # 邻接矩阵对角线置1
    rowsum = np.array(adj_csr.sum(1))  # 统计每行中的元素个数
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.  # 无穷处置零
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # 化为对角阵
    degree_np = r_mat_inv_sqrt.A
    return degree_np


def get_edge_list(adj_np: np.ndarray) -> np.ndarray:
    dim = adj_np.shape[0]
    edge_num = np.sum(adj_np)
    edge_list = np.zeros((2, int(edge_num / 2)), dtype=int)
    adj_coo = sp.coo_matrix(adj_np)
    k = 0
    for ii in range(int(adj_coo.nnz / 2)):
        edge_list[0, ii] = adj_coo.row[ii]
        edge_list[1, ii] = adj_coo.col[ii]
    return edge_list


def normalize_adj(adj_np: np.ndarray) -> np.ndarray:
    """输入邻接矩阵 输出D^-0.5 * A * D^-0.5"""
    adj_csr = sp.csr_matrix(adj_np)

    # 加上对角阵
    adj_eye_csr = adj_csr + sp.eye(adj_np.shape[1])
    adj_eye_np = adj_eye_csr.A

    # 计算D Degree 度矩阵
    row_sum = np.array(adj_eye_np.sum(1))
    # 计算D D^-0.5
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    mx_dot = adj_eye_csr.dot(r_mat_inv_sqrt)  # A*D^0.5
    mx_dot_trans = mx_dot.transpose()  # 转置  (A*D^0.5)^T
    mx_dot_trans_dot = mx_dot_trans.dot(r_mat_inv_sqrt)  # (A*D^0.5)^T * D^0.5
    mx_dot_trans_dot_np = mx_dot_trans_dot.A

    return mx_dot_trans_dot_np


def nor_sub_adj(p_sub_adj, p_sub_d):
    # 计算d^-0.5
    d_inv_sqrt = np.power(np.diagonal(p_sub_d), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt_sp = sp.diags(d_inv_sqrt)

    # adj_csr
    adj_eye_csr = sp.csr_matrix(p_sub_adj)

    # D^-0.5 * A * D^-0.5
    mx_dot = adj_eye_csr.dot(d_inv_sqrt_sp)  # A*D^0.5
    mx_dot_trans = mx_dot.transpose()  # 转置
    mx_dot_trans_dot = mx_dot_trans.dot(d_inv_sqrt_sp)  # (A*D^0.5)^T * D^0.5
    mx_dot_trans_dot_np = mx_dot_trans_dot.A

    return mx_dot_trans_dot_np

def nor_sub_adj_eye(p_sub_adj, p_sub_d):
    """
    额外对邻接矩阵增加添加对角阵的工作 A = A+I 后 再标准化
    :param p_sub_adj:
    :param p_sub_d:
    :return:
    """
    # 计算d^-0.5
    d_inv_sqrt = np.power(np.diagonal(p_sub_d), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt_sp = sp.diags(d_inv_sqrt)

    # adj_csr
    adj_eye_csr = sp.csr_matrix(p_sub_adj + np.eye(p_sub_adj.shape[0]))

    # D^-0.5 * A * D^-0.5
    mx_dot = adj_eye_csr.dot(d_inv_sqrt_sp)  # A*D^0.5
    mx_dot_trans = mx_dot.transpose()  # 转置
    mx_dot_trans_dot = mx_dot_trans.dot(d_inv_sqrt_sp)  # (A*D^0.5)^T * D^0.5
    mx_dot_trans_dot_np = mx_dot_trans_dot.A

    return mx_dot_trans_dot_np
