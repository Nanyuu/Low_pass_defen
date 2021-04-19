import numpy as np
import scipy.sparse as sp
import torch as t

from config import opts
from fold_util import F_Info

def Per_add_fake_node(adj_np: np.ndarray, feat_np: np.ndarray, backdoor_index: int, node_num=1):
    """
    对后门节点添加无特征的假节点
    :param adj_np:
    :param target_index:
    :param node_num:
    :return: adj_fake_np, feat_fake_np
    """
    # adj_sp = sp.csr_matrix(adj_np)
    # adj_fake_sp = sp.csr_matrix((adj_sp.data, adj_sp.indices, adj_sp.indptr), shape=(adj_sp.shape[0]+1, adj_sp.shape[0]+1))
    shape = adj_np.shape[0]
    # 创建矩阵
    adj_fake_np = np.zeros((shape + node_num, shape + node_num))
    adj_fake_np[:shape, :shape] = adj_np

    # 添加节点
    for i in range(node_num):
        adj_fake_np[backdoor_index, shape + i] = 1
        adj_fake_np[shape + i, backdoor_index] = 1

    # 添加特征
    feat_num = feat_np.shape[1]
    feat_fake_np = np.zeros((shape + node_num, feat_num))
    feat_fake_np[:shape, :feat_num] = feat_np
    return adj_fake_np, feat_fake_np


def Per_add_feat_single_node(feat_np: np.ndarray, idx_node: int, idx_feat: int) -> np.ndarray:
    """将index_node的第idx_feat位特征添加入feat_np"""
    temp_feat_np = np.copy(feat_np)
    temp_feat_np[idx_node][idx_feat] = 1
    return temp_feat_np


def Per_add_adj(adj_np, backdoor_idx, target_idx) -> np.ndarray:
    """将adj中的backdoor_node和target_idx 连接起来"""
    temp_adj_np = np.copy(adj_np)
    temp_adj_np[backdoor_idx][target_idx] = 1
    temp_adj_np[target_idx][backdoor_idx] = 1
    return temp_adj_np


def Per_add_fake_feat_based_on_grad(p_fake_feat_np: np.ndarray, p_feat_grad_np: np.ndarray,
                                    p_fake_node_idx: np.ndarray):
    """
    通过求得的梯度，排序，计算最大/最小值，将其置为1 返回被污染的特征矩阵feat
    :param p_fake_feat_np:
    :param p_feat_grad_fake_np:
    :param p_fake_node_idx:
    :return:                        fake_feat_mat
    """
    temp_fake_feat_np = np.copy(p_fake_feat_np)
    grad_feat_fake_node = p_feat_grad_np[p_fake_node_idx]
    if opts.is_multi_feat_decay:
        # 如果用到多特征衰减，则将已置1部分的特征乘以对应特征衰减
        temp_fake_node_feat = temp_fake_feat_np[p_fake_node_idx]
        for ii in range(grad_feat_fake_node.shape[0]):
            # 获取已经置1的个数
            already_set_num = np.where(temp_fake_node_feat[ii] == 1)[0].shape[0]
            # 将对应特征乘以特征衰减
            grad_feat_fake_node[ii] = grad_feat_fake_node[ii] * opts.multi_feat_decay ** already_set_num




    grad_feat_fake_node_sort = np.unique(np.sort(grad_feat_fake_node.flatten()))  # 去除重复元素 从小到大排序
    for ii in range(50):
        flag = 0
        temp_max = grad_feat_fake_node_sort[ii]
        temp_max_index_np = np.where(grad_feat_fake_node.flatten() == temp_max)[0]
        for jj in range(temp_max_index_np.shape[0]):
            # 由flatten之后的index转原始index
            temp_node_idx = int(temp_max_index_np[jj] / grad_feat_fake_node.shape[1])
            temp_feat_idx = int(temp_max_index_np[jj] % grad_feat_fake_node.shape[1])
            # 检查是否已经置1
            if temp_fake_feat_np[temp_node_idx + temp_fake_feat_np.shape[0] - p_fake_node_idx.shape[0], temp_feat_idx] == 1:
                continue
            else:
                print("选择的index为:{}节点的第{}维特征".format(temp_node_idx+temp_fake_feat_np.shape[0]- p_fake_node_idx.shape[0], temp_feat_idx))
                temp_fake_feat_np[temp_node_idx + temp_fake_feat_np.shape[0] - p_fake_node_idx.shape[0], temp_feat_idx] = 1
                flag = 1
                break
        if flag == 1:
            print("此时的梯度大小为{}".format(grad_feat_fake_node_sort[ii]))
            break
    print("完成")
    return temp_fake_feat_np


def Per_add_fake_feat_based_on_grad_for_Class_Node(grad_np: np.ndarray, p_feat_fake: np.ndarray,
                                                   p_fake_node_idx_np: np.ndarray):
    """
    根据输入的特征,统计求和后排序得到最后的那一维特征。
    直接加权求和， 结果不理想
    方法1：直接统计求和      方法2： 统计个数
    :param p_fake_node_idx_np: 假节点的index
    :param p_feat_fake:  假特征
    :param grad_np: 特征的梯度 [target_node_num , fake_node_num , feat_num]
    :return:
    """
    target_node_num = grad_np.shape[0]
    fake_node_num = p_fake_node_idx_np.shape[0]
    feat_num = grad_np.shape[2]
    grad_sum = np.zeros([fake_node_num, feat_num])
    feat_fake = np.copy(p_feat_fake)

    for ii in range(target_node_num):
        # 遍历所有节点，求和
        for jj in range(fake_node_num):
            grad_sum[jj] = grad_sum[jj] + grad_np[ii][jj]

    # 根据求和的结果， 对特征grad进行排序
    grad_feat_node_sort = np.unique(np.sort(grad_sum.flatten()))  # 去除重复元素 从小到大排序

    # 记录当前是否特征改变成功
    flag = 0

    for ii in range(50):
        temp_feat_arg = grad_feat_node_sort[ii]  #
        temp_feat_idx_np = np.where(grad_sum.flatten() == temp_feat_arg)[0]  # 找到当前最大值的temp_feat
        for jj in range(temp_feat_idx_np.shape[0]):
            temp_feat_node_idx = int(int(temp_feat_idx_np[jj]) / int(feat_fake.shape[1]))
            temp_feat_idx = int(int(temp_feat_idx_np[jj]) % int(feat_fake.shape[1]))

            if feat_fake[temp_feat_node_idx + feat_fake.shape[0] - p_fake_node_idx_np.shape[0], temp_feat_idx] == 1:
                continue
            else:
                feat_fake[temp_feat_node_idx + feat_fake.shape[0] - p_fake_node_idx_np.shape[0], temp_feat_idx] = 1
                flag = 1
                break
        if flag == 1:
            print("此时的梯度大小为： {}".format(grad_feat_node_sort[ii]))
            break
    print("完成")
    return feat_fake


def Per_add_fake_feat_based_on_grad_for_Class_Node_Unchanged_Label(p_grad_np: np.ndarray, output: np.ndarray,
                                                                   p_feat_fake: np.ndarray,
                                                                   p_fake_node_idx_np: np.ndarray, backdoor_idx,
                                                                   Origin_Info: F_Info.C_per_info):
    """
    根据目标输出的梯度信息，先滤去已经改变标签的节点梯度， 然后对梯度进行求和， 计算最小值。
    对于已经置1的特征，额外循环进行置1操作
    :param Origin_Info: C_per_info 的类实现 用于保存原有图的信息
    :param backdoor_idx: 后门节点的idx 用于查看该节点的标签
    :param p_grad_np:   输出的梯度 [10,5,1433] [target_node_num, fake_node_num, feat_num]
    :param output:      输出的标签预测值 [10, 7] [target_node_num, label_num]
    :param p_feat_fake: 当前的特征 [2713, 1433]  [total_node_num, feat_num]
    :param p_fake_node_idx_np:  假节点的index [5] [fake_node_num]
    :return:    fake_feat [2713, 1433] [total_node_num, feat_num]
    """

    # 初始化一些变量
    fake_node_num = p_fake_node_idx_np.shape[0]
    target_node_num = p_grad_np.shape[0]
    feat_num = p_grad_np.shape[2]
    label_num = output.shape[1]
    total_node_num = p_feat_fake.shape[0]
    backdoor_label = np.where(Origin_Info.label_np[backdoor_idx])[0][0]
    feat_fake = np.copy(p_feat_fake)

    """1、 除去已经变为后门标签的target_label结果"""
    changed_label_num = 0
    changed_label_index = np.array([])
    for i in range(target_node_num):
        temp_label = np.argmax(output[i])
        if temp_label == backdoor_label:
            changed_label_index = np.append(changed_label_index, i)
            changed_label_num = changed_label_num + 1
        else:
            continue
    grad_exclude_np = np.delete(p_grad_np, changed_label_index, 0)  # 删除第changed_label_index行的结果
    target_excluded_num = grad_exclude_np.shape[0]

    """2、对排除后的梯度进行求和， 得到梯度的加权求和"""
    grad_exclude_sum_np = np.zeros([fake_node_num, feat_num])
    for ii in range(target_excluded_num):
        grad_exclude_sum_np = grad_exclude_sum_np + grad_exclude_np[ii]

    """3、在加权求和的结果中找到最小值，并对其进行排序"""
    grad_exclude_sum_sort = np.unique(np.sort(grad_exclude_sum_np.flatten()))

    # 记录当前是否特征改变成功
    flag = 0

    for ii in range(50):
        temp_feat_arg = grad_exclude_sum_sort[ii]
        temp_feat_idx_np = np.where(grad_exclude_sum_np.flatten() == temp_feat_arg)[0]  # 找到最大值的数组

        for jj in range(temp_feat_idx_np.shape[0]):
            temp_feat_node_idx = int(temp_feat_idx_np[jj] / feat_num) + total_node_num - fake_node_num
            temp_feat_idx = int(temp_feat_idx_np[jj] % feat_num)

            if feat_fake[temp_feat_node_idx, temp_feat_idx] == 1:
                continue
            else:
                feat_fake[temp_feat_node_idx, temp_feat_idx] = 1
                flag = 1
                break

        if flag == 1:
            # print("此时的梯度大小为：{}".format(grad_exclude_sum_sort[ii]))
            # print("添加特征为：第{}个节点的,第{}位\n".format(temp_feat_node_idx, temp_feat_idx))
            break

    return feat_fake


def Per_add_fake_feat_based_on_grad_multi_anchor_nodes(p_grad_np: np.ndarray, p_feat_fake: np.ndarray):
    """
    根据目标输出的梯度信息，先滤去已经改变标签的节点梯度， 然后对梯度进行求和， 计算最小值。
    对于已经置1的特征，额外循环进行置1操作
    """

    # 初始化一些变量
    fake_node_num = p_grad_np.shape[0]
    feat_num = p_grad_np.shape[1]
    total_node_num = p_feat_fake.shape[0]
    feat_fake = np.copy(p_feat_fake)
    """3、在加权求和的结果中找到最小值，并对其进行排序"""
    grad_exclude_sum_sort = np.unique(np.sort(p_grad_np.flatten()))

    # 记录当前是否特征改变成功
    flag = 0

    for ii in range(50):
        temp_feat_arg = grad_exclude_sum_sort[ii]
        temp_feat_idx_np = np.where(p_grad_np.flatten() == temp_feat_arg)[0]  # 找到最大值的数组

        for jj in range(temp_feat_idx_np.shape[0]):
            temp_feat_node_idx = int(temp_feat_idx_np[jj] / feat_num) + total_node_num - fake_node_num
            temp_feat_idx = int(temp_feat_idx_np[jj] % feat_num)

            if feat_fake[temp_feat_node_idx, temp_feat_idx] == 1:
                continue
            else:
                feat_fake[temp_feat_node_idx, temp_feat_idx] = 1
                flag = 1
                break

        if flag == 1:
            print("此时的梯度大小为：{}".format(grad_exclude_sum_sort[ii]))
            print("添加特征为：第{}个节点的,第{}位\n".format(temp_feat_node_idx, temp_feat_idx))
            break

    return feat_fake


def Per_add_fake_feat_based_on_grad_vec(p_grad_np_vec:np.ndarray, p_output_minus:np.ndarray ,p_feat_fake: np.ndarray):
    """
    根据输出的梯度信息，改变节点特征，引入vector的概念
    0.9、 选取anci_node中非0的部分，生成一个ancillary_node列表
    1、计算 G = (▽ft- ▽fc) / |▽ft-▽fc|2    -> 单位向量
    2、除去 非同向的向量即 G1*G2 < 0
    3、计算 G * △f
    4、加和后 取最大值，即为所求向量
    :param p_grad_np_vec:   [anci_num , fake_node_num , feat_num]
    :param p_feat_fake:     [node_num + fake_node_num , feat_num]
    :param p_output_minus:  [anci_num]
    :return:
    """

    # ancillary node列表
    anci_node_num = p_grad_np_vec.shape[0]
    anci_node_id = []
    for i in range(anci_node_num):
        if np.all(p_grad_np_vec[i].flatten() == 0):
            continue
        anci_node_id.append(i)
    anci_node_id_useful = np.array(anci_node_id)
    if anci_node_id_useful.size == 0:
        return p_feat_fake



    fake_node_num = p_grad_np_vec.shape[1]
    original_node_num = p_feat_fake.shape[0] - fake_node_num
    total_node_num = p_feat_fake.shape[0]
    feat_num = p_feat_fake.shape[1]
    fake_node_id_np = np.arange(original_node_num-1, total_node_num-1)

    """step1. Calculate G"""
    GradFlatVec = np.zeros([anci_node_num, fake_node_num*feat_num])
    GradFlatVecHat = np.zeros([anci_node_num, fake_node_num * feat_num])
    for i in anci_node_id_useful:
        GradFlatVec[i] = p_grad_np_vec[i].flatten()     # transfer to Vector
        GradFlatVecHat[i] = GradFlatVec[i] / np.linalg.norm(GradFlatVec[i])     # calculate Unit Vector

    """step2. Randomly select a vector as the base"""
    base_id = np.random.choice(anci_node_id_useful)
    for i in anci_node_id_useful:
        if i == base_id:
            continue
        temp_cosine = np.sum(GradFlatVecHat[i]*GradFlatVecHat[base_id],axis=0)
        if temp_cosine < 0:
            # 说明角度过大，需要排除
            GradFlatVecHat[i].fill(0)

    """step3. 计算 G * △f"""
    for i in range(anci_node_num):
        GradFlatVecHat[i] = GradFlatVecHat[i] * abs(p_output_minus[i])

    """step4. 计算 Σ GHat"""
    GradFlatVecHatSum = np.zeros(fake_node_num * feat_num)
    for i in anci_node_id_useful:
        GradFlatVecHatSum = GradFlatVecHatSum + GradFlatVecHat[i]

    """step5. choose the max_feat"""
    grad_exclude_sum_sort = np.unique(np.sort(GradFlatVecHatSum))

    for lp_i in range(50):
        temp_feat_arg = grad_exclude_sum_sort[lp_i]
        temp_feat_idx_np = np.where(GradFlatVecHatSum == temp_feat_arg)[0]

        for jj in range(temp_feat_idx_np.shape[0]):
            temp_feat_node_idx = int(temp_feat_idx_np[jj] / feat_num) + total_node_num - fake_node_num
            temp_feat_idx = int(temp_feat_idx_np[jj] % feat_num)

            if p_feat_fake[temp_feat_node_idx,temp_feat_idx]==1:
                continue
            else:
                # print("此时的梯度大小为：{}".format(grad_exclude_sum_sort[lp_i]))
                # print("添加特征为：第{}个节点的,第{}位\n".format(temp_feat_node_idx, temp_feat_idx))
                p_feat_fake[temp_feat_node_idx,temp_feat_idx] =1
                return p_feat_fake

    return p_feat_fake











