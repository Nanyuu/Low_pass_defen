import numpy as np
import torch as t
import torch.nn.functional as F

from fold_util import F_Normalize as F_Nor
from config import opts


def Test_for_single_node_attack(adj, feat, label, backdoor_idx, target_idx, opt: opts, is_nor_feat=False, is_test = False):
    """
    test函数需要 邻接矩阵adj ， 特征矩阵feat， 后门节点， 目标节点，初始化信息
    根据输入信息，计算交叉熵loss，返回梯度信息
    :param is_nor_feat:
    :param adj:
    :param feat:
    :param label:
    :param backdoor_idx:
    :param target_idx:
    :param opt:
    :return: 特征x的梯度
    """
    # 初始化必要信息
    use_gpu = t.cuda.is_available()
    label_backdoor_np = np.where(label[backdoor_idx])[0]
    label_target_np = np.where(label[target_idx])[0]

    # A -> D^-0.5 * A * D^-0.5
    if is_test:
        adj_nor = F_Nor.normalize_adj_sym(adj)
    else:
        if opt.flag_att_single_node == 1:
            # 如果是对单节点进行攻击
            if np.all(opt.normed_adj_single_node == 0):
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_single_node = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_single_node)
        elif opt.flag_att_single_node_group_test == 1:
            # 如果是对单节点多测试进行攻击
            if np.all(opt.normed_adj_single_node_group_np[opt.normed_adj_idx] == 0):
                # 如果是全0数组 则说明还未计算normed_adj
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_single_node_group_np[opt.normed_adj_idx] = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_single_node_group_np[opt.normed_adj_idx])
        elif opt.flag_att_class_nodes == 1:
            # 如果是对一类的节点进行攻击
            if np.all(opt.normed_adj_class_np[opt.normed_adj_idx] == 0):
                # 如果是全0数组 则还未计算normed_adj
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_class_np[opt.normed_adj_idx] = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_class_np[opt.normed_adj_idx])
        elif opt.flag_att_group_nodes == 1:
            # 如果是对全图节点进行攻击
            if np.all(opt.normed_adj_group_np[opt.normed_adj_idx] == 0):
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_group_np[opt.normed_adj_idx] = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_group_np[opt.normed_adj_idx])
        else:
            raise NotImplementedError

    # 是否对特征进行标准化
    if is_nor_feat:
        feat_nor_np = F_Nor.normalize_feat(feat)
    else:
        feat_nor_np = feat

    # numpy -> tensor
    adj_T = t.from_numpy(adj_nor).float()
    feat_T = t.from_numpy(feat_nor_np).float()
    label_T = t.from_numpy(label).long()
    label_target_T = t.from_numpy(label_target_np).long()
    label_backdoor_T = t.from_numpy(label_backdoor_np).long()

    # 读取模型
    load_model = t.load("{}/{}/{}.t7".format(opt.model_path, opt.dataset, opt.model))
    model = load_model['model'].cpu()

    # cpu -> gpu
    if use_gpu:
        model.cuda()
        adj_T, feat_T, label_T, label_target_T, label_backdoor_T = list(
            map(lambda x: x.cuda(), [adj_T, feat_T, label_T, label_target_T, label_backdoor_T]))

    # 特征矩阵需要梯度
    feat_T.requires_grad = True

    # 测试模式
    model.eval()
    output = model(feat_T, adj_T)

    # 计算交叉熵
    loss = F.nll_loss(output[[target_idx]], label_backdoor_T)

    # 反向传播
    loss.backward()

    return feat_T.grad.cpu(), output[target_idx].cpu()


def Test_for_single_node_group_attack(adj, feat, label, backdoor_idx, target_idx, opt: opts, is_nor_feat=False):
    """
    test函数需要 邻接矩阵adj ， 特征矩阵feat， 后门节点， 目标节点，初始化信息
    根据输入信息，计算交叉熵loss，返回梯度信息
    :param is_nor_feat:
    :param adj:
    :param feat:
    :param label:
    :param backdoor_idx:
    :param target_idx:
    :param opt:
    :return: 特征x的梯度
    """
    # 初始化必要信息
    use_gpu = t.cuda.is_available()
    label_backdoor_np = np.where(label[backdoor_idx])[0]
    label_target_np = np.where(label[target_idx])[0]

    # A -> D^-0.5 * A * D^-0.5
    adj_nor = F_Nor.normalize_adj_sym(adj)

    # 是否对特征进行标准化
    if is_nor_feat:
        feat_nor_np = F_Nor.normalize_feat(feat)
    else:
        feat_nor_np = feat

    # numpy -> tensor
    adj_T = t.from_numpy(adj_nor).float()
    feat_T = t.from_numpy(feat_nor_np).float()
    label_T = t.from_numpy(label).long()
    label_target_T = t.from_numpy(label_target_np).long()
    label_backdoor_T = t.from_numpy(label_backdoor_np).long()

    # 读取模型
    load_model = t.load("{}/{}/{}.t7".format(opt.model_path, opt.dataset, opt.model))
    model = load_model['model'].cpu()

    # cpu -> gpu
    if use_gpu:
        model.cuda()
        adj_T, feat_T, label_T, label_target_T, label_backdoor_T = list(
            map(lambda x: x.cuda(), [adj_T, feat_T, label_T, label_target_T, label_backdoor_T]))

    # 特征矩阵需要梯度
    feat_T.requires_grad = True

    # 测试模式
    model.eval()
    output = model(feat_T, adj_T)

    # 计算交叉熵
    loss = F.nll_loss(output[[target_idx]], label_backdoor_T)

    # 反向传播
    loss.backward()

    return feat_T.grad.cpu(), output[target_idx].cpu()


def Test_for_minus_node_attack(adj, feat, label, backdoor_idx, target_idx, opt: opts, is_nor_feat=False, is_test=False):
    """
    Test函数，按照最大label和目标label的差值来计算梯度信息，求取最终结果
    :param adj:
    :param feat:
    :param label:
    :param backdoor_idx:
    :param target_idx:
    :param opt:
    :param is_nor_feat:
    :return:
    """

    # 初始化信息
    use_gpu = t.cuda.is_available()
    label_backdoor_np = np.where(label[backdoor_idx])[0]
    label_target_np = np.where(label[target_idx])[0]
    label_not_one_hot = np.where(label)[1]

    """A->D^-0.5 * A * D^-0.5"""
    # 对已正则化后的邻接矩阵进行存储
    if is_test:
        adj_nor = F_Nor.normalize_adj_sym(adj)
    else:
        if opt.flag_att_single_node == 1:
            # 如果是对单节点进行攻击
            if np.all(opt.normed_adj_single_node == 0):
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_single_node = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_single_node)
        elif opt.flag_att_single_node_group_test == 1:
            # 如果是对单节点多测试进行攻击
            if np.all(opt.normed_adj_single_node_group_np[opt.normed_adj_idx] == 0):
                # 如果是全0数组 则说明还未计算normed_adj
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_single_node_group_np[opt.normed_adj_idx] = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_single_node_group_np[opt.normed_adj_idx])
        elif opt.flag_att_class_nodes == 1:
            # 如果是对一类的节点进行攻击
            if np.all(opt.normed_adj_class_np[opt.normed_adj_idx] == 0):
                # 如果是全0数组 则还未计算normed_adj
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_class_np[opt.normed_adj_idx] = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_class_np[opt.normed_adj_idx])
        elif opt.flag_att_group_nodes == 1:
            # 如果是对全图节点进行攻击
            if np.all(opt.normed_adj_group_np[opt.normed_adj_idx] == 0):
                adj_nor = F_Nor.normalize_adj_sym(adj)
                opt.normed_adj_group_np[opt.normed_adj_idx] = np.copy(adj_nor)
            else:
                adj_nor = np.copy(opt.normed_adj_group_np[opt.normed_adj_idx])
        else:
            raise NotImplementedError

    # 是否对特征进行标准化
    if is_nor_feat:
        feat_nor_np = F_Nor.normalize_feat(feat)
    else:
        feat_nor_np = np.copy(feat)

    # numpy -> tensor
    adj_T = t.from_numpy(adj_nor).float()
    feat_T = t.from_numpy(feat_nor_np).float()
    label_T = t.from_numpy(label).long()
    label_target_T = t.from_numpy(label_target_np).long()
    label_backdoor_T = t.from_numpy(label_backdoor_np).long()

    # 读取模型
    load_model = t.load("{}/{}/{}.t7".format(opt.model_path, opt.dataset, opt.model))
    model = load_model['model'].cpu()

    # cpu -> gpu
    if use_gpu:
        model.cuda()
        adj_T, feat_T, label_T, label_target_T, label_backdoor_T = list(
            map(lambda x: x.cuda(), [adj_T, feat_T, label_T, label_target_T, label_backdoor_T]))

    # 特征矩阵需要梯度
    feat_T.requires_grad = True

    # 测试模式
    model.eval()
    output = model(feat_T, adj_T)

    label_target_T = output[[target_idx]].squeeze().argmax().unsqueeze(dim=0)

    # 计算交叉熵
    loss_target = F.nll_loss(output[[target_idx]], label_target_T)
    loss_backdoor = F.nll_loss(output[[target_idx]], label_backdoor_T)

    # 相减后反向传播
    loss_minus = loss_backdoor - loss_target

    # 反向传播
    loss_minus.backward()

    return feat_T.grad.cpu(), output[target_idx].cpu()


def Test_is_equal_label(label_OneHot, output: t.Tensor, backdoor_idx):
    """根据output 检查label是否已经发生了改变"""
    label_backdoor_OneHot = np.where(label_OneHot[backdoor_idx])[0][0]
    target_label = output.argmax().item()
    # print("后门label为{},目标label为{}".format(label_backdoor_OneHot, target_label))
    if label_backdoor_OneHot == target_label:
        return True
    else:
        return False


def Test_attack_success_rate_for_Class_Node(label_OneHot: np.ndarray, output: np.ndarray, backdoor_idx: int):
    """
    测试一组节点的输出分类攻击成功率
    :param label_OneHot: 独热编码的 label        [2708,7]
    :param output: K个目标节点的输出 output         [K,7]
    :param backdoor_idx: 后门节点的Index         Int
    :return:    改变率 ， 即攻击成功率
    """
    total_target_num = output.shape[0]  # 总共目标节点的数量
    changed_num = 0  # 保存改变了的节点数量
    backdoor_label = np.where(label_OneHot[backdoor_idx])[0][0]

    for ii in range(total_target_num):
        target_label = output[ii].argmax().item()
        if target_label == backdoor_label:
            changed_num = changed_num + 1
        else:
            continue

    attack_success_rate = float(changed_num) / float(total_target_num)

    return attack_success_rate


def Test_for_minus_node_attack_multi_anchor_node(adj_nor, feat, label, anchor_idx, ancillary_idx, opt: opts, is_nor_feat=False, is_test=False):
    """
    Test函数，按照最大label和目标label的差值来计算梯度信息，求取最终结果
    :param adj:
    :param feat:
    :param label:
    :param anchor_idx:
    :param ancillary_idx:
    :param opt:
    :param is_nor_feat:
    :return: feat_grad_cpu, output[anci_idx]
    """

    # 初始化信息
    use_gpu = t.cuda.is_available()
    label_anchor_np = np.where(label[anchor_idx])[0]
    label_anci_np = np.where(label[ancillary_idx])[0]
    label_not_one_hot = np.where(label)[1]

    """A->D^-0.5 * A * D^-0.5"""
    feat_nor_np = F_Nor.normalize_feat(feat)

    # numpy -> tensor
    adj_T = t.from_numpy(adj_nor).float()
    feat_T = t.from_numpy(feat_nor_np).float()
    label_T = t.from_numpy(label).long()
    label_anci_T = t.from_numpy(label_anci_np).long()
    label_backdoor_T = t.from_numpy(label_anchor_np).long()

    # 读取模型
    load_model = t.load("{}/{}/{}.t7".format(opt.model_path, opt.dataset, opt.model))
    model = load_model['model'].cpu()

    # cpu -> gpu
    if use_gpu:
        model.cuda()
        adj_T, feat_T, label_T, label_anci_T, label_backdoor_T = list(
            map(lambda x: x.cuda(), [adj_T, feat_T, label_T, label_anci_T, label_backdoor_T]))

    # 特征矩阵需要梯度
    feat_T.requires_grad = True

    # 测试模式
    model.eval()
    output = model(feat_T, adj_T)

    label_anci_T = output[[ancillary_idx]].squeeze().argmax().unsqueeze(dim=0)

    # 计算交叉熵
    loss_target = F.nll_loss(output[[ancillary_idx]], label_anci_T)
    loss_backdoor = F.nll_loss(output[[ancillary_idx]], label_backdoor_T)

    # 相减后反向传播
    loss_minus = loss_backdoor - loss_target

    # 反向传播
    loss_minus.backward()

    return feat_T.grad.cpu(), output[ancillary_idx].cpu()
