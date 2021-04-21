import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from config import opts
import random
import os
import sys
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from fold_model import GCN
from fold_util import F_Normalize as F_Nor
# from fold_model import GCN_PyG

def train(p_adj_np: np.ndarray, p_feat_np: np.ndarray, p_labels_np: np.ndarray, p_idx_train_np, p_idx_val_np, is_normalize_feat=False):
    use_gpu = t.cuda.is_available()
    random.seed(40)
    np.random.seed(40)
    t.manual_seed(40)
    best_acc = 0
    opt = opts()
    label_tensor = t.LongTensor(np.where(p_labels_np)[1])

    # 邻接矩阵对角线元素加1
    p_adj_np = (sp.csr_matrix(p_adj_np) + sp.eye(p_adj_np.shape[1])).A
    degree_np = F_Nor.normalize_adj_degree(p_adj_np)  # D^0.5

    '''特征标准化？'''
    if is_normalize_feat:
        feat_nor_np = F_Nor.normalize_feat(p_feat_np)  # 按行标准化特征p
    else:
        feat_nor_np = p_feat_np
    """模型预定义"""
    model = GCN(
        nfeat=feat_nor_np.shape[1],
        nhid=opt.num_hiden_layer,
        nclass=label_tensor.max().item() + 1,
        dropout=opt.drop_out,
        init=opt.init_type
    )

    """优化器定义"""
    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError

    """Numpy to Tensor"""
    adj_tensor = t.from_numpy(p_adj_np).float()
    feat_nor_tensor = t.from_numpy(feat_nor_np).float()

    idx_train_tensor = t.from_numpy(p_idx_train_np).long()
    idx_val_tensor = t.from_numpy(p_idx_val_np).long()
    degree_tensor = t.from_numpy(degree_np).float()

    """Tensor CPU -> GPU"""
    if use_gpu:
        model.cuda()
        adj_tensor, feat_nor_tensor, label_tensor, idx_train_tensor, idx_val_tensor, degree_tensor = \
            list(map(lambda x: x.cuda(),
                     [adj_tensor, feat_nor_tensor, label_tensor, idx_train_tensor, idx_val_tensor, degree_tensor]))

    adj_tensor, feat_nor_tensor, label_tensor, degree_tensor = list(
        map(lambda x: Variable(x), [adj_tensor, feat_nor_tensor, label_tensor, degree_tensor]))

    feat_nor_tensor.requires_grad = True

    # 对称拉普拉斯 D^-0.5 * A * D^-0.5
    D_Adj_tensor = t.mm(degree_tensor, adj_tensor).cuda()  # D^-0.5 * A
    adj_nor_tensor = t.mm(D_Adj_tensor, degree_tensor).cuda()  # D^-0.5 * A * D^-0.5

    """保存模型"""
    save_point = os.path.join('./checkpoint', opt.dataset)
    if not os.path.isdir(save_point):
        os.mkdir(save_point)

    for epoch in np.arange(1, opt.epoch + 1):
        model.train()  # 训练模式

        optimizer.lr = F_lr_scheduler(epoch, opt)  # 学习率增减
        optimizer.zero_grad()  # 重置梯度

        output = model(feat_nor_tensor, adj_nor_tensor)  # 输出模型结果 [cora : 7 class]
        loss_train = F.nll_loss(output[idx_train_tensor], label_tensor[idx_train_tensor])
        acc_train = F_accuracy(output[idx_train_tensor], label_tensor[idx_train_tensor])

        loss_train.backward()  # 反向传播
        optimizer.step()  # 优化参数

        # Validation
        model.eval()
        output = model(feat_nor_tensor, adj_nor_tensor)
        acc_val = F_accuracy(output[idx_val_tensor], label_tensor[idx_val_tensor])

        if acc_val > best_acc:
            best_acc = acc_val
            state = {
                'model': model,
                'acc': best_acc,
                'epoch': epoch,
            }
            t.save(state, os.path.join(save_point, '%s.t7' % opt.model))  # 保存成以模型为名称的.t7文件 eg. GCN.t7
        if epoch % 10 == 0:
            sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.write(" => Training Epoch #{}".format(epoch))
            sys.stdout.write(" | Training acc : {:6.2f}%".format(acc_train.data.cpu().numpy() * 100))
            sys.stdout.write(" | Learning Rate: {:6.4f}".format(optimizer.lr))
            sys.stdout.write(" | Best acc : {:.2f}".format(best_acc.data.cpu().numpy() * 100))


def F_lr_scheduler(epoch, opt):
    return opt.lr * (0.5 ** (epoch / opt.lr_decay_epoch))


# 定义准确率
def F_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def Train_PyG(p_adj_np: np.ndarray, p_feat_np: np.ndarray, p_label_np: np.ndarray, p_idx_train: np.ndarray,
              p_idx_val: np.ndarray):
    use_gpu = t.cuda.is_available()
    random.seed(40)
    np.random.seed(40)
    t.manual_seed(40)
    best_acc = 0
    opt = opts()
    p_adj_np = p_adj_np + np.eye(p_adj_np.shape[0])
    label_tensor = t.LongTensor(np.where(p_label_np)[1])

    # 由邻接矩阵获取Edge List
    edge_list_np = F_Nor.get_edge_list(p_adj_np)

    """特征标准化"""
    feat_nor_np = p_feat_np
    # feat_nor_np = F_Nor.normalize_feat(p_feat_np)

    """模型预定义"""
    model = GCN_PyG(
        nfeat=feat_nor_np.shape[1],
        nhid=opt.num_hiden_layer,
        nclass=label_tensor.max().item() + 1
    )

    """优化器定义"""
    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError

    """Numpy to Tensor"""
    feat_nor_tensor = t.from_numpy(feat_nor_np).float().cuda()
    edge_list_tensor = t.from_numpy(edge_list_np).long().cuda()
    label_tensor = t.from_numpy(np.where(p_label_np)[1]).long().cuda()
    idx_train_tensor = t.from_numpy(p_idx_train).long().cuda()
    idx_val_tensor = t.from_numpy(p_idx_val).long().cuda()
    model.cuda()

    edge_list_tensor, feat_nor_tensor, label_tensor = list(
        map(lambda x: Variable(x), [edge_list_tensor, feat_nor_tensor, label_tensor]))

    feat_nor_tensor.requires_grad = True

    """保存模型"""
    save_point = os.path.join('./checkpoint', "GCN_PyG")
    if not os.path.isdir(save_point):
        os.mkdir(save_point)

    for epoch in np.arange(1, opt.epoch + 1):
        model.train()  # 训练阶段

        optimizer.lr = F_lr_scheduler(epoch, opt)  # 学习率衰减
        optimizer.zero_grad()

        output = model(feat_nor_tensor, edge_list_tensor)
        loss_train = F.nll_loss(output[idx_train_tensor], label_tensor[idx_train_tensor])
        acc_train = F_accuracy(output[idx_train_tensor], label_tensor[idx_train_tensor])

        loss_train.backward()
        optimizer.step()

    # validation
    model.eval()
    # output = model(feat_nor_tensor, edge_list_tensor)
    # acc_val = F_accuracy(output[idx_val_tensor], label_tensor[idx_val_tensor])
    _,pred = model(feat_nor_tensor,edge_list_tensor).max(dim=1)
    correct = float(pred[idx_val_tensor].eq(label_tensor[idx_val_tensor]).sum().item())
    acc = correct / idx_val_tensor.sum().item()
    print("acc : {:.4f}".format(acc))

    #
    # model.eval()
    #
    # state = {
    #     'model': model,
    #     'acc': best_acc,
    #     'epoch': epoch,
    # }
    # t.save(state, os.path.join(save_point, 'GCN_PyG.t7'))

