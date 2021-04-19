import numpy as np
import torch as t

from config import opts


class C_per_info():
    def __init__(self, p_adj: np.ndarray, p_feat: np.ndarray, p_label: np.ndarray, p_idx_train: np.ndarray,
                 p_idx_val: np.ndarray, p_idx_test: np.ndarray, p_opts: opts = opts):
        self.adj_np = p_adj
        self.feat_np = p_feat
        self.label_np = p_label
        self.idx_train = p_idx_train
        self.idx_val = p_idx_val
        self.idx_test = p_idx_test
        self.opt = p_opts
        self.random_seed = self.opt.np_random_seed

    def F_get_top_n_feat_each_class(self, p_top=5):
        """返回每类中数量最多的特征"""
        label_enum = np.arange(self.label_np.shape[1])  # 标签数

        label_temp = np.where(self.label_np)[1]  # one-hot转正确数字

        # 返回矩阵，由p_top和Label_Num决定
        feat_top_np = np.zeros((self.label_np.shape[1], p_top))

        for ii in label_enum:
            idx_in_label = np.where(label_temp == ii)
            element_in_label = np.array(idx_in_label).squeeze()
            feat_temp_ii = self.feat_np[element_in_label].sum(axis=1)
            feat_top_np[ii] = (-feat_temp_ii).argsort()
        return feat_top_np[:, :p_top]

    def F_get_random_n_nodes_from_each_class(self, node_num):
        label = F_one_hot_to_label(self.label_np)
        class_num = label.max() + 1

        # 初始化保存node_index的矩阵
        node_index = np.zeros([class_num, node_num], np.int16)
        for ii in range(class_num):
            temp_index_np = np.where(label == ii)[0]
            np.random.seed(self.random_seed)
            node_index[ii] = np.random.choice(temp_index_np, node_num)
        return node_index

    def F_idx_to_class(self, idx):
        """返回index为idx的标签值 （非 onehot）"""
        return np.where(self.label_np)[1][idx]

    def F_get_top_n_feat_two_class(self, target_class, second_class, p_top=5):
        """统计两类特征，返回相减后的top_n个特征"""
        label_temp = np.where(self.label_np)[1]  # one-hot转数字

        # 寻找目标类和第二类的idx
        idx_target_class = np.array(np.where(label_temp == target_class)).squeeze()
        idx_second_class = np.array(np.where(label_temp == second_class)).squeeze()

        # 对特征求和
        feat_target_class_sum = self.feat_np[idx_target_class].sum(axis=0)
        feat_second_class_sum = self.feat_np[idx_second_class].sum(axis=0)

        # 特征相减 求出target类中最多的特征
        feat_top = (-(feat_target_class_sum - feat_second_class_sum)).argsort()

        return feat_top[:p_top]

    def F_get_top_n_feat_two_idx(self, target_idx, second_idx, p_top=5):
        """统计两个节点的标签所在特征 返回相减后的top_n个特征"""
        target_class = self.F_idx_to_class(target_idx)
        second_class = self.F_idx_to_class(second_idx)

        return self.F_get_top_n_feat_two_class(target_class, second_class, p_top)

    def F_get_K_random_idx_of_single_class(self, target_class: int, node_num=10) -> np.ndarray:
        """找到在单类中的K个节点标签，作为攻击目标集合"""
        # total_label_idx_np = np.arange(self.label_np.shape[0])
        # idx_choose_not_train = np.delete(total_label_idx_np, self.idx_train)    # 除去训练用节点
        # idx_choose = np.delete(idx_choose_not_train, self.idx_val)     # 除去测试节点
        # label_one_hot = self.label_np[idx_choose]   # 在除去训练和测试节点的部分中选择

        label_not_one_hot = np.where(self.label_np)[1]  # onehot转标签号
        idx_target_class = np.where(label_not_one_hot == target_class)[0]  # 找到与目标类相同的idx
        idx_target_class_not_train = np.setdiff1d(idx_target_class, self.idx_train)  # 删除训练节点
        idx_target_class_not_train_val = np.setdiff1d(idx_target_class_not_train, self.idx_val)  # 删除测试节点

        np.random.seed(self.random_seed)  # 设定随机因子
        idx_sample_np = np.random.choice(idx_target_class_not_train_val, node_num)

        return idx_sample_np

    def F_get_K_random_idx_except_one_class(self, except_class: int, node_num=10) -> np.ndarray:
        label_not_one_hot = np.where(self.label_np)[1]
        idx_target_class = np.where(label_not_one_hot != except_class)[0]

        np.random.seed(self.random_seed)  # 设定随机因子
        idx_sample_np = np.random.choice(idx_target_class, node_num)

        return idx_sample_np

    def F_get_K_random_idx_of_Group_Nodes(self, backdoor_index: int, node_num=10, opt_random_seed=40) -> np.ndarray:
        """找到除去验证节点和训练节点外的K个节点"""
        label_not_one_hot = np.where(self.label_np)[1]  # OneHot转标签号
        backdoor_class = self.F_idx_to_class(backdoor_index)

        idx_not_backdoor_class = np.where(label_not_one_hot != backdoor_class)[0]
        idx_not_backdoor_class_not_train = np.setdiff1d(idx_not_backdoor_class, self.idx_train)  # 除去训练的idx
        idx_not_backdoor_class_not_train_val = np.setdiff1d(idx_not_backdoor_class_not_train, self.idx_val)  # 除去验证的idx

        np.random.seed(opt_random_seed)
        idx_sample_np = np.random.choice(idx_not_backdoor_class_not_train_val, node_num)

        return idx_sample_np

    def F_from_label_to_idx(self, label_id: int) -> np.ndarray:
        """通过label 找到对应的index"""
        label_not_one_hot = F_one_hot_to_label(self.label_np)

        idx_label_np = np.where(label_not_one_hot == label_id)[0]

        return idx_label_np

    def F_get_vote_top_feat_for_Single_Node(self, backdoor_index: int) -> int:
        """用投票的方式找到除去目标类外的最大特征idx"""
        backdoor_label = self.F_idx_to_class(backdoor_index)  # 后门节点的label
        label_num = self.label_np.shape[1]  # 标签数量

        record_feat = np.zeros(label_num - 1)  # 一共标签数量-1个投票

        """统计后门类中的特征求和"""
        backdoor_feat_add_up = np.zeros(self.feat_np.shape[1])
        backdoor_label_idx = self.F_from_label_to_idx(backdoor_label)
        for ii in range(backdoor_label_idx.shape[0]):
            temp_index = backdoor_label_idx[ii]
            backdoor_feat_add_up = backdoor_feat_add_up + self.feat_np[temp_index]

        temp_loop = 0
        for ii in range(label_num):
            if ii == backdoor_label:
                continue
            temp_feat_add_up = np.zeros(self.feat_np.shape[1])
            temp_label = ii
            temp_index_np = self.F_from_label_to_idx(temp_label)
            """统计特征总和"""
            for jj in range(temp_index_np.shape[0]):
                temp_index = temp_index_np[jj]
                temp_feat_add_up = temp_feat_add_up + self.feat_np[temp_index]
            temp_feat_subtract = backdoor_feat_add_up - temp_feat_add_up
            record_feat[temp_loop] = int(np.argmax(temp_feat_subtract))  # 记录最大的feat的index
            temp_loop = temp_loop + 1
        top_index = np.argmax(np.bincount(record_feat.astype(int)))  # 找到数组中出现次数最多的元素

        return top_index


def F_one_hot_to_label(label: np.ndarray):
    """返回非独热编码的label"""
    return np.where(label)[1]
