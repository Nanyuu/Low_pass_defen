import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch as t


class c_dataset_loader():
    def __init__(self, dataset_name, data_path):
        self.path = data_path
        self.dataset = dataset_name

    def get_train(self):
        '''
        读取训练集
        :return: ndarry - x，ndarry - y
        '''
        index_train = ['x', 'y']  # 训练集
        objects = []
        for i in range(len(index_train)):
            with open("{}/ind.{}.{}".format(self.path, self.dataset, index_train[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))
        x, y = tuple(objects)  # x- csr_matrix, y - ndarray
        x = x.A  # csr_matrix -> ndarray
        return x, y

    def get_test(self):
        """
        读取验证集
        :return: ndarray -tx , ndarray - ty
        """
        index_val = ['tx', 'ty']
        objects = []
        for i in range(len(index_val)):
            with open("{}/ind.{}.{}".format(self.path, self.dataset, index_val[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))
        tx, ty = tuple(objects)  # tx- csr_matrix, y - ndarray
        tx = tx.A  # csr_matrix -> ndarray
        return tx, ty

    def get_all(self):
        """
        读取测试集，并对测试部分你重新排序
        :return: ndarray - allx, ndarray - ally
        """
        index_test = ['allx', 'ally']
        objects = []
        for i in range(len(index_test)):
            with open("{}/ind.{}.{}".format(self.path, self.dataset, index_test[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))
        allx, ally = tuple(objects)
        allx = allx.A
        return allx, ally

    def parse_index_file(self, filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def get_adj(self):
        """
        获取图结构信息（邻接矩阵等）
        :return: graph - NetworkX
        """
        with open("{}/ind.{}.{}".format(self.path, self.dataset, "graph"), 'rb') as f:
            graph = pkl.load(f, encoding='latin1')
        # adj - csr_matrix
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        # 对角线上加上1
        # adj = adj + sp.eye(adj.shape[0])
        print("| # of nodes:{}".format(adj.shape[0]))
        print("| # of edges:{}".format(adj.sum().sum() / 2))
        return adj.A  # csr_matrix -> NdArray

    def process_data(self):
        """
        获取 adj, feature, label, index_train, index_val , index_test
        :return: adj, feature, label, index_train, index_val , index_test
        """
        # 获取数据 NdArray
        x, y = self.get_train()  # x - NdArray ~ [140,1433] ; y - NdArray ~ [140,7]  Cora
        tx, ty = self.get_test()  # tx - NdArray ~ [1000,1433] ; y - NdArray ~ [1000,7]   Cora
        allx, ally = self.get_all()  # allx - NdArray ~ [1708,1433] ; ally - NdArray ~ [1708,1433]   Cora
        adj = self.get_adj()

        # 重新排序测试集
        test_idx_reorder = self.parse_index_file("{}/ind.{}.test.index".format(self.path, self.dataset))
        test_idx_range = np.sort(test_idx_reorder)


        # Citeseer中包含一部分离散点
        if self.dataset == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extend = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extend[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extend

            ty_extend = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extend[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extend

        # 构建特征矩阵
        # features_np = np.vstack((allx, tx))  # 合成为特征矩阵 [1708,1433] + [1000,1433] -> [2708,1433]  Cora
        features_lil = sp.lil_matrix(sp.vstack((sp.lil_matrix(allx),sp.lil_matrix(tx))))
        features_lil[test_idx_reorder, :] = features_lil[test_idx_range, :]
        features_np_noNor = features_lil.toarray()

        # 重新排序标签
        labels_np = np.vstack((ally, ty))  # 合称为特征矩阵 [1708,7] + [1000,7] -> [2708,7] Cora
        labels_np[test_idx_reorder, :] = labels_np[test_idx_range, :]

        if self.dataset == 'citeseer':
            save_label = np.where(labels_np)[1]

        # 设置训练，测试，验证集
        index_train = np.arange(len(y))  # 140个训练
        index_val = np.arange(len(y), len(y) + 500)  # 140-540验证
        index_test = test_idx_range  # 1708-2708 测试

        # 找到丢失的元素，置为第0类
        def missing_elements(L):
            start, end = L[0], L[-1]
            return sorted(set(range(start, end + 1)).difference(L))
        if self.dataset == 'citeseer':
            L = np.sort(index_test)
            missing = missing_elements(L)
            for element in missing:
                save_label = np.insert(save_label, element, 0)
            labels_np = save_label

        # 如果是Citeseer，将标签转换为oneHot
        if self.dataset == 'citeseer':
            label_class_num = labels_np.max()+1
            labels_np = np.eye(label_class_num)[labels_np]

        # 返回
        return adj, features_np_noNor, labels_np, index_train, index_val, index_test


if __name__ == '__main__':
    data = c_dataset_loader('cora', './Data/Planetoid')
    adj, feat, labels, index_trian, index_val, index_test = data.process_data()
