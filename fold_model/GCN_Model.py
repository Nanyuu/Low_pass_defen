import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
# from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        gc -> ReLU -> dropout -> gc -> log_SoftMax
        :param x: 特征矩阵
        :param adj: 邻接矩阵
        :return: 标签值 log_softMax
        """
        h1_gc = self.gc1(x, adj)
        h1_relu = F.relu(h1_gc)
        h1_dropout = F.dropout(h1_relu, self.dropout, training=self.training)
        h2_gc = self.gc2(h1_dropout, adj)
        h2_softmax = F.log_softmax(h2_gc, dim=1)

        return h2_softmax

'''
class GCN_PyG(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN_PyG, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)

    def forward(self, x, edge_index):
        h1_gc = self.gc1(x, edge_index)
        h1_relu = F.relu(h1_gc)
        h1_dropout = F.dropout(h1_relu,0.6, training=self.training)
        h2_gc = self.gc2(h1_dropout, edge_index)
        h2_log_softmax = F.log_softmax(h2_gc, dim=1)

        return h2_log_softmax
        '''
