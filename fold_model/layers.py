import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """
    https://arxiv.org/pdf/1609.02907.pdf
    GCN layer h(l+1) = D^-1/2 * A * D^-1/2 * H(l) * W
    3 different initialization strategy
        - uniform
        - xavier
        - kaiming
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_feat = in_features
        self.out_feat = out_features
        self.weight = nn.Parameter(t.FloatTensor(in_features, out_features))  # 初始化权重
        if bias:
            self.bias = nn.Parameter(t.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        if init == 'uniform':
            print("| Uniform Initialization")
            self.init_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.init_parameters_xavier()
        elif init == 'K=kaiming':
            print("| Kaiming Initialization")
            self.init_parameters_kaiming()
        else:
            raise NotImplementedError

    # uniform - 正态分布
    def init_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # xavier - 每层的激活值保持高斯分布 bias初始为0， Normalize后的参数乘以Rescale=1/sqrt(N)
    def init_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    # kaiming - 更适用于非线性Relu层, Rescale = sqrt(2/n)
    def init_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = t.mm(input, self.weight)
        output = t.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ \
               + "(" + str(self.in_feat) \
               + "->" + str(self.out_feat) + ")"
