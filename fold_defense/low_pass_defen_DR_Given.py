"""
Use the adj_per to test defense capability
"""
import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
from deeprobust.graph.defense import GCNJaccard
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
# Or we can just use setting='prognn' to get the splits
data = Dataset(root=r'D:\Python Project\defense\Low_pass_defense\fold_defense\tmp\\', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
print('==================')
print('=== load graph perturbed by Zugner metattack (under seed 15) ===')
perturbed_data = PrePtbDataset(root=r'D:\Python Project\defense\Low_pass_defense\fold_defense\tmp\\',
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)
perturbed_adj = perturbed_data.adj

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Setup GCN Model
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)

model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=True)
# # using validation to pick model
# model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)

# Setup Defense Model
model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, device=device)

model = model.to(device)

print('=== testing GCN-Jaccard on perturbed graph ===')
model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.01)
model.eval()
output = model.test(idx_test)

# from fold_defense import low_pass_adj_sym
# from fold_defense import simGraph_init
# base_feat = features.A
# neighbor_num = 20
# adj_per = perturbed_adj.A
# base_adj = adj.A
# from deeprobust.graph import utils
# import torch as t
#
# # do the low_pass
# sim_adj_h0 = simGraph_init(base_feat, p_neighbor_num=neighbor_num, p_layer_id=0)
#
# modified_adj_h0 = low_pass_adj_sym(adj_per, sim_adj_h0, base_feat, p_filter_value=1)
#
# modified_adj_h0_t = t.from_numpy(modified_adj_h0).float().cuda()
#
# hidden_low_pass_0 = t.from_numpy(base_feat).float().cuda()
# hidden_low_pass_h0 = modified_adj_h0_t @ hidden_low_pass_0 @ model.gc1.weight.cuda()
# if model.with_bias:
#     hidden_low_pass_h0 = hidden_low_pass_h0 + model.gc1.bias
# if model.with_relu:
#     hidden_low_pass_h0 = F.relu(hidden_low_pass_h0)
#
#
# sim_adj_h1 = simGraph_init(hidden_low_pass_h0.detach().cpu().numpy(), p_neighbor_num=neighbor_num, p_layer_id=1)
#
# modified_adj_h1 = low_pass_adj_sym(base_adj, sim_adj_h1, hidden_low_pass_h0.detach().cpu().numpy(), p_filter_value=1)
# modified_adj_h1_t = t.from_numpy(modified_adj_h1).float().cuda()
# hidden_low_pass_h1 = modified_adj_h1_t @ hidden_low_pass_h0 @ model.gc2.weight.cuda()
# if model.with_bias:
#     hidden_low_pass_h1 = hidden_low_pass_h1 + model.gc2.bias
#
# h2_softmax = F.log_softmax(hidden_low_pass_h1, dim=1)
#
# acc_after_low_pass = utils.accuracy(h2_softmax[idx_test], labels[idx_test])