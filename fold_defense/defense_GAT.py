import scipy as sp
import torch
import torch as t
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCNJaccard
from deeprobust.graph.utils import *
from deeprobust.graph.defense import GAT
from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg
import argparse
from config import opts
from fold_data import dataset
from deeprobust.graph import utils


model_path = "../checkpoint"

opt = opts()
opt.data_path = r"../fold_data/Data/Planetoid"
opt.model_path = "../checkpoint"
opt.dataset = 'citeseer'
opt.model = 'GCN'
rand_seed = 11

# 读取数据集
data_load = dataset.c_dataset_loader(opt.dataset, opt.data_path)
base_adj, base_feat,_,_,_,_  = data_load.process_data()
if opt.dataset == 'cora' and rand_seed<12:
    attack_info = t.load("../fold_attack/GCN/Nettack/attack_info_{}".format(rand_seed))
else:
    attack_info = t.load("../fold_attack/GCN/Nettack/attack_info_{}_{}".format(opt.dataset,rand_seed))


adj_per = attack_info['adj_per'][15].A
model = attack_info['surrogate']

idx_train = attack_info['idx_train']
idx_val = attack_info['idx_val']
idx_test = attack_info['idx_test']

labels = model.labels.detach().cpu().numpy()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make sure you use the same data splits as you generated attacks
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# check output
adj_per_nor_t = t.from_numpy(adj_per).float().cuda()

hidden = t.from_numpy(base_feat).float().cuda()
hidden = adj_per_nor_t @ hidden @ model.gc1.weight
if model.with_bias:
    hidden = hidden + model.gc1.bias
if model.with_relu:
    hidden = F.relu(hidden)
hidden = adj_per_nor_t @ hidden @ model.gc2.weight
if model.with_bias:
    hidden = hidden + model.gc2.bias

output = F.log_softmax(hidden, dim=1)
acc_after_attack = utils.accuracy(output[idx_test], labels[idx_test])

# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
# Or we can just use setting='prognn' to get the splits
# data = Dataset(root='./tmp/', name=args.dataset, setting='prognn')
# adj, features, labels_1 = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


# load pre-attacked graph
# perturbed_data = PrePtbDataset(root='./tmp/',
#         name=args.dataset,
#         attack_method='meta',
#         ptb_rate=args.ptb_rate)

# use data splist provided by prognn
data = Dataset(root='./tmp/', name=args.dataset, setting='prognn')
data.adj = sp.csr_matrix(adj_per)
data.features = sp.csr_matrix(base_feat)
data.labels = labels
data.idx_train = idx_train
data.idx_val = idx_val
data.idx_test =idx_test

perturbed_adj = sp.csr_matrix(adj_per,dtype=float)
features = sp.csr_matrix(base_feat, dtype=float)

# Setup Defense Model
gat = GAT(nfeat=features.shape[1],
      nhid=8, heads=8,
      nclass=labels.max().item() + 1,
      dropout=0.5, device=device)
gat = gat.to(device)


pyg_data = Dpr2Pyg(data)
gat.fit(pyg_data, verbose=True) # train with earlystopping
gat.test()

print("rand_seed = {}".format(rand_seed))



# print('=== testing GCN-Jaccard on perturbed graph ===')
# model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.01, verbose=False)
# model.eval()
# output = model.test(idx_test)
#
# print("rand_seed = {}".format(rand_seed))