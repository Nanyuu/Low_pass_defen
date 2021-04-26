import scipy as sp
from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset
from deeprobust.graph.targeted_attack import Nettack
from config import opts
from fold_data import dataset
from fold_util import F_Info
from attack_util import random_select_train_test
import numpy as np
from tqdm import tqdm
import torch as t
from deeprobust.graph.utils import accuracy
import scipy.sparse as sp

opt = opts()
rand_seed = 3
device = 'cuda:0'

data_load = dataset.c_dataset_loader(opt.dataset, ".{}".format(opt.data_path))
adj, features, label, _,_,_ = data_load.process_data()
labels = F_Info.F_one_hot_to_label(label)

# data = Dataset(root='{}/'.format(opts.data_path_graphRobust), name='cora')
# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# idx_unlabeled = np.union1d(idx_val, idx_test)
'''edit 2021/4/22 random_choose 20 node from each class as the train and validation samples '''
idx_train, idx_val, idx_test = random_select_train_test(labels, 20, p_rand_seed=rand_seed)
idx_unlabeled = np.union1d(idx_val, idx_test)

# surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=True,
                with_bias=True, device='cuda:0').cuda()
# train model
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

def test_acc(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)

    gcn.eval()
    output = gcn.predict()
    # probs = t.exp(output[[target_node]])[0]
    # print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("\nOverall test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def multi_test_poison():
    # test nodes on poisoning attack
    cnt = 0
    data_load = dataset.c_dataset_loader(opt.dataset, ".{}".format(opt.data_path))
    adj, features, label, _, _, _ = data_load.process_data()
    labels = F_Info.F_one_hot_to_label(label)
    degrees = adj.sum(0)
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)
    total_att_node = 1000      # 攻击的节点数目

    # node_list = np.random.choice(np.arange(adj.shape[0]), 10, False)
    node_list = np.random.choice(idx_test, total_att_node, replace=False)

    # 每100个节点记录一次平均asr
    adj_pert_record = {}
    acc_pert_record = {}

    num = len(node_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
    acc = 0
    target_node_id = 1
    for target_node_idx in tqdm(node_list):
        n_perturbations = 2
        model = Nettack(surrogate, nnodes= adj.shape[0], attack_structure=True, attack_features=False,device='cuda:0')
        model = model.to('cuda:0')
        model.attack(features, adj,labels,target_node_idx, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        if target_node_id % 50 == 0:
            print("target_node_idx : {}\n".format(target_node_idx))
            acc = test_acc(modified_adj, features, target_node_idx)
        if target_node_id % 100 == 0:
            print("-------------Recoding---------\n")
            adj_pert_record[target_node_id/100] = sp.csr_matrix(modified_adj)
            acc_pert_record[target_node_id/100] = acc
            pass
        adj = modified_adj
        target_node_id = target_node_id +1
    info_collect = {}
    info_collect['adj_per'] = adj_pert_record
    info_collect['surrogate'] = surrogate
    info_collect['idx_train'] = idx_train
    info_collect['idx_val'] = idx_val
    info_collect['idx_test'] = idx_test
    info_collect['acc_after_att'] = acc_pert_record
    info_collect['total_att_node'] = total_att_node
    info_collect['random_seed'] = rand_seed
    return info_collect

if __name__ == '__main__':
    attack_info = multi_test_poison()
    t.save(attack_info, './GCN/Nettack/attack_info_{}'.format(rand_seed))

