from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset
from deeprobust.graph.targeted_attack import Nettack

data = Dataset(root='/tmp/', name='cora')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid= 16, dropout=0, with_relu=True, with_bias=True, device='cpu').cuda()
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

target_node = 0
model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu').cuda()

model.attack(features, adj, labels, target_node, n_perturbations=2)
modified_adj = model.modified_adj
