import numpy as np


def random_select_train_test(p_labels: np.ndarray, p_pick_num_each, p_rand_seed=0):
    total_class_num = p_labels.max(axis=0) + 1
    res_train = np.array([])
    res_val = np.array([])
    res_test = np.arange(p_labels.shape[0])
    for i in range(total_class_num):
        candidates = np.where(p_labels == i)[0]
        np.random.seed(p_rand_seed)
        chosen = np.random.choice(candidates, p_pick_num_each * 2, replace=False)
        res_train = np.union1d(res_train, chosen[:p_pick_num_each])
        res_val = np.union1d(res_val, chosen[p_pick_num_each:])

    res_test = np.setdiff1d(res_test, res_train)
    res_test = np.setdiff1d(res_test, res_val)
    return res_train, res_val, res_test
