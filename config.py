import numpy as np


class opts():
    """模型的基础设定"""
    lr = 0.0045  # 学习率
    num_hiden_layer = 8  # 隐层数量
    init_type = 'xavier'  # 标准化方法 [norm, xavier, kaiming]
    drop_out = 0.6  # drop out的概率 防过拟合
    optim = "adam"  # 优化器方法 [adam, sgd]
    weight_decay = 5e-4  # 权重损失
    dataset = 'cora'  # 数据集 [cora, citeseer, pubmed]
    epoch = 800  # 训练轮数
    lr_decay_epoch = 5000  # 每轮训练降低一定的学习率
    model = 'SGC'
    feature_Nor = True  # 决定是否使用特征标准化 （按行标准化）
    data_path = "./fold_data/Data/Planetoid"  # 数据集默认路径
    data_path_graphRobust = r'D:\Learning Project\python_project\low_pass_def\fold_data\Data\Graphrobust'
    model_path = "./checkpoint"  # 模型默认路径
    np_random_seed = 100  # numpy的随机因子
    use_cuda = True
    is_save = False

    """攻击的一些设定"""
    top_n_feat = 5  # 决定返回多少个特征值

    is_att_minus_grad = 1   # 决定是使用哪种方式进行攻击

    fake_node_num = 14  # 设定多少个假节点
    is_add_fake_feat = False  # 是否添加后门特征
    limit_fake_feat = 25  # 最多添加多少个假特征
    is_success_rate_control = False     # 是否要进行跳出循环判定
    attack_success_rate_control = 0.95  # 在攻击成功率为0.8的时候跳出循环
    is_multi_feat_decay = True    # 是否多特征梯度衰减
    multi_feat_decay = 0.5      # 添加第二个特征时的衰减

    target_node_num = 50  # 一共攻击K个节点
    target_train_node_num = 20  # 一共用20个节点作为训练
    target_test_node_num = target_node_num - target_train_node_num  # 一共用10个节点作为测试
    single_attack_test_num = 30     # 单节点攻击测试用的节点数量

    # 后门节点和目标节点
    backdoor_node_idx = 1767   # 决定后门节点的idx
    target_node_idx = 2021  # 只适用于 Attack_Single_Node 决定攻击的目标节点
    target_node_class = 2  # 只适用于 Attack_Class_Node 决定攻击的所属类

    # 设置标志
    flag_att_single_node = 0    # 置1则进行单节点攻击
    flag_att_single_node_group_test = 0     # 置1则进行单节点多测试攻击
    flag_att_class_nodes = 0     # 置1则进行单类节点攻击
    flag_att_group_nodes = 2    # 置1则进行全图节点攻击

    # 记录邻接矩阵
    adj_num = {"cora":2708,"citeseer":3327, "pubmed":19717}
    total_node_num = adj_num[dataset] + fake_node_num
    # normed_adj_single_node = np.zeros([total_node_num, total_node_num])
    # normed_adj_single_node_group_np = np.zeros([single_attack_test_num, total_node_num, total_node_num])
    # normed_adj_class_np = np.zeros([target_train_node_num, total_node_num, total_node_num], np.float16)
    # normed_adj_group_np = np.zeros([target_train_node_num, total_node_num, total_node_num], np.float16)
    normed_adj_idx = 0      # 记录当前为第几个输入的idx


    # # 对于已经进行过邻接矩阵变换的结果进行保存 D^-0.5 * A D^-0.5
    # # 总共如果有K个目标节点，则一共有K个拉普拉斯变换后的矩阵结果
    # is_adj_normed_single = np.zeros(single_attack_test_num)
    # normed_adj = np.ndarray