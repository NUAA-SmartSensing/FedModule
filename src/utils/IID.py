import utils
import numpy as np
from utils.JsonTool import dict_to_list, list_to_dict
from utils.Tools import generate_label_lists_by_step, generate_label_lists, generate_data_lists
from torch.utils.data import TensorDataset


def generate_non_iid_data(iid_config, dataset, clients, left, right, target_dataset=None, params={}):
    if "customize" in iid_config.keys() and iid_config["customize"]:
        label_config = iid_config['label']
        data_config = iid_config['data']
        customize_distribution(label_config, data_config, dataset, clients, left, right, target_dataset if target_dataset is not None else TensorDataset, params)
    else:
        dirichlet_distribution(iid_config, dataset, clients, left, right, target_dataset if target_dataset is not None else TensorDataset, params)


def customize_distribution(label_config, data_config, dataset, clients, left, right, target_dataset, params):
    # 生成label lists
    # 洗牌算法
    shuffle = False
    if "shuffle" in label_config.keys() and label_config["shuffle"]:
        shuffle = True
    if isinstance(label_config, dict):
        # step
        if "step" in label_config.keys():
            label_lists = generate_label_lists_by_step(label_config["step"], label_config["list"], left, right, shuffle)
        # {list:[]}
        elif "list" in label_config.keys():
            label_lists = generate_label_lists(label_config["list"], left, right, shuffle)
        # {[],[],[]}
        else:
            label_lists = dict_to_list(label_config)
    # 生成data lists
    # {}
    if len(data_config) == 0:
        size = dataset.train_data_size // clients
        data_lists = generate_data_lists(size, size, clients, label_lists)
    # max,min
    else:
        data_lists = generate_data_lists(data_config["max"], data_config["min"], clients, label_lists)
    # 生成datasets
    dataset.datasets = utils.Tools.generate_non_iid_data(dataset.train_data, dataset.train_labels, label_lists,
                                                         data_lists, target_dataset, params)
    # 保存label至配置文件
    dataset.iid_config['label'] = list_to_dict(label_lists)


def dirichlet_distribution(iid_config, dataset, clients, left, right, target_dataset, params):
    beta = iid_config["beta"]
    label_distribution = np.random.dirichlet([beta] * clients, right - left)
    class_idx = [np.argwhere(dataset.train_labels == y).flatten() for y in range(right-left)]
    client_idx = [[] for _ in range(clients)]
    for c, fracs in zip(class_idx, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idx[i] += [idcs]
    client_idx = [np.concatenate(idcs) for idcs in client_idx]
    client_datasets = []
    for i in range(len(client_idx)):
        y = dataset.train_labels[client_idx[i]]
        x = dataset.train_data[client_idx[i]]
        client_datasets.append(target_dataset(x.clone().detach(), y.clone().detach(), **params))
    dataset.datasets = client_datasets
