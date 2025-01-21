import random

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from utils import Random
from utils.JsonTool import dict_to_list, list_to_dict
from utils import ModuleFindTool

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def generate_iid_data(dataset, clients):
    class_idx = [np.argwhere(dataset.train_labels == y).flatten() for y in range(max(dataset.train_labels)-min(dataset.train_labels)+1)]
    client_idx = [[] for _ in range(clients + 1)]
    for c in class_idx:
        for i, idcs in enumerate(np.array_split(c, clients)):
            client_idx[i] += [idcs]
    client_idx = [np.concatenate(client_idx[i]) for i in range(clients)]
    return client_idx


def generate_non_iid_data(config, dataset, clients, left, right, datasets):
    iid_config = config['global']['iid']
    if "customize" in iid_config.keys() and iid_config["customize"]:
        label_config = iid_config['label']
        data_config = iid_config['data']
        return customize_distribution(label_config, data_config, dataset, clients, left, right, datasets)
    elif 'group_similarity' in iid_config.keys() and iid_config['group_similarity']:
        return group_similarity_distribution(config, dataset, clients, left, right)
    else:
        return dirichlet_distribution(iid_config, dataset, clients, left, right)
    
def group_similarity_distribution(config, dataset, clients, left, right):

    group_manager_config = config['group_manager']
    iid_config = config['global']['iid']
    beta = iid_config['beta']

    group_info = pd.read_csv(group_manager_config['group_method']['params']['group_info'])
    group_num = group_info['Orbit'].unique().__len__()
    clients_per_group = group_info[['Orbit','client_id']].groupby('Orbit').count()

    # 第一步：分组
    # 
    group_distribution = np.random.dirichlet([beta] * group_num, right - left)
    # 对标签进行分类
    class_idx = [np.argwhere(dataset.train_labels == y).flatten() for y in range(right - left)]
    group_idx = [[] for _ in range(group_num)]
    for c, fracs in zip(class_idx, group_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        # 将第k类的样本划分到每个group
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            group_idx[i] += [idcs]
    # counts = []
    # for i in range(len(group_idx)):
    #     count = 0
    #     for group in group_idx[i]:
    #         print(len(group))
    #         count += len(group)
    #     counts.append(count)
    # print(counts)
    

    # 第二步：每组内部分配
    client_idx = [[] for _ in range(clients)]
    for group in group_idx:
        # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例, K个列表（对应于class总数），每个列表记录该类别分配到每个client的比例
        internal_distribution = np.random.dirichlet([beta] * clients_per_group[clients_per_group['Orbit'] == group]['client_id'], len(group))
        for c, fracs in zip(group, internal_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs) * len(c)).astype(int))):
                client_idx[i + group.index(group) * clients_per_group] += [idcs]

    client_idx = [np.concatenate(idcs) for idcs in client_idx if len(idcs) > 0]
    return client_idx







    return client_idx



def customize_distribution(label_config, data_config, dataset, clients, left, right, datasets):
    # 生成label lists
    # 洗牌算法
    label_lists = []
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
    # 保存label至配置文件
    dataset.iid_config['label'] = list_to_dict(label_lists)
    # 生成序列
    return generate_non_iid_dataset(dataset.train_data, dataset.train_labels, label_lists,
                                    data_lists)


def dirichlet_distribution(iid_config, dataset, clients, left, right):
    beta = iid_config["beta"]
    #  label_distribution 是一个 NumPy 数组，形状为 (right-left, clients)
    label_distribution = np.random.dirichlet([beta] * clients, right - left)
    # 使用列表推导，对于每个从 left 到 right 的类别 y，找到 dataset.train_labels 中等于 y 的元素的索引。
    # class_idx 是一个列表，列表中的每个元素都是一个一维 NumPy 数组，包含对应类别的索引。
    # 生成class嵌套列表，每个子列表为同一class对应的数据集的索引
    class_idx = [np.argwhere(dataset.train_labels == y).flatten() for y in range(right - left)]
    client_idx = [[] for _ in range(clients+1)]

    for c, fracs in zip(class_idx, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs) * len(c)).astype(int))):
            client_idx[i] += [idcs]
    client_idx = [np.concatenate(idcs) for idcs in client_idx]
    return client_idx


def generate_non_iid_dataset(x, y, label_lists, data_lists):
    client_idx_list = []
    for i in range(len(label_lists)):
        index_list = []
        for j in range(len(label_lists[i])):
            ids = np.flatnonzero(y == label_lists[i][j])
            ids = np.random.choice(ids, data_lists[i][j], replace=False)
            index_list.append(ids)
        index_list = np.hstack(index_list)
        client_idx_list.append(index_list)
    return client_idx_list


def generate_data_lists(max_size, min_size, num, label_lists):
    data_lists = []
    data_list = generate_data_list(max_size, min_size, num)
    for i in range(len(label_lists)):
        tmp_list = []
        for j in range(len(label_lists[i]) - 1):
            tmp_list.append(data_list[i] // len(label_lists[i]))
        tmp_list.append(data_list[i] - data_list[i] // len(label_lists[i]) * (len(label_lists[i]) - 1))
        data_lists.append(tmp_list)
    return data_lists


def generate_data_list(max_size, min_size, num):
    ans = []
    for _ in range(num):
        ans.append(random.randint(min_size, max_size))
    return ans


def generate_label_lists(label_num_list, left, right, shuffle=False):
    label_lists = []
    if shuffle:
        label_total = 0
        for label_num in label_num_list:
            label_total = label_total + label_num
        epoch = int(label_total // (right - left)) + 1
        label_all_list = []
        for i in range(epoch):
            label_all_list = label_all_list + Random.shuffle_random(left, right)
        pos = 0
        for label_num in label_num_list:
            label_lists.append(label_all_list[pos: pos + label_num])
            pos += label_num
    else:
        labels = range(left, right)
        for label_num in label_num_list:
            label_list = np.random.choice(labels, label_num, replace=False)
            label_lists.append(label_list.tolist())
    return label_lists


def generate_label_lists_by_step(step, num_list, left, right, shuffle=False):
    label_lists = []
    bound = 1
    if shuffle:
        label_total = 0
        label_all_lists = []
        for i in num_list:
            label_total += bound * i
            bound += step
        bound = 1
        epoch = int(label_total // (right - left)) + 1
        for i in range(epoch):
            label_all_lists += Random.shuffle_random(left, right)
        pos = 0
        for i in range(len(num_list)):
            for j in range(num_list[i]):
                label_lists.append(label_all_lists[pos: pos + bound])
                pos = pos + bound
            bound += step
    else:
        labels = range(left, right)
        for i in range(len(num_list)):
            for j in range(num_list[i]):
                s = np.random.choice(labels, bound, replace=False)
                label_lists.append(s.tolist())
            bound += step
    return label_lists
