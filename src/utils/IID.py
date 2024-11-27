import copy
import random
from collections import Counter

import numpy as np

from utils.Tools import dict_to_list


def print_dist(index_list, labels):
    for i, client_index_list in enumerate(index_list):
        total = len(client_index_list)
        print(f'({i}: {total},', end=' ')
        counts = Counter(labels[client_index_list].tolist())
        print(dict(counts))


def generate_data(iid_config, labels, clients_num):
    if isinstance(iid_config, bool):
        print("generating iid data...")
        index_list = generate_iid_data(labels, clients_num)
    else:
        print("generating non_iid data...")
        index_list = generate_non_iid_data(iid_config, labels, clients_num)
    print_dist(index_list, labels)
    return index_list


def generate_iid_data(labels, clients_num):
    class_lst = set(labels)
    class_idx = [np.argwhere(labels == y).flatten() for y in class_lst]
    client_idx = [[] for _ in range(clients_num + 1)]
    for c in class_idx:
        for i, idcs in enumerate(np.array_split(c, clients_num)):
            client_idx[i] += [idcs]
    client_idx = [np.concatenate(client_idx[i]) for i in range(clients_num)]
    return client_idx


def generate_non_iid_data(iid_config, labels, clients_num):
    classes = set(labels)
    if "customize" in iid_config.keys() and iid_config["customize"]:
        label_config = iid_config['label']
        data_config = iid_config['data']
        return customize_distribution(label_config, data_config, labels, clients_num, classes)
    else:
        return dirichlet_distribution(iid_config["beta"], labels, clients_num, classes)


def customize_distribution(label_config, data_config, labels, clients_num, classes):
    # 生成label lists
    # 洗牌算法
    label_lists = []
    shuffle = False
    if "shuffle" in label_config.keys() and label_config["shuffle"]:
        shuffle = True
    if isinstance(label_config, dict):
        # step
        if "step" in label_config.keys():
            label_lists = generate_label_lists_by_step(label_config["step"], label_config["list"], classes, shuffle)
        # {list:[]}
        elif "list" in label_config.keys():
            label_lists = generate_label_lists(label_config["list"], classes, shuffle)
        # {[],[],[]}
        else:
            label_lists = dict_to_list(label_config)
    # 生成data lists
    # {}
    if len(data_config) == 0:
        size = len(labels) // clients_num
        data_lists = generate_data_lists(size, size, clients_num, label_lists)
    elif isinstance(data_config, dict):
        # max,min
        if "max" in data_config or "min" in data_config:
            data_max = data_config["max"] if "max" in data_config else len(labels)
            data_min = data_config["min"] if "min" in data_config else 0
            data_lists = generate_data_lists(data_max, data_min, clients_num, label_lists)
        # step
        elif "step" in data_config:
            shuffle = data_config["shuffle"] if "shuffle" in data_config else True
            num_random = data_config["random"] if "random" in data_config else True
            data_lists = generate_data_lists_by_step(data_config["step"], data_config["list"], label_lists, num_random, shuffle)
        else:
            data_lists = [None for _ in range(len(label_lists))]
            for k, v in data_config.items():
                data_lists[int(k)] = v
    # []
    elif isinstance(data_config, list):
        data_lists = average_data_list(data_config, label_lists)
    else:
        raise ValueError("data_config error")
    # 生成序列
    return generate_non_iid_dataset(labels, label_lists, data_lists)


def dirichlet_distribution(beta, labels, clients_num, classes):
    label_distribution = np.random.dirichlet([beta] * clients_num, len(classes))
    class_idx = [np.argwhere(labels == y).flatten() for y in classes]
    client_idx = [[] for _ in range(clients_num+1)]
    for c, fracs in zip(class_idx, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs) * len(c)).astype(int))):
            client_idx[i] += [idcs]
    client_idx = [np.concatenate(idcs) for idcs in client_idx]
    return client_idx


def generate_non_iid_dataset(y, label_lists, data_lists):
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
    data_list = generate_data_list(max_size, min_size, num)
    data_lists = average_data_list(data_list, label_lists)
    return data_lists


def average_data_list(data_list, label_lists):
    data_lists = []
    for i in range(len(label_lists)):
        data_num = data_list[i] // len(label_lists[i])
        tmp_list = [data_num] * (len(label_lists[i]) - 1)
        tmp_list.append(data_list[i] - data_list[i] // len(label_lists[i]) * (len(label_lists[i]) - 1))
        data_lists.append(tmp_list)
    return data_lists


def generate_data_list(max_size, min_size, num):
    ans = []
    for _ in range(num):
        ans.append(random.randint(min_size, max_size))
    return ans


def generate_data_lists_by_step(step, num_list, label_lists, num_random=True, shuffle=True):
    data_lists = []
    data_list = []
    bound = 0
    for num in num_list:
        data_list += [bound] * num
        bound += step
    if shuffle:
        random.shuffle(data_list)
    if num_random:
        for num, label_list in zip(data_list, label_lists):
            data_lists.append(random_partition(num, len(label_list)))
    else:
        data_lists = average_data_list(data_list, label_lists)
    return data_lists


def generate_label_lists(label_num_list, classes, shuffle=False):
    label_lists = []
    classes_copy = copy.copy(list(classes))
    if shuffle:
        label_total = sum(label_num_list)
        epoch = int(label_total // len(classes)) + 1
        label_all_list = []
        for i in range(epoch):
            random.shuffle(classes_copy)
            label_all_list = label_all_list + classes_copy
        pos = 0
        for label_num in label_num_list:
            label_lists.append(label_all_list[pos: pos + label_num])
            pos += label_num
    else:
        for label_num in label_num_list:
            label_list = np.random.choice(classes_copy, label_num, replace=False)
            label_lists.append(label_list.tolist())
    return label_lists


def generate_label_lists_by_step(step, num_list, classes, shuffle=False):
    label_lists = []
    bound = 1
    label_list = copy.copy(list(classes))
    if shuffle:
        label_total = 0
        label_all_lists = []
        for i in num_list:
            label_total += bound * i
            bound += step
        bound = 1
        epoch = int(label_total // len(classes)) + 1
        for i in range(epoch):
            random.shuffle(label_list)
            label_all_lists += label_list
        pos = 0
        for i in range(len(num_list)):
            for j in range(num_list[i]):
                label_lists.append(label_all_lists[pos: pos + bound])
                pos = pos + bound
            bound += step
    else:
        for i in range(len(num_list)):
            for j in range(num_list[i]):
                s = np.random.choice(label_list, bound, replace=False)
                label_lists.append(s.tolist())
            bound += step
    return label_lists


def random_partition(n, k):
    """
    Randomly partition an integer n into k non-negative integers.

    Args:
        n: The integer to partition into.
        k: The number of integers to partition into.

    Returns:
        A list of k non-negative integers whose sum is n. If k > n or k <= 0, returns None.
    """
    if k > n or k <= 0:
        return None

    partitions = sorted(random.choices(range(n + 1), k=k - 1))
    result = [partitions[0]]
    for i in range(1, k - 1):
        result.append(partitions[i] - partitions[i - 1])
    result.append(n - partitions[-1])
    return result
