import json
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.DatasetUtils import CustomDataset


def generate_stale_list(step, shuffle, n):
    stale_list = []
    for i in range(n[0]):
        stale_list.append(0)

    bound = 0
    for i in range(1, len(n)):
        for j in range(n[i]):
            while True:
                s = random.randint(bound, bound + step)
                if s != bound:
                    break
            stale_list.append(s)
        bound += step
    if shuffle:
        np.random.shuffle(stale_list)
    return stale_list


def get_stale_list(filename):
    stale_list = []
    with open(filename) as f:
        for line in f:
            stale_list.append(int(line))
    return stale_list


def get_order_as_tuple(filename):
    mylist = []
    with open(filename) as f:
        for line in f:
            mylist.append(int(line))
    return torch.tensor(mylist)


def saveOrder(filename, result):
    save = open("../results/" + filename, "w")
    for w in result:
        save.write(str(w.numpy()) + "\n")
    save.close()


def saveAns(filename, result):
    save = open(filename, "w")
    save.write(str(result))
    save.close()


def saveJson(filename, result, indent=4):
    try:
        with open(filename, 'w', encoding='utf8') as fp:
            json.dump(result, fp, indent=indent)
    except shutil.SameFileError:
        pass


def result_to_markdown(filename, config):
    md = open(filename, "w")
    md.write("实验数据集: " + config["global"]["dataset"]["path"] + "\n")
    md.write("实验服务器模型: " + config["server"]["model"]["path"] + "\n")
    md.write("聚合算法: " + config["server"]["updater"]["path"] + "\n")
    md.write("调度算法: " + config["server"]["scheduler"]["path"] + "\n")
    md.write("实验客户端模型: " + config["client"]["model"]["path"] + "\n")
    md.write("客户端数量: " + str(config["global"]["client_num"]) + "\n")
    md.write("全局迭代次数: " + str(config["server"]["epochs"]) + "\n")
    md.write("数据集分布: " + "iid" if isinstance(config["global"]["iid"], bool) else "non-iid" + "\n")
    md.close()


def getJson(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        config = json.load(fp)
    return config


def to_cpu(data):
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cpu(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data


def to_dev(data, dev):
    if not torch.cuda.is_available():
        return data
    if isinstance(data, dict):
        return {k: to_dev(v, dev) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_dev(v, dev) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_dev(v, dev) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(dev)
    else:
        return data


def dict_to_list(src):
    des = []
    for _, v in src.items():
        des.append(v)
    return des


def list_to_dict(src):
    des = {}
    for i in range(len(src)):
        des[i] = src[i]
    return des


def random_seed_set(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def generate_random_seed():
    seed = int(time.time() * 1000) % 2147483647
    return seed


def _read_data(dataset):
    data = []
    targets = []
    dl = DataLoader(dataset, batch_size=1)
    for x, y in dl:
        data.append(x[0])
        targets.append(y[0])
    data = torch.stack(data)
    targets = torch.stack(targets)
    data.share_memory_()
    targets.share_memory_()
    return data, targets


def send_dataset(train_dataset, test_dataset, message_queue, global_config):
    # 预加载
    if 'dataset_pre_load' in global_config and global_config['dataset_pre_load']:
        data, targets = _read_data(train_dataset)
        message_queue.set_train_dataset(CustomDataset(data, targets))
        data, targets = _read_data(test_dataset)
        message_queue.set_test_dataset(CustomDataset(data, targets))
    # 静态加载
    else:
        message_queue.set_train_dataset(train_dataset)
        message_queue.set_test_dataset(test_dataset)
    return train_dataset, test_dataset


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # 递归转换嵌套的字典
                value = DictToObject(value)
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"
