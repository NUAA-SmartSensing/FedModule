import random

import numpy as np
import torch


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


def to_cpu(data):
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data


def to_dev(data, dev):
    if isinstance(data, dict):
        return {k: to_dev(v, dev) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_dev(v, dev) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.to(dev)
    else:
        return data
