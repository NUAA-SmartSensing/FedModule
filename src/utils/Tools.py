import random
import torch
import numpy as np
from utils import Random
from torch.utils.data import TensorDataset


def generate_stale_list(step, shuffle, n):
    stale_list = []
    for i in range(n[0]):
        stale_list.append(0)

    bound = 0
    for i in range(1, len(n)):
        for j in range(n[i]):
            while True:
                s = random.randint(bound, bound + step)
                if s != 0:
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


def generate_non_iid_data(x, y, label_lists, data_lists):
    client_datasets = []
    for i in range(len(label_lists)):
        index_list = []
        for j in range(len(label_lists[i])):
            ids = np.flatnonzero(y == label_lists[i][j])
            ids = np.random.choice(ids, data_lists[i][j], replace=False)
            index_list.append(ids)
        index_list = np.hstack(index_list)
        client_x = x[index_list]
        client_y = y[index_list]
        client_datasets.append(TensorDataset(client_x.clone().detach(), client_y.clone().detach()))
    return client_datasets


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
            label_lists.append(label_list)
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
                label_lists.append(label_all_lists[pos : pos + bound])
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
    md.write("实验架构：" + config["global"]["mode"] + "\n")
    md.write("实验数据集: " + config["global"]["data_name"] + "\n")
    md.write("实验服务器模型: " + config["server"]["model_name"] + "\n")
    md.write("聚合算法: " + config["server"]["updater"]["update_name"] + "\n")
    md.write("调度算法: " + config["server"]["scheduler"]["schedule_name"] + "\n")
    md.write("实验客户端模型: " + config["client"]["model_name"] + "\n")
    md.write("客户端数量: " + str(config["global"]["client_num"]) + "\n")
    md.write("全局迭代次数: " + str(config["server"]["epochs"]) + "\n")
    md.write("数据集分布: " + "iid" if config["global"]["iid"] else "non-iid" + "\n")
    md.close()


if __name__ == '__main__':
    print(generate_label_lists_by_step(1, [5, 2, 4, 3, 3, 4], 0, 10, True))
