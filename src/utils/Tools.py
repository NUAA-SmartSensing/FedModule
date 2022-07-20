import random
import torch
import numpy as np
from torch.utils.data import TensorDataset


def generate_stale_list(step, shuffle, n1, *n):
    stale_list = []
    for i in range(n1):
        stale_list.append(0)

    bound = 0
    for i in n:
        for j in range(i):
            while True:
                s = np.random.randint(bound, bound + step)
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
            ids = np.floatnonzero(y==label_lists[i][j])
            ids = np.random.choice(ids, data_lists[i][j], replace=False)
            index_list.append(ids)
        index_list = np.hstack(index_list)
        client_x = x[index_list]
        client_y = y[index_list]
        client_datasets.append(TensorDataset(torch.tensor(client_x), torch.tensor(client_y)))
    return client_datasets


def generate_data_lists(max_size, min_size, num, label_lists):
    data_lists = []
    data_list = generate_data_list(max_size, min_size, num)
    for i in range(len(label_lists)):
        tmp_list = []
        for j in range(len(label_lists[i]) - 1):
            tmp_list.append(data_list[i]//len(label_lists[i]))
        tmp_list.append(data_list[i]-data_list[i]//len(label_lists[i])*(len(label_lists[i])-1))
        data_lists.append(tmp_list)
    return data_lists


def generate_data_list(max_size, min_size, num):
    ans = []
    for _ in range(num):
        ans.append(random.randint(min_size, max_size))
    return ans


def generate_label_lists(label_num_list, left, right):
    label_lists = []
    for label_num in label_num_list:
        label_list = []
        for i in range(label_num):
            while True:
                y = random.randint(left, right)
                if y not in label_list:
                    label_list.append(y)
                    break
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
    generate_stale_list(15, True, [1,2,3])
