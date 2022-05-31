import random
import torch
import numpy as np


def generate_stale_list(step, shuffle, n1, *n):
    stale_list = []
    for i in range(n1):
        stale_list.append(0)

    bound = 0
    for i in n:
        for j in range(i):
            while True:
                s = random.randint(bound, bound + step)
                if s != 0:
                    break
            stale_list.append(s)
        bound += step
    if shuffle:
        np.random.shuffle(stale_list)
    return stale_list


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
