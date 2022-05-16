import random
import torch
import numpy as np


def generate_stale_list(filename, step, shuffle, n1, *n):
    stale_list = []
    for i in range(n1):
        stale_list.append(0)

    bound = 0
    for i in n:
        for j in range(i):
            stale_list.append(random.randint(bound, bound + step))
        bound += step
    if shuffle:
        np.random.shuffle(stale_list)
    saveAns(filename, stale_list)


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
    save = open("../results/" + filename, "w")
    for w in result:
        save.write(str(w) + "\n")
    save.close()


if __name__ == '__main__':
    generate_stale_list("", 15, 20, 15)
