# import copy
# import random
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import cifar10
# import numpy as np
# import scipy.stats as stats
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# import matplotlib.pyplot as plt
#
# import Datasets.MNIST as MNIST
# import Datasets.FashionMNIST as FashionMNIST
# import Datasets.CIFAR10 as CIFAR10
# import Datasets.CIFAR100 as CIFAR100
# import Datasets.ImageNette as ImageNette
#
# from MainFrame.Client import Client
#
import random


def generate_stale_list(n1, *n):
    stale_list = []
    for i in range(n1):
        stale_list.append(0)

    bound = 0
    for i in n:
        for j in range(i):
            stale_list.append(random.randint(bound, bound + 10))
        bound += 10
    return stale_list


if __name__ == '__main__':
    alist = generate_stale_list(10, 5, 5)
    print(alist)
