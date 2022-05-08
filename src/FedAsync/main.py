import datetime
import os
import threading

import torch

import AsyncServer
from utils.Tools import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_TYPE = "MNIST"
MODEL_NAME = "CNN"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "VGG13"

RESULT_FILE_NAME = "FashionMNIST"

EPOCHS = 300

PRE_TRAIN_EPOCH = 0

CLIENT_NUMBER = 50
USE_CLIENT_NUMBER = 100

USE_IID_CLIENTS = 2

BATCH_SIZE = 50
CLIENT_RATIO = 0.1
E = 5
R_RHO = 7
SCHEDULER_INTERVAL = 10
CHECK_IN_INTERVAL = 600  # 200
CHECK_IN_NUM = 200  # 100

CLIENT_STALENESS_SETTING = [2, 128, 63, 40]  # lower, upper, mu, sigma

# USE_GPU_ID = [0, 1, 2, 3, 4, 5]
USE_GPU_ID = [0]

if DATA_TYPE == "MNIST" or "FashionMNIST" or "CIFAR10" or "CIFAR100":
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)
else:
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    accuracy_lists = []
    loss_lists = []

    alpha_list = [1]
    # client_staleness_list = []
    # for i in range(CLIENT_NUMBER):
    #     client_staleness_list.append(0)
    # client_staleness_list = Tools.generate_stale_list(10, 10, 10, 10, 5, 5)
    # saveAns("IID/MNIST/stale.txt", client_staleness_list)
    client_staleness_list = get_stale_list("../results/IID/MNIST/stale.txt")
    async_server = AsyncServer.AsyncServer(DATA_TYPE, MODEL_NAME, CLIENT_RATIO, CLIENT_NUMBER,
                                           BATCH_SIZE, E,
                                           EPOCHS, SCHEDULER_INTERVAL, CHECK_IN_INTERVAL,
                                           CHECK_IN_NUM,
                                           client_staleness_list)

    async_server.run()
    print("")

    accuracy_list, loss_list = async_server.get_accuracy_and_loss_list()
    accuracy_lists.append(accuracy_list)
    loss_lists.append(loss_list)

    del async_server

    print("Thread count =", threading.activeCount())
    print(*threading.enumerate(), sep="\n")

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    try:
        saveAns("IID/MNIST/fedavg_accuracy.txt", accuracy_lists)
        saveAns("IID/MNIST/fedavg_time.txt", [end_time - start_time])
    except:
        print(accuracy_list)
        print(end_time - start_time)
    # ResultManager.handle_result(RESULT_FILE_NAME, EPOCHS, len(alpha_list), curve_name_list, accuracy_lists, loss_lists)
