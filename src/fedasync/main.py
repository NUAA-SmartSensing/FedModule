import datetime
import os
import threading

import sys
import AsyncServer
from shutil import copyfile
from utils.Tools import *
from utils.ConfigManager import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        config_file = "config.json"
    else:
        config_file = sys.argv[1]
    config = getConfig(config_file)
    global_config = config['global']
    server_config = config['server']
    client_config = config['client']

    # 实验路径相关
    if not global_config["experiment"].endswith("/"):
        global_config["experiment"] = global_config["experiment"] + "/"
    if not os.path.exists("../results/" + global_config["experiment"]):
        os.makedirs("../results/" + global_config["experiment"])
    stale_path = "../results/"+global_config["experiment"]+"stale.txt"
    if os.path.exists("stale_path"):
        client_staleness_list = get_stale_list("../results/"+global_config["experiment"]+"/stale.txt")
    else:
        client_staleness_list = []
        for i in range(global_config["client_num"]):
            client_staleness_list.append(0)
    client_config["stale_list"] = client_staleness_list

    start_time = datetime.datetime.now()

    accuracy_lists = []
    loss_lists = []

    async_server = AsyncServer.AsyncServer(global_config, server_config, client_config)

    async_server.run()
    print("")

    accuracy_list = async_server.get_accuracy_list()

    del async_server

    print("Thread count =", threading.activeCount())
    print(*threading.enumerate(), sep="\n")

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    # 保存结果
    saveAns("../results/" + global_config["experiment"] + "accuracy.txt", list(accuracy_list))
    saveAns("../results/" + global_config["experiment"] + "time.txt", [end_time - start_time])
    # 保存配置文件
    copyfile("../../" + config_file, "../results/" + global_config["experiment"] + "config.json")
