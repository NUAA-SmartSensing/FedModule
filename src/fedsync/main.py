import copy
import datetime
import os
import shutil
import threading
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.Tools import *
from utils.ConfigManager import *
from exception import ClientSumError
import SyncServer

if __name__ == '__main__':
    # 创建结果文件夹
    if not os.path.exists("../results"):
        os.mkdir("../results")

    # 配置文件读取
    if len(sys.argv) < 2:
        config_file = "config.json"
    else:
        config_file = sys.argv[1]

    config = getConfig(config_file)
    global_config = copy.deepcopy(config['global'])
    server_config = copy.deepcopy(config['server'])
    client_config = copy.deepcopy(config['client'])
    manager_config = copy.deepcopy(config['client_manager'])

    # 实验路径相关
    if not global_config["experiment"].endswith("/"):
        global_config["experiment"] = global_config["experiment"] + "/"
    if not os.path.exists("../results/" + global_config["experiment"]):
        os.makedirs("../results/" + global_config["experiment"])

    # 客户端延迟文件生成
    stale = global_config['stale']
    if isinstance(stale, list):
        client_staleness_list = stale
    elif isinstance(stale, bool):
        client_staleness_list = []
        for i in range(global_config["client_num"]):
            client_staleness_list.append(0)
    else:
        total_sum = 0
        for i in stale['list']:
            total_sum += i
        if total_sum != global_config['client_num']:
            raise ClientSumError.ClientSumError()
        client_staleness_list = generate_stale_list(stale['step'], stale['shuffle'], stale['list'])
    client_config["stale_list"] = client_staleness_list

    is_cover = True
    # 保存配置文件
    if os.path.exists("../results/" + global_config["experiment"] + "config.json"):
        is_cover = input("实验路径已存在config.json，是否覆盖(y/n):")
        if is_cover == 'y' or is_cover == 'Y':
            is_cover = True
        else:
            print("config.json将不会被存储")
            is_cover = False

    if is_cover:
        try:
            config['global']['stale'] = client_staleness_list
            with open("../results/" + global_config["experiment"] + "config.json", "w") as r:
                json.dump(config, r, indent=4)
        except shutil.SameFileError:
            pass

    start_time = datetime.datetime.now()

    accuracy_lists = []
    loss_lists = []

    sync_server = SyncServer.SyncServer(global_config, server_config, client_config, manager_config)

    sync_server.run()
    print("")

    accuracy_list, loss_list = sync_server.get_accuracy_and_loss_list()

    del sync_server

    print("Thread count =", threading.activeCount())
    print(*threading.enumerate(), sep="\n")

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    # 保存结果
    saveAns("../results/" + global_config["experiment"] + "accuracy.txt", list(accuracy_list))
    saveAns("../results/" + global_config["experiment"] + "loss.txt", list(loss_list))
    saveAns("../results/" + global_config["experiment"] + "time.txt", end_time - start_time)
    result_to_markdown("../results/" + global_config["experiment"] + "实验阐述.md", config)
