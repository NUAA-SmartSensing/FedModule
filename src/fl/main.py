import datetime
import os
import shutil
import sys
import threading

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.Tools import *
from utils import ModuleFindTool
from utils.ConfigManager import *
from exception import ClientSumError

if __name__ == '__main__':
    # 创建结果文件夹
    if not os.path.exists("../results"):
        os.mkdir("../results")

    # 配置文件读取
    if len(sys.argv) < 2:
        config_file = "../../config.json"
    else:
        config_file = sys.argv[1]

    config = getConfig(config_file)
    global_config = config['global']
    server_config = config['server']
    client_config = config['client']
    manager_config = config['client_manager']
    wandb_config = config['wandb']

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

    # 初始化wandb
    if wandb_config["enabled"]:
        wandb.init(
            project=wandb_config["project"],
            config=config,
            name=wandb_config["name"],
        )
    if wandb_config["enabled"]:
        try:
            global_config['stale'] = client_staleness_list
            with open(os.path.join(wandb.run.dir, "config.json"), "w") as r:
                json.dump(config, r, indent=4)
        except shutil.SameFileError:
            pass
    start_time = datetime.datetime.now()

    # 改用文件系统存储内存
    if global_config['use_file_system']:
        torch.multiprocessing.set_sharing_strategy('file_system')
    accuracy_lists = []
    loss_lists = []
    # wandb启动配置植入update_config中
    server_config['updater']['enabled'] = wandb_config['enabled']
    server_class = ModuleFindTool.find_class_by_path(server_config["path"])
    server = server_class(config)
    server.run()

    accuracy_list, loss_list = server.get_accuracy_and_loss_list()
    config = server.get_config()
    del server

    print("Thread count =", threading.active_count())
    print(*threading.enumerate(), sep="\n")

    print("Time used:")
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    # 保存配置文件
    if is_cover:
        try:
            global_config['stale'] = client_staleness_list
            with open("../results/" + global_config["experiment"] + "config.json", "w") as r:
                json.dump(config, r, indent=4)
        except shutil.SameFileError:
            pass

    # 保存结果
    saveAns("../results/" + global_config["experiment"] + "accuracy.txt", list(accuracy_list))
    saveAns("../results/" + global_config["experiment"] + "loss.txt", list(loss_list))
    saveAns("../results/" + global_config["experiment"] + "time.txt", end_time - start_time)
    result_to_markdown("../results/" + global_config["experiment"] + "实验阐述.md", config)
    if wandb_config['enabled']:
        saveAns(os.path.join(wandb.run.dir, "accuracy.txt"), list(accuracy_list))
        saveAns(os.path.join(wandb.run.dir, "loss.txt"), list(loss_list))
        saveAns(os.path.join(wandb.run.dir, "time.txt"), end_time - start_time)
        result_to_markdown(os.path.join(wandb.run.dir, "实验阐述.md"), config)
