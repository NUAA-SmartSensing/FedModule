import datetime
from doctest import debug
import os
import shutil
import sys
import threading

import SharedArray
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import MessageQueueFactory
from utils.Tools import *
from utils import ModuleFindTool
from utils.ConfigManager import *
from exception import ClientSumError


def main():
    # 创建结果文件夹
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results"))

    debug_mode  = False
    # 配置文件读取
    if len(sys.argv) < 2:
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/FedAT-config.json")
        debug_mode = True
    else:
        config_file = sys.argv[1]

    config = getConfig(config_file)
    global_config = config['global']
    server_config = config['server']
    client_config = config['client']
    client_manager_config = config['client_manager']
    queue_manager_config = config['queue_manager']
    wandb_config = config['wandb']
    GlobalVarGetter().set({'config': config, 'global_config': global_config,
                           'server_config': server_config,
                           'client_config': client_config,
                           'client_manager_config': client_manager_config,
                           'queue_manager_config': queue_manager_config})
    MessageQueueFactory.create_message_queue().set_config(GlobalVarGetter().get())

    # 实验路径相关
    if not global_config["experiment"].endswith("/"):
        global_config["experiment"] = global_config["experiment"] + "/"
    if not os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"])):
        os.makedirs(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"]))

    # 客户端延迟文件生成
    stale = global_config['stale']
    if isinstance(stale, list):
        client_staleness_list = stale
    elif isinstance(stale, bool): #stale为0
        client_staleness_list = []
        for i in range(global_config["client_num"]):
            client_staleness_list.append(0)
    else:
        total_sum = 0
        for i in stale['list']:
            total_sum += i
        if total_sum != global_config['client_num']:
            raise ClientSumError.ClientSumError()
        client_staleness_list = generate_stale_list(stale['step'], stale['shuffle'], stale['list']) # 以step为固定间隔，在之间分别随机抽取list[1:]为stale值
    client_config["stale_list"] = client_staleness_list
    # print("stale_list",client_staleness_list)

    is_cover = True
    # 保存配置文件
    if debug_mode:
        is_cover = False
    else:
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"], "config.json")) :
            is_cover = input("实验路径已存在，是否覆盖(y/n):")
            if is_cover == 'y' or is_cover == 'Y':
                is_cover = True
            else:
                print("试验结果将不会被存储")
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
    
    print("stale_list:",client_staleness_list)

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
            with open(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                                 "config.json"), "w") as r:
                json.dump(config, r, indent=4)
        except shutil.SameFileError:
            pass

        # 保存结果
        saveAns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                             "accuracy.txt"), list(accuracy_list))
        saveAns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                             "loss.txt"), list(loss_list))
        saveAns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                             "time.txt"), end_time - start_time)
        result_to_markdown(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                         "实验阐述.md"), config)
        if wandb_config['enabled']:
            saveAns(os.path.join(wandb.run.dir, "accuracy.txt"), list(accuracy_list))
            saveAns(os.path.join(wandb.run.dir, "loss.txt"), list(loss_list))
            saveAns(os.path.join(wandb.run.dir, "time.txt"), end_time - start_time)
            result_to_markdown(os.path.join(wandb.run.dir, "实验阐述.md"), config)
            wandb.finish()


def cleanup():
    print()
    print("="*20)
    print("开始缓存清理")
    global_var = GlobalVarGetter().get()
    if 'shared_mem' in global_var:
        for i in global_var['shared_mem']:
            try:
                print(f"开始清理{i}")
                SharedArray.delete(i)
            finally:
                pass
    print("缓存清理完成")
    print("="*20)


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup()
