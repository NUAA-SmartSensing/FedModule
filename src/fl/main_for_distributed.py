import argparse
import datetime
import os
import sys
import warnings

import torch.multiprocessing as mp
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.Runtime import running_mode
from utils.GlobalVarGetter import GlobalVarGetter
from core.MessageQueue import MessageQueueFactory
from utils.Tools import *
from utils import ModuleFindTool


def generate_random_seed():
    seed = int(time.time() * 1000) % 2147483647
    return seed


def main():
    parser = argparse.ArgumentParser(description='FedModule Framework')
    parser.add_argument('config_file', nargs='?', default='', help='config file path')
    parser.add_argument('--config', type=str, default='', help='config file path')
    parser.add_argument('--uid', type=str, default='', help='process uid to distinguish different runs')
    parser.add_argument('--mid', type=int, default=-1, help='manager uid to distinguish different machines')
    args = parser.parse_args()

    # 创建结果文件夹
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results"))

    # 配置文件读取
    config_file = args.config_file if args.config_file else args.config
    if config_file == '':
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.json")
    config = getJson(config_file)

    config["global"]["uid"] = args.uid
    if args.mid == -1:
        raise ValueError("Please specify the manager id.")
    else:
        config["client_manager"]["manager_id"] = args.mid

    # 随机数种子
    if "seed" not in config["global"]:
        seed = generate_random_seed()
        config["global"]["seed"] = seed
    else:
        seed = config["global"]["seed"]
    # 设置随机数种子
    random_seed_set(seed)

    global_config = config['global']
    server_config = config['server']
    client_config = config['client']
    client_manager_config = config['client_manager']
    queue_manager_config = config['queue_manager']
    wandb_config = config['wandb']
    client_config["seed"] = seed

    # 实验路径相关
    if not global_config["experiment"].endswith("/"):
        global_config["experiment"] = global_config["experiment"] + "/"
    if not os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"])):
        os.makedirs(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"]))

    if "save" in global_config and not global_config["save"]:
        is_cover = False
    else:
        is_cover = True

    # 保存配置文件
    if os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                         "config.json")) and is_cover:
        is_cover = input("实验路径已存在，是否覆盖(y/n):")
        if is_cover == 'y' or is_cover == 'Y':
            is_cover = True
        else:
            print("试验结果将不会被存储")
            is_cover = False
    global_config["save"] = is_cover

    # 初始化wandb
    if wandb_config["enabled"]:
        warnings.warn("Wandb is enabled. If submanager has no data to log, please disable it.")
        params = {}
        if "params" in wandb_config:
            params = wandb_config["params"]
        wandb.init(
            project=wandb_config["project"],
            config=config,
            name=f'{wandb_config["name"]}-{config["client_manager"]["manager_id"]}',
            **params
        )

    GlobalVarGetter.set({'config': config, 'global_config': global_config,
                         'server_config': server_config,
                         'client_config': client_config,
                         'client_manager_config': client_manager_config,
                         'queue_manager_config': queue_manager_config})
    global_var = GlobalVarGetter.get()
    message_queue = MessageQueueFactory.create_message_queue(True)
    message_queue.set_config(global_var)

    start_time = datetime.datetime.now()

    # 改用文件系统存储内存
    if 'use_file_system' in global_config and global_config['use_file_system']:
        torch.multiprocessing.set_sharing_strategy('file_system')

    client_manager_config["stale_list"] = []
    global_var['client_staleness_list'] = []
    client_manager_config["index_list"] = []
    global_var['client_index_list'] = []
    global_var['test_index_list'] = []

    if "message_queue" in global_config:
        mask_list = global_config['message_queue'].get('mask_list', [])
        if "train_dataset" in mask_list or "test_dataset" in mask_list:
            dataset_class = ModuleFindTool.find_class_by_path(global_config["dataset"]["path"])
            dataset = dataset_class(global_config["client_num"], global_config["iid"],
                                    global_config["dataset"]["params"])
            train_dataset = dataset.get_train_dataset()
            test_dataset = dataset.get_test_dataset()
            send_dataset(train_dataset, test_dataset, message_queue, global_config)

    # 启动client_manager
    # get the running mode of client
    running_mode(config, output=True)
    client_manager_class = ModuleFindTool.find_class_by_path(client_manager_config["sub_path"])
    client_manager = client_manager_class(config)
    client_manager.client_join()


    print("Time used:")
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")


def cleanup():
    print()
    print("=" * 20)
    print("开始缓存清理")
    # to clean up some memory
    print("缓存清理完成")
    print("=" * 20)


if __name__ == '__main__':
    try:
        mp.set_start_method('forkserver', force=True)
        main()
    finally:
        cleanup()
