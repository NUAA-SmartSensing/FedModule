import copy
import datetime
import os
import sys
import uuid

import torch.multiprocessing as mp
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.Runtime import running_mode
from utils.GlobalVarGetter import GlobalVarGetter
from core.MessageQueue import MessageQueueFactory, ManagerWrapper
from utils.Tools import *
from utils import ModuleFindTool
import argparse


def generate_client_stale_list(global_config):
    stale = global_config['stale']
    if isinstance(stale, list):
        client_staleness_list = stale
    elif isinstance(stale, bool):
        client_staleness_list = []
        for i in range(global_config["client_num"]):
            client_staleness_list.append(0)
    elif isinstance(stale, dict) and "path" in stale:
        stale_generator = ModuleFindTool.find_class_by_path(stale["path"])()(stale["params"])
        client_staleness_list = stale_generator.generate_staleness_list()
    else:
        total_sum = sum(stale['list'])
        if total_sum < global_config['client_num']:
            raise Exception("The sum of the client number in stale list must not less than the client number.")
        client_staleness_list = generate_stale_list(stale['step'], stale['shuffle'], stale['list'])
    return client_staleness_list


def get_client_data_distri(index_lists, targets):
    client_num = len(index_lists)
    label_counts = [{i: 0 for i in range(10)} for _ in range(client_num)]
    for i, index_list in enumerate(index_lists):
        for index in index_list:
            label_counts[i][int(targets[index])] += 1
        print(f'({i}, {len(index_list)}, {label_counts[i]})')
    return label_counts


def main():
    parser = argparse.ArgumentParser(description='FedModule Framework')
    parser.add_argument('config_file', nargs='?', default='', help='config file path')
    parser.add_argument('--config', type=str, default='', help='config file path')
    parser.add_argument('--uid', type=str, default='', help='process uid to distinguish different runs')
    args = parser.parse_args()

    # 创建结果文件夹
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results"))

    # 配置文件读取
    config_file = args.config_file if args.config_file else args.config
    if config_file == '':
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.json")
    config = getJson(config_file)

    # 生成uuid
    uid = args.uid
    if uid == '':
        uuid_v4 = uuid.uuid4()
        uid = uuid_v4.hex
    config["global"]["uid"] = uid
    print("global uid:", uid)

    # 随机数种子
    if "seed" not in config["global"]:
        seed = generate_random_seed()
        config["global"]["seed"] = seed
    else:
        seed = config["global"]["seed"]
    # 设置随机数种子
    random_seed_set(seed)

    raw_config = copy.deepcopy(config)
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
        params = {}
        if "params" in wandb_config:
            params = wandb_config["params"]
        wandb.init(
            project=wandb_config["project"],
            config=config,
            name=wandb_config["name"],
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

    # 客户端延迟文件生成
    client_staleness_list = generate_client_stale_list(global_config)
    client_manager_config["stale_list"] = client_staleness_list
    global_var['client_staleness_list'] = client_staleness_list

    # 生成dataset
    dataset_class = ModuleFindTool.find_class_by_path(global_config["dataset"]["path"])
    dataset = dataset_class(global_config["client_num"], global_config["iid"], global_config["dataset"]["params"])
    train_dataset = dataset.get_train_dataset()
    test_dataset = dataset.get_test_dataset()
    train_dataset, test_dataset = send_dataset(train_dataset, test_dataset, message_queue, global_config)
    index_list = dataset.get_index_list()
    test_index_list = dataset.get_test_index_list()
    client_manager_config["index_list"] = index_list
    global_var['client_index_list'] = index_list
    global_var['test_index_list'] = test_index_list

    # 启动client_manager
    # get the running mode of client
    running_mode(config, output=True)
    client_manager_class = ModuleFindTool.find_class_by_path(client_manager_config["path"])
    client_manager = client_manager_class(config)
    client_manager.start_all_clients()

    # 获取数据集分布
    try:
        label_counts = get_client_data_distri(index_list, train_dataset.targets)
    except:
        print("can't get the data distribution")
    # wandb启动配置植入update_config中
    server_config['updater']['enabled'] = wandb_config['enabled']
    server_class = ModuleFindTool.find_class_by_path(server_config["path"])
    server = server_class(config)
    server.run()

    accuracy_list, loss_list = server.get_accuracy_and_loss_list()
    config = server.get_config()

    # 终止所有client线程
    client_manager.stop_all_clients()
    client_manager.client_join()

    del server

    print("Time used:")
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    # 保存配置文件
    if is_cover:
        raw_config['global']['stale'] = client_staleness_list
        saveJson(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                              "config.json"), raw_config)

        # 保存结果
        saveAns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                             "accuracy.txt"), list(accuracy_list))
        saveAns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                             "loss.txt"), list(loss_list))
        saveAns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                             "time.txt"), end_time - start_time)
        saveJson(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                              "data_distribution.json"), label_counts)
        result_to_markdown(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                         "实验阐述.md"), config)
    if wandb_config['enabled']:
        saveAns(os.path.join(wandb.run.dir, "accuracy.txt"), list(accuracy_list))
        saveAns(os.path.join(wandb.run.dir, "loss.txt"), list(loss_list))
        saveAns(os.path.join(wandb.run.dir, "time.txt"), end_time - start_time)
        saveJson(os.path.join(wandb.run.dir, "data_distribution.json"), label_counts)
        saveJson(os.path.join(wandb.run.dir, "config.json"), raw_config)
        result_to_markdown(os.path.join(wandb.run.dir, "实验阐述.md"), config)


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
        MessageQueueFactory.del_message_queue()
