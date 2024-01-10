import copy
import datetime
import os
import shutil
import sys
import threading
import time

import torch
import torch.multiprocessing as mp
import wandb
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.DataReader import CustomDataset
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import MessageQueueFactory
from utils.Tools import *
from utils import ModuleFindTool
from utils.ConfigManager import *
from exception import ClientSumError

def generate_random_seed():
    seed = int(time.time()*1000) % 2147483647
    return seed

def _read_data(dataset):
    data = []
    targets = []
    dl = DataLoader(dataset, batch_size=1)
    for x, y in dl:
        data.append(x[0])
        targets.append(y[0])
    data = torch.stack(data)
    targets = torch.stack(targets)
    data.share_memory_()
    targets.share_memory_()
    return data, targets


def send_dataset(train_dataset, test_dataset, message_queue, global_config):
    # 预加载
    if 'dataset_pre_load' in global_config and global_config['dataset_pre_load']:
        data, targets = _read_data(train_dataset)
        message_queue.set_train_dataset(CustomDataset(data, targets))
        data, targets = _read_data(test_dataset)
        message_queue.set_test_dataset(CustomDataset(data, targets))
    # 静态加载
    else:
        message_queue.set_train_dataset(train_dataset)
        message_queue.set_test_dataset(test_dataset)


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
        total_sum = 0
        for i in stale['list']:
            total_sum += i
        if total_sum != global_config['client_num']:
            raise ClientSumError.ClientSumError()
        client_staleness_list = generate_stale_list(stale['step'], stale['shuffle'], stale['list'])
    return client_staleness_list


def main():
    # 创建结果文件夹
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results"))

    # 配置文件读取
    if len(sys.argv) < 2:
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.json")
    else:
        config_file = sys.argv[1]

    config = getConfig(config_file)

    # 随机数种子
    if "seed" not in config["global"]:
        seed = generate_random_seed()
        config["global"]["seed"] = seed
    else:
        seed = config["global"]["seed"]
    # 设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    raw_config = copy.deepcopy(config)
    global_config = config['global']
    server_config = config['server']
    client_config = config['client']
    client_manager_config = config['client_manager']
    queue_manager_config = config['queue_manager']
    wandb_config = config['wandb']
    GlobalVarGetter.set({'config': config, 'global_config': global_config,
                         'server_config': server_config,
                         'client_config': client_config,
                         'client_manager_config': client_manager_config,
                         'queue_manager_config': queue_manager_config})
    global_var = GlobalVarGetter.get()
    message_queue = MessageQueueFactory.create_message_queue()
    message_queue.set_config(global_var)

    # 实验路径相关
    if not global_config["experiment"].endswith("/"):
        global_config["experiment"] = global_config["experiment"] + "/"
    if not os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"])):
        os.makedirs(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"]))

    is_cover = True
    # 保存配置文件
    if os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                         "config.json")):
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
    start_time = datetime.datetime.now()

    # 改用文件系统存储内存
    if global_config['use_file_system']:
        torch.multiprocessing.set_sharing_strategy('file_system')

    # 客户端延迟文件生成
    client_staleness_list = generate_client_stale_list(global_config)
    client_config["stale_list"] = client_staleness_list
    global_var['client_staleness_list'] = client_staleness_list

    # 生成dataset
    dataset_class = ModuleFindTool.find_class_by_path(global_config["dataset"]["path"])
    dataset = dataset_class(global_config["client_num"], global_config["iid"], global_config["dataset"]["params"])
    train_dataset = dataset.get_train_dataset()
    test_dataset = dataset.get_test_dataset()
    send_dataset(train_dataset, test_dataset, message_queue, global_config)
    index_list = dataset.get_index_list()
    client_config["index_list"] = index_list
    global_var['client_index_list'] = index_list

    # 启动client_manager
    client_manager_class = ModuleFindTool.find_class_by_path(client_manager_config["path"])
    client_manager = client_manager_class(config)
    client_manager.start_all_clients()

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
        try:
            raw_config['global']['stale'] = client_staleness_list
            with open(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", global_config["experiment"],
                                 "config.json"), "w") as r:
                json.dump(raw_config, r, indent=4)
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
            try:
                raw_config['global']['stale'] = client_staleness_list
                with open(os.path.join(wandb.run.dir, "config.json"), "w") as r:
                    json.dump(raw_config, r, indent=4)
            except shutil.SameFileError:
                pass


def cleanup():
    print()
    print("=" * 20)
    print("开始缓存清理")
    # to clean up some memory
    print("缓存清理完成")
    print("=" * 20)


if __name__ == '__main__':
    try:
        mp.set_start_method('forkserver')
        main()
    finally:
        cleanup()
