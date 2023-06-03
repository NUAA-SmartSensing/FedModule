import threading

import torch.cuda

from utils import ModuleFindTool, Time
from utils.GlobalVarGetter import GlobalVarGetter


class BaseServer:
    def __init__(self, config):
        self.config = config
        self.global_config = config['global']
        self.server_config = config['server']
        self.client_config = config['client']
        self.client_manager_config = config['client_manager']
        self.queue_manager_config = config['queue_manager']

        # 全局存储变量
        self.global_var = GlobalVarGetter().set({'server': self, 'config': config, 'global_config': self.global_config,
                                                 'server_config': self.server_config,
                                                 'client_config': self.client_config,
                                                 'client_manager_config': self.client_manager_config,
                                                 'queue_manager_config': self.queue_manager_config})
        # 全局模型
        model_class = ModuleFindTool.find_class_by_path(self.server_config["model"]["path"])
        self.server_network = model_class(**self.server_config["model"]["params"])
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.server_network = self.server_network.to(self.dev)
        self.global_var['server_network'] = self.server_network

        # 数据集
        dataset_class = ModuleFindTool.find_class_by_path(self.global_config["dataset_path"])
        self.dataset = dataset_class(self.global_config["client_num"], self.global_config["iid"])
        self.global_var['dataset'] = self.dataset
        self.config['global']['iid'] = self.dataset.get_config()

        # 计时变量
        self.T = self.server_config["epochs"]
        self.current_t = Time.Time(1)
        self.schedule_t = Time.Time(1)
        self.global_var['current_t'] = self.current_t
        self.global_var['schedule_t'] = self.schedule_t
        self.global_var['T'] = self.T

        # 运行时变量
        self.accuracy_list = []
        self.loss_list = []
        self.print_lock = threading.Lock()
        self.global_var['print_lock'] = self.print_lock
        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.server_thread_lock = threading.Lock()

        # 主类
        # initialization of the server
        # the process has an order
        self.queue_manager = None
        self.client_manager = None
        self.scheduler_thread = None
        self.updater_thread = None

    def run(self):
        print("Start server:")

        # 启动server中的两个线程
        self.scheduler_thread.start()
        self.updater_thread.start()

        client_thread_list = self.client_manager.get_client_thread_list()
        for client_thread in client_thread_list:
            client_thread.join()
        self.scheduler_thread.join()
        print("scheduler_thread joined")
        self.updater_thread.join()
        print("updater_thread joined")
        print("Thread count =", threading.active_count())
        print(*threading.enumerate(), sep="\n")

        # 队列报告
        self.queue_manager.stop()

        self.accuracy_list, self.loss_list = self.updater_thread.get_accuracy_and_loss_list()
        # 结束主类
        self.kill_main_class()
        print("End!")

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_config(self):
        return self.config

    def kill_main_class(self):
        del self.scheduler_thread
        del self.updater_thread
        del self.client_manager
        del self.queue_manager
