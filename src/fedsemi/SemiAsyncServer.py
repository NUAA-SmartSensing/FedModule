import threading

import torch.cuda

from fedsemi import SchedulerThread
from fedsemi import SemiAsyncClientManager
from fedsemi import UpdaterThread
from utils import ModuleFindTool, Queue, Time


class SemiAsyncServer:
    def __init__(self, config, global_config, server_config, client_config, manager_config):
        self.config = config
        # 全局存储变量
        self.global_var = {'server': self}
        # 全局模型
        model_class = ModuleFindTool.find_class_by_path(server_config["model"]["path"])
        self.server_network = model_class()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.server_network = self.server_network.to(self.dev)

        # 数据集
        dataset_class = ModuleFindTool.find_class_by_path(global_config["dataset_path"])
        self.dataset = dataset_class(global_config["client_num"], global_config["iid"])
        self.test_data = self.dataset.get_test_dataset()
        self.config['global']['iid'] = self.dataset.get_config(**client_config["model"]["params"])
        self.T = server_config["epochs"]

        # 运行时变量
        self.current_t = Time.Time(1)
        self.schedule_t = Time.Time(1)
        self.accuracy_list = []
        self.loss_list = []
        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.server_thread_lock = threading.Lock()

        init_weights = self.server_network.state_dict()
        datasets = self.dataset.get_train_dataset()

        self.mutex_sem = threading.Semaphore(1)
        self.empty_sem = threading.Semaphore(1)
        self.full_sem = threading.Semaphore(0)
        grouping_class = ModuleFindTool.find_class_by_path(server_config['grouping']['grouping_path'])
        self.group_manager = grouping_class(server_config['grouping']["params"])
        self.network_list = []
        self.semi_client_manager = SemiAsyncClientManager.SemiAsyncClientManager(init_weights,
                                                                                 global_config["client_num"],
                                                                                 global_config["multi_gpu"],
                                                                                 datasets, self.group_manager,
                                                                                 self.current_t, self.schedule_t,
                                                                                 self.stop_event, client_config,
                                                                                 manager_config, self.global_var)
        self.global_var['client_manager'] = self.semi_client_manager
        self.queue_list = [Queue.Queue() for _ in range(self.group_manager.group_num)]
        self.semi_client_manager.set_queue_list(self.queue_list)
        self.epoch_list = [0] * self.group_manager.group_num
        self.updater_thread = UpdaterThread.UpdaterThread(self.queue_list, self.server_thread_lock,
                                                          self.T, self.current_t, self.schedule_t,
                                                          self.server_network,
                                                          self.network_list, self.epoch_list,
                                                          self.group_manager, self.stop_event,
                                                          self.test_data, server_config["updater"],
                                                          self.mutex_sem, self.empty_sem, self.full_sem,
                                                          self.global_var)
        self.global_var['updater'] = self.updater_thread
        self.scheduler_thread = SchedulerThread.SchedulerThread(self.server_thread_lock, self.semi_client_manager,
                                                                self.queue_list, self.current_t, self.schedule_t,
                                                                server_config["scheduler"], self.epoch_list,
                                                                self.server_network, self.network_list, self.T,
                                                                self.group_manager, self.updater_thread,
                                                                self.mutex_sem, self.empty_sem, self.full_sem,
                                                                self.global_var)
        self.global_var['scheduler'] = self.scheduler_thread

    def run(self):
        print("Start server:")

        # 启动server中的两个线程
        self.scheduler_thread.start()
        self.updater_thread.start()

        client_thread_list = self.semi_client_manager.get_client_thread_list()
        for client_thread in client_thread_list:
            client_thread.join()
        self.scheduler_thread.join()
        print("scheduler_thread joined")
        self.updater_thread.join()
        print("updater_thread joined")

        print("Thread count =", threading.active_count())
        print(*threading.enumerate(), sep="\n")

        # if not self.queue.empty():
        #     print("\nUn-used client weights:", self.queue.qsize())
        #     for q in range(self.queue.qsize()):
        #         self.queue.get()
        # self.queue.close()

        self.accuracy_list, self.loss_list = self.updater_thread.get_accuracy_and_loss_list()
        del self.scheduler_thread
        del self.updater_thread
        del self.semi_client_manager
        print("End!")

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_config(self):
        return self.config
