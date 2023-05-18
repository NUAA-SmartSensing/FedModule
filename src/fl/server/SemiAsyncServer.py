import threading

import torch.cuda

from fl.server import BaseServer
from fedsemi import SchedulerThread
from fedsemi import SemiAsyncClientManager
from fedsemi import UpdaterThread
from utils import ModuleFindTool, Queue, Time


class SemiAsyncServer(BaseServer.BaseServer):
    def __init__(self, config, global_config, server_config, client_config, manager_config):
        BaseServer.BaseServer.__init__(config, global_config, server_config)
        init_weights = self.server_network.state_dict()
        datasets = self.dataset.get_train_dataset()

        self.mutex_sem = threading.Semaphore(1)
        self.empty_sem = threading.Semaphore(1)
        self.full_sem = threading.Semaphore(0)
        grouping_class = ModuleFindTool.find_class_by_path(server_config['grouping']['grouping_path'])
        self.group_manager = grouping_class(server_config['grouping']["params"])
        self.network_list = []
        self.client_manager = SemiAsyncClientManager.SemiAsyncClientManager(init_weights,
                                                                            global_config["client_num"],
                                                                            global_config["multi_gpu"],
                                                                            datasets, self.group_manager,
                                                                            self.current_t, self.schedule_t,
                                                                            self.stop_event, client_config,
                                                                            manager_config, self.global_var)
        self.global_var['client_manager'] = self.client_manager
        self.queue_list = [Queue.Queue() for _ in range(self.group_manager.group_num)]
        self.client_manager.set_queue_list(self.queue_list)
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
        self.scheduler_thread = SchedulerThread.SchedulerThread(self.server_thread_lock, self.client_manager,
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

        client_thread_list = self.client_manager.get_client_thread_list()
        for client_thread in client_thread_list:
            client_thread.join()
        self.scheduler_thread.join()
        print("scheduler_thread joined")
        self.updater_thread.join()
        print("updater_thread joined")
        print("Thread count =", threading.active_count())
        print(*threading.enumerate(), sep="\n")

        self.accuracy_list, self.loss_list = self.updater_thread.get_accuracy_and_loss_list()
        del self.scheduler_thread
        del self.updater_thread
        del self.client_manager
        print("End!")

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_config(self):
        return self.config
