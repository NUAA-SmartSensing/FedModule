import time

import torch.utils.data

from updater.BaseUpdater import BaseUpdater
from utils import ModuleFindTool


class SemiAsyncUpdater(BaseUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        BaseUpdater.__init__(self, server_thread_lock, stop_event, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.server_thread_lock = server_thread_lock
        self.group_manager = self.global_var["group_manager"]

        self.accuracy_list = []
        self.loss_list = []
        group_update_class = ModuleFindTool.find_class_by_path(config["group"]["path"])
        self.group_update = group_update_class(self.config["group"]["params"])


    def run(self):
        for i in range(self.T):
            self.full_sem.acquire()
            self.mutex_sem.acquire()
            epoch = self.current_time.get_time()
            update_list = []
            # 接收所有的更新
            while not self.queue_manager.empty(self.queue_manager.group_ready_num):
                update_list.append(self.queue_manager.get(self.queue_manager.group_ready_num))
            self.group_manager.network_list[self.queue_manager.group_ready_num] = self.update_group_weights(epoch, update_list)
            self.group_manager.epoch_list[self.queue_manager.group_ready_num] = self.group_manager.epoch_list[self.queue_manager.group_ready_num] + 1

            self.server_thread_lock.acquire()
            self.update_server_weights(epoch, self.group_manager.network_list)
            self.run_server_test(epoch)
            self.server_thread_lock.release()
            time.sleep(0.01)

            # 本轮结束
            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()

        # 终止所有client线程
        self.client_manager.stop_all_clients()

    def update_group_weights(self, epoch, update_list):
        updated_parameters = self.group_update.update_server_weights(epoch, update_list)
        for key, var in updated_parameters.items():
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        return updated_parameters

    def update_server_weights(self, epoch, network_list):
        update_list = []
        for i in range(self.global_var['group_manager'].group_num):
            update_list.append({"weights": network_list[i]})
        updated_parameters = self.update_caller.update_server_weights(epoch, update_list)
        for key, var in updated_parameters.items():
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        self.server_network.load_state_dict(updated_parameters)
