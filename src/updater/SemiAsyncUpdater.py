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
        self.group_update = group_update_class(self.config["group"]["params"]) # group内的update方式，包括fedavg等

    def run(self):
        for i in range(self.T): # epoch
            self.full_sem.acquire()
            self.mutex_sem.acquire()
            epoch = self.current_time.get_time()
            update_list = [] # 获取从queue_manager下对应group下的clients的updaye
            # 接收所有的更新
            while not self.queue_manager.empty(self.queue_manager.group_ready_num): # 即组下都更新完,group_ready_num指本轮训练的group_id非数量
                update_list.append(self.queue_manager.get(self.queue_manager.group_ready_num)) #待更新的group下的update
            self.group_manager.network_list[self.queue_manager.group_ready_num] = self.update_group_weights(epoch,
                                                                                                            update_list) # 组内使用FedAvg进行聚合，得到组内聚合模型
            # 组内epoch+1,即每个group参与训练次数加一
            self.group_manager.epoch_list[self.queue_manager.group_ready_num] = self.group_manager.epoch_list[
                                                                                    self.queue_manager.group_ready_num] + 1

            self.server_thread_lock.acquire()
            self.update_server_weights(epoch, self.group_manager.network_list) # 传的是聚合后的模型 network_list[],包含每个组的加过
            print('Updated Model to Server: Group ', self.queue_manager.group_ready_num)
            self.run_server_test(epoch)
            self.server_thread_lock.release()
            time.sleep(0.01)

            # 本轮结束
            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()

    def update_group_weights(self, epoch, update_list):
        global_model, _ = self.group_update.update_server_weights(epoch, update_list)
        if torch.cuda.is_available():
            for key, var in global_model.items():
                global_model[key] = global_model[key].cuda()
        return global_model

    def update_server_weights(self, epoch, network_list):
        update_list = []
        for i in range(self.global_var['group_manager'].group_num):
            update_list.append({"weights": network_list[i]})
        BaseUpdater.update_server_weights(self, epoch, update_list)
