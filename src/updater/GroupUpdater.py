import copy
import time

import torch

from updater.AsyncUpdater import AsyncUpdater
from utils.Structures import ComputableDict


class GroupUpdater(AsyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config):
        AsyncUpdater.__init__(self, server_thread_lock, stop_event, config)
        self.total_data_sum = None
        self.total_update_list = None
        self.num_matrix = None
        self.epoch_num = None
        self.update_list = None
        self.data_sum_list = None
        self.group_manager = self.global_var['group_manager']
        self.id_group_list = None
        self.alpha = config['alpha']
        self.initialized = False

    def run(self):
        self.id_group_list = self.global_var['scheduler'].id_group_list
        self.update_list = {i: {} for i in range(self.group_manager.get_group_num())}
        self.total_update_list = {}
        self.data_sum_list = {i: {} for i in range(self.group_manager.get_group_num())}
        self.total_data_sum = {}
        self.num_matrix = [{} for _ in range(self.group_manager.get_group_num())]
        self.epoch_num = {}
        epoch = 0
        while epoch < self.T:
            while (not self.initialized) and (self.global_var['scheduler'].schedule_queue.empty()):
                time.sleep(0.01)
            if not self.global_var['scheduler'].schedule_queue.empty():
                self.initialized = True
                self.id_group_list = self.global_var['scheduler'].id_group_list
                current_t, schedule_queue = self.global_var['scheduler'].schedule_queue.get()
                self.epoch_num[current_t] = len(schedule_queue)
                for i in range(self.group_manager.get_group_num()):
                    if current_t not in self.num_matrix[i]:
                        self.num_matrix[i][current_t] = 0
                for i in schedule_queue:
                    group_id = self.id_group_list.get_group_for_id(i)
                    self.num_matrix[group_id][current_t] = self.num_matrix[group_id][current_t] + 1
            while True:
                # 接收一个client发回的模型参数和时间戳
                if not self.queue_manager.empty():
                    # 等待上传
                    self.nums = self.num_generator.get_num()
                    self.queue_manager.receive(self.nums)
                    update_list = []
                    for i in range(self.nums):
                        update_list.append(self.queue_manager.get())
                        c_id = update_list[i]["client_id"]
                        time_stamp = update_list[i]["time_stamp"]
                        self.sum_delay += (self.current_time.get_time() - time_stamp)
                        self.print_lock.acquire()
                        print("Updater received data from client", c_id, "| staleness =", time_stamp, "-",
                              self.current_time.get_time(), "| queue size = ", self.queue_manager.size())
                        self.print_lock.release()
                    self.event.set()
                else:
                    update_list = []

                if self.event.is_set():
                    # 使用接收的client发回的模型参数和时间戳对全局模型进行更新
                    self.server_thread_lock.acquire()
                    self.update_server_weights(epoch, update_list)
                    self.server_thread_lock.release()
                    self.event.clear()
                    time.sleep(0.01)
                    break
                else:
                    time.sleep(0.01)
            epoch = self.current_time.get_time()
            time.sleep(0.01)

        print("Average delay =", (self.sum_delay / self.T))

        # 终止所有client线程
        self.client_manager.stop_all_clients()

    def update_server_weights(self, epoch, update_list):
        if len(update_list) != 1:
            raise Exception("Group Updater can only receive one update at a time")
        update = update_list[0]
        weights = ComputableDict(update["weights"])
        weights = weights.to('cuda')
        data_sum = update["data_sum"]
        time_stamp = update["time_stamp"]
        client_id = update["client_id"]
        group_id = self.id_group_list.get_group_for_id(client_id)
        if time_stamp not in self.update_list[group_id]:
            self.update_list[group_id][time_stamp] = weights
            self.data_sum_list[group_id][time_stamp] = data_sum
        else:
            # 加权存储
            self.update_list[group_id][time_stamp] = self.update_list[group_id][time_stamp] + weights * data_sum
            self.data_sum_list[group_id][time_stamp] = data_sum + self.data_sum_list[group_id][time_stamp]
        # 检索
        self.num_matrix[group_id][time_stamp] = self.num_matrix[group_id][time_stamp] - 1
        self.epoch_num[time_stamp] = self.epoch_num[time_stamp] - 1
        updated_parameters = None
        if self.num_matrix[group_id][time_stamp] == 0:
            # 组内更新 fedavg
            if time_stamp in self.total_update_list:
                self.total_update_list[time_stamp] = self.update_list[group_id][time_stamp] + self.total_update_list[
                    time_stamp]
                self.total_data_sum[time_stamp] = self.total_data_sum[time_stamp] + self.data_sum_list[group_id][
                    time_stamp]
            else:
                self.total_update_list[time_stamp] = copy.deepcopy(self.update_list[group_id][time_stamp])
                self.total_data_sum[time_stamp] = copy.deepcopy(self.data_sum_list[group_id][time_stamp])
            self.update_list[group_id][time_stamp] = self.update_list[group_id][time_stamp] / self.data_sum_list[group_id][time_stamp]

            # 更新全局 fedasync
            updated_parameters = self.update_caller.update_server_weights(epoch, [
                {"weights": self.update_list[group_id][time_stamp], "time_stamp": time_stamp}])
            # 清理空间
            self.update_list[group_id].pop(time_stamp)
            self.data_sum_list[group_id].pop(time_stamp)
        if self.epoch_num[time_stamp] == 0:
            # 更新
            print("all averaging!!!")
            self.total_update_list[time_stamp] = self.total_update_list[time_stamp] / self.total_data_sum[time_stamp]
            updated_parameters = ComputableDict(updated_parameters) * (1 - self.alpha) + self.total_update_list[time_stamp] * self.alpha
            # 清理空间
            self.total_update_list.pop(time_stamp)
            self.total_data_sum.pop(time_stamp)
        if updated_parameters is not None:
            updated_parameters = dict(updated_parameters)
            # 下发给客户端的权重
            self.global_var['scheduler'].server_weights = copy.deepcopy(updated_parameters)
            if torch.cuda.is_available():
                for key, var in updated_parameters.items():
                    updated_parameters[key] = updated_parameters[key].cuda()
            self.server_network.load_state_dict(updated_parameters)
            self.run_server_test(epoch)
            self.current_time.time_add()
