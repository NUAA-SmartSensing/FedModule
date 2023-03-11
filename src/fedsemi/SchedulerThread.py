import copy
import threading
import time
from utils import ModuleFindTool


class SchedulerThread(threading.Thread):
    def __init__(self, server_thread_lock, sync_client_manager,
                 queue_list, current_t, scheduler_config, epoch_list,
                 server_network, network_list, t, group_manager, updater_thread, mutex_sem, empty_sem, full_sem):
        threading.Thread.__init__(self)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.server_thread_lock = server_thread_lock
        self.sync_client_manager = sync_client_manager
        self.updater_thread = updater_thread
        self.queue_list = queue_list
        self.group_manager = group_manager
        # 全局迭代次数
        self.current_t = current_t
        # 各组迭代次数
        self.epoch_list = epoch_list
        self.server_network = server_network
        self.network_list = network_list
        self.num_list = []
        self.T = t
        schedule_class = ModuleFindTool.find_class_by_path(scheduler_config["scheduler_path"])
        self.schedule = schedule_class()
        receiver_config = scheduler_config["receiver"]
        receiver_class = ModuleFindTool.find_class_by_path(receiver_config["receiver_file"])
        self.receiver = receiver_class(queue_list, receiver_config)
        self.config = scheduler_config

    def run(self):
        last_s_time = -1
        group_num = -1
        while self.current_t.get_time() <= self.T:
            # 每隔一段时间进行一次schedule
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            current_time = self.current_t.get_time()
            if last_s_time != current_time:
                if current_time > self.T:
                    break
                print("| current_epoch |", current_time)
                # 第一轮启动所有层
                if current_time == 1:
                    print("starting all groups")
                    last_s_time = current_time
                    self.server_thread_lock.acquire()
                    server_weights = copy.deepcopy(self.server_network.state_dict())
                    self.server_thread_lock.release()
                    for i in range(self.group_manager.get_group_num()):
                        for j in self.group_manager.get_group_list()[i]:
                            j.set_group_id(i)
                        print(f"begin select group {i}")
                        selected_client_threads = self.client_select(i, self.config["params"])
                        print("\nSchedulerThread select(", len(selected_client_threads), "clients):")
                        # 存储调度的客户端数量
                        self.num_list.append(len(selected_client_threads))
                        # 全局存储各组模型列表
                        self.network_list.append(server_weights)
                        for s_client_thread in selected_client_threads:
                            print(s_client_thread.get_client_id(), end=" | ")
                            # 将server的模型参数和时间戳发给client
                            s_client_thread.set_client_weight(server_weights)
                            s_client_thread.set_time_stamp(current_time)
                            # 启动一次client线程
                            s_client_thread.set_event()
                        print(
                            "\n-----------------------------------------------------------------Schedule complete")
                    del server_weights
                else:
                    print(f"begin select group {group_num}")
                    last_s_time = current_time
                    selected_client_threads = self.client_select(group_num, self.config["params"])
                    self.num_list[group_num] = len(selected_client_threads)
                    print("\nSchedulerThread select(", len(selected_client_threads), "clients):")
                    self.server_thread_lock.acquire()
                    server_weights = copy.deepcopy(self.server_network.state_dict())
                    self.server_thread_lock.release()
                    for s_client_thread in selected_client_threads:
                        print(s_client_thread.get_client_id(), end=" | ")
                        # 将server的模型参数和时间戳发给client
                        s_client_thread.set_client_weight(server_weights)
                        s_client_thread.set_time_stamp(current_time)

                        # 启动一次client线程
                        s_client_thread.set_event()
                    del server_weights
                    print("\n-----------------------------------------------------------------Schedule complete")
                # 等待所有客户端上传更新
                group_num = self.receiver.receive(self.num_list)
                self.updater_thread.set_update_group(group_num)
                # 通知updater聚合权重
                self.mutex_sem.release()
                self.full_sem.release()
                time.sleep(0.01)

    def client_select(self, i, params):
        client_list = self.group_manager.get_group_list()[i]
        selected_client_threads = self.schedule.schedule(client_list, params)
        return selected_client_threads
