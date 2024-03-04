import copy
import time

from scheduler import BaseScheduler
import pandas as pd

class LeoAsyncScheduler(BaseScheduler.BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.updater = self.global_var['updater']
        self.group_manager = self.global_var['group_manager']
        self.global_var['access_group_info'] = pd.read_csv(self.config["scheduler_access"])

    def run(self):
        last_s_time = -1
        group_id = -1
        # sat和基站通信通信时间进行手动调度
        while self.current_t.get_time() <= self.T:
            # 每隔一段时间进行一次schedule
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            current_time = self.current_t.get_time()
            schedule_time = self.schedule_t.get_time()
            # 每轮开始时check是否更新分组并自动更新(指分组是否发生变动stale_list)
            self.group_manager.check_update()
            if last_s_time != current_time:
                if current_time > self.T:
                    break
                print("| current_epoch |", current_time)
                # 第一轮启动所有层，保证组里有东西
                if current_time == 1:
                    print("starting all groups")
                    last_s_time = current_time
                    for i in range(self.group_manager.get_group_num()):
                        for j in self.group_manager.get_group_list()[i]: # i,j 分别为组号和组下的client_id
                            self.message_queue.put_into_downlink(j, "group_id", i)
                        print(f"\nbegin select group {i}")
                        selected_clients = self.client_select(i)
                        print("SchedulerThread select(", len(selected_clients), "clients):")
                        # 存储调度的客户端数量
                        self.group_manager.group_client_num_list.append(len(selected_clients))
                        # 全局存储各组模型列表
                        self.group_manager.network_list.append(self.server_weights)
                        for client_id in selected_clients:
                            print(client_id, end=" | ")
                            # 将server的模型参数和时间戳发给client
                            self.send_weights(client_id, current_time, schedule_time)
                            # 启动一次client线程
                            self.selected_event_list[client_id].set()
                        print(
                            "\n-----------------------------------------------------------------Schedule complete")
                else:
                    print(f"\nbegin select group {group_id}")
                    last_s_time = current_time
                    selected_clients = self.client_select(group_id)
                    self.group_manager.group_client_num_list[group_id] = len(selected_clients)
                    print("\nSchedulerThread select(", len(selected_clients), "clients):")
                    self.server_thread_lock.acquire()
                    server_weights = copy.deepcopy(self.server_network.state_dict()) # sever选中某个组后，再发往对应组的clients
                    self.server_thread_lock.release()
                    for client_id in selected_clients:
                        print(client_id, end=" | ")
                        # 将server的模型参数和时间戳发给client
                        self.send_weights(client_id, current_time, schedule_time)

                        # 启动一次client线程
                        self.selected_event_list[client_id].set()
                    del server_weights
                    print("\n-----------------------------------------------------------------Schedule complete")
                # 等待所有客户端上传更新
                # 选下一个轮epoch的group_id 
                    # 这个receive决定了选择哪个group开启下一轮
                self.queue_manager.receive(self.group_manager.group_client_num_list) # 嵌套列表，告知每个组选了多少个，updater好方便聚合
                # group_id = self.global_var['access_group_info']['Orbit'].iloc[self.current_t.get_time()]
                group_id = self.queue_manager.group_id_next
                # 通知updater聚合权重
                self.mutex_sem.release()
                self.full_sem.release()
                time.sleep(0.01)

    def client_select(self, *args, **kwargs):
        client_list = self.group_manager.get_group_list()[args[0]]
        selected_clients = self.schedule_caller.schedule(client_list)
        return selected_clients
