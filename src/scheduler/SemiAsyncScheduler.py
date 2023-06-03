import copy
import time

from scheduler import BaseScheduler


class SemiAsyncScheduler(BaseScheduler.BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.updater = self.global_var['updater']
        self.group_manager = self.global_var['group_manager']

    def run(self):
        last_s_time = -1
        group_num = -1
        while self.current_t.get_time() <= self.T:
            # 每隔一段时间进行一次schedule
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            current_time = self.current_t.get_time()
            schedule_time = self.schedule_t.get_time()
            # 每轮开始时check是否更新分组并自动更新
            self.group_manager.check_update()
            if last_s_time != current_time:
                if current_time > self.T:
                    break
                print("| current_epoch |", current_time)
                # 第一轮启动所有层
                if current_time == 1:
                    print("starting all groups")
                    last_s_time = current_time
                    for i in range(self.group_manager.get_group_num()):
                        for j in self.group_manager.get_group_list()[i]:
                            j.set_group_id(i)
                        self.print_lock.acquire()
                        print(f"\nbegin select group {i}")
                        selected_client_threads = self.client_select(i)
                        print("SchedulerThread select(", len(selected_client_threads), "clients):")
                        self.print_lock.release()
                        # 存储调度的客户端数量
                        self.group_manager.group_client_num_list.append(len(selected_client_threads))
                        # 全局存储各组模型列表
                        self.group_manager.network_list.append(self.server_weights)
                        for s_client_thread in selected_client_threads:
                            print(s_client_thread.get_client_id(), end=" | ")
                            # 将server的模型参数和时间戳发给client
                            s_client_thread.set_client_weight(self.server_weights)
                            s_client_thread.set_time_stamp(current_time)
                            s_client_thread.set_schedule_time_stamp(schedule_time)
                            # 启动一次client线程
                            s_client_thread.set_event()
                        print(
                            "\n-----------------------------------------------------------------Schedule complete")
                else:
                    self.print_lock.acquire()
                    print(f"\nbegin select group {group_num}")
                    self.print_lock.release()
                    last_s_time = current_time
                    selected_client_threads = self.client_select(group_num)
                    self.group_manager.group_client_num_list[group_num] = len(selected_client_threads)
                    self.print_lock.acquire()
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
                    self.print_lock.release()
                # 等待所有客户端上传更新
                self.queue_manager.receive(self.group_manager.group_client_num_list)
                group_num = self.queue_manager.group_ready_num
                # 通知updater聚合权重
                self.mutex_sem.release()
                self.full_sem.release()
                time.sleep(0.01)

    def client_select(self, *args, **kwargs):
        client_list = self.group_manager.get_group_list()[args[0]]
        selected_client_threads = self.schedule_caller.schedule(client_list)
        return selected_client_threads
