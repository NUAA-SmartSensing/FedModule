import time
from queue import Queue

from scheduler.AsyncScheduler import AsyncScheduler
from utils.Structures import BidirectionalMappedList


class GroupScheduler(AsyncScheduler):
    def __init__(self, server_thread_lock, config):
        AsyncScheduler.__init__(self, server_thread_lock, config)
        self.group_manager = self.global_var['group_manager']
        self.schedule_client_delay = config['schedule_client_delay']
        self.schedule_queue = Queue()
        self.id_group_list = BidirectionalMappedList()
        self.client_num = len(self.client_manager.get_client_id_list())

    def run(self):
        last_s_time = -1
        last_s_num = -1
        while self.current_t.get_time() <= self.T:
            current_time = self.current_t.get_time() - 1
            schedule_time = self.schedule_t.get_time()
            # 每轮开始时check是否更新分组并自动更新
            if self.group_manager.check_update():
                # 下发分组信息
                for i in range(self.group_manager.get_group_num()):
                    for j in self.group_manager.get_group_list()[i]:
                        # self.message_queue.put_into_downlink(j, "group_id", i)
                        self.id_group_list.add_data(j, i, j)

            # 每隔一段时间进行一次schedule
            if self.global_var['queue_manager'].get.count / self.schedule_interval != last_s_num and current_time != last_s_time:
                # self.print_lock.acquire()
                # print("| current_time |", current_time % self.schedule_interval, "= 0", current_time, "!=", last_s_time)
                # print("| queue.size |", self.queue_manager.size(), "<= ", self.schedule_delay)
                # self.print_lock.release()
                untraining_num = self.client_num - self.message_queue.get_training_client_num()
                # 如果server已收到且未使用的client更新数小于schedule delay，则进行schedule
                if untraining_num >= self.schedule_client_delay and self.schedule_delay >= self.queue_manager.size():
                    last_s_time = current_time
                    last_s_num = self.global_var['queue_manager'].get.count / self.schedule_interval
                    self.print_lock.acquire()
                    print("Begin client select")
                    self.print_lock.release()
                    selected_clients = self.client_select()
                    self.schedule_queue.put((current_time, selected_clients))
                    self.print_lock.acquire()
                    print("\nSchedulerThread select(", len(selected_clients), "clients):")
                    for k,v in self.server_weights.items():
                        self.server_weights[k] = v.cpu()
                    for client_id in selected_clients:
                        print(client_id, end=" | ")
                        # 将server的模型参数和时间戳发给client
                        self.send_weights(client_id, current_time, schedule_time)
                        # 启动一次client线程
                        self.client_manager.selected_event_list[client_id].set()
                    print("\n-----------------------------------------------------------------Schedule complete")
                    self.print_lock.release()
                    self.schedule_t.time_add()
                else:
                    #
                    # self.print_lock.acquire()
                    # print("\n-----------------------------------------------------------------No Schedule, untraining client is too few")
                    # self.print_lock.release()
                    pass
                time.sleep(0.01)
            else:
                time.sleep(0.01)

    def client_select(self, *args, **kwargs):
        client_list = self.client_manager.get_client_id_list()
        training_status = self.message_queue.get_training_status()
        ready_client_list = [c for c in client_list if c not in training_status or training_status[c] is False]
        selected_clients = self.schedule_caller.schedule(ready_client_list)
        return selected_clients
