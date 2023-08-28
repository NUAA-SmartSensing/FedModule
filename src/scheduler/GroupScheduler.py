import time
from queue import Queue

from scheduler.AsyncScheduler import AsyncScheduler
from utils.Structures import BidirectionalMappedList


class GroupScheduler(AsyncScheduler):
    def __init__(self, server_thread_lock, config):
        AsyncScheduler.__init__(self, server_thread_lock, config)
        self.group_manager = config["group_manager"]
        self.schedule_queue = Queue()
        self.id_group_list = BidirectionalMappedList()

    def run(self):
        last_s_time = -1
        while self.current_t.get_time() <= self.T:
            current_time = self.current_t.get_time() - 1
            schedule_time = self.schedule_t.get_time()
            # 每轮开始时check是否更新分组并自动更新
            self.group_manager.check_update()
            # 下发分组信息
            for i in range(self.group_manager.get_group_num()):
                for j in self.group_manager.get_group_list()[i]:
                    self.message_queue.put_into_downlink(j, "group_id", i)
                    self.id_group_list.add_data(j, i, j)
            # 每隔一段时间进行一次schedule
            if self.global_var['queue_manager'].get.count % self.schedule_interval == 0 and current_time != last_s_time:
                self.print_lock.acquire()
                print("| current_time |", current_time % self.schedule_interval, "= 0", current_time, "!=", last_s_time)
                print("| queue.size |", self.queue_manager.size(), "<= ", self.schedule_delay)
                self.print_lock.release()
                # 如果server已收到且未使用的client更新数小于schedule delay，则进行schedule
                if self.queue_manager.size() <= self.schedule_delay:
                    last_s_time = current_time
                    self.print_lock.acquire()
                    print("Begin client select")
                    self.print_lock.release()
                    selected_clients = self.client_select()
                    self.print_lock.acquire()
                    print("\nSchedulerThread select(", len(selected_clients), "clients):")
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
                    self.print_lock.acquire()
                    print("\n-----------------------------------------------------------------No Schedule")
                    self.print_lock.release()
                time.sleep(0.01)
            else:
                time.sleep(0.01)
