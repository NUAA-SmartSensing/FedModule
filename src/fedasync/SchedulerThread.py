import copy
import threading
import time
from utils import ModuleFindTool


class SchedulerThread(threading.Thread):
    def __init__(self, server_thread_lock, async_client_manager,
                 queue, current_t, schedule_t, scheduler_config,
                 server_network, t, global_var):
        threading.Thread.__init__(self)
        self.server_thread_lock = server_thread_lock
        self.schedule_interval = scheduler_config["params"]["schedule_interval"]
        self.async_client_manager = async_client_manager
        self.queue = queue
        self.current_t = current_t
        self.schedule_t = schedule_t
        self.server_network = server_network
        self.T = t
        schedule_class = ModuleFindTool.find_class_by_path(scheduler_config["scheduler_path"])
        self.schedule = schedule_class()
        self.config = scheduler_config
        self.global_var = global_var

    def run(self):
        last_s_time = -1
        # last_c_time = -1
        while self.current_t.get_time() <= self.T:
            current_time = self.current_t.get_time() - 1
            # 每隔一段时间进行一次schedule
            if current_time % self.schedule_interval == 0 and current_time != last_s_time:
                print("| current_time |", current_time % self.schedule_interval, "= 0", current_time, "!=", last_s_time)
                print("| queue.size |", self.queue.qsize(), "<= 2 *", self.schedule_interval)
                # 如果server已收到且未使用的client更新数小于schedule interval，则进行schedule
                if self.queue.qsize() <= self.schedule_interval * 2:
                    last_s_time = current_time
                    print("Begin client select")
                    selected_client_threads = self.client_select(self.config["params"])
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
                else:
                    print("\n-----------------------------------------------------------------No Schedule")
                time.sleep(0.01)
            else:
                time.sleep(0.01)

    def client_select(self, params):
        client_list = self.async_client_manager.get_client_thread_list()
        selected_client_threads = self.schedule.schedule(client_list, params)
        return selected_client_threads
