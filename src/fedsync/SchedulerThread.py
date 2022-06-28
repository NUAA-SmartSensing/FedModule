import copy
import threading
import time
from utils import ModuleFindTool


class SchedulerThread(threading.Thread):
    def __init__(self, server_thread_lock, sync_client_manager,
                 queue, current_t, scheduler_config, checkin_config,
                 server_network, t, mutex_sem, empty_sem, full_sem):
        threading.Thread.__init__(self)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.server_thread_lock = server_thread_lock
        self.schedule_interval = scheduler_config["scheduler_interval"]
        self.check_in_interval = checkin_config["checkin_interval"]
        self.sync_client_manager = sync_client_manager
        self.queue = queue
        self.current_t = current_t
        self.server_network = server_network
        self.T = t
        schedule_class = ModuleFindTool.find_class_by_string("schedule", scheduler_config["schedule_file"], scheduler_config["schedule_name"])
        self.schedule = schedule_class()
        receiver_config = scheduler_config["receiver"]
        receiver_class = ModuleFindTool.find_class_by_string("receiver", receiver_config["receiver_file"], receiver_config["receiver_name"])
        self.receiver = receiver_class(queue, receiver_config)
        self.config = scheduler_config

    def run(self):
        last_s_time = -1
        while self.current_t.get_time() < self.T:
            current_time = self.current_t.get_time()
            if last_s_time != current_time:
                # 每隔一段时间进行一次schedule
                self.empty_sem.acquire()
                self.mutex_sem.acquire()
                if self.current_t.get_time() == self.T:
                    break
                print("| current_epoch |", current_time)
                print("Begin client select")
                last_s_time = current_time
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
                # 等待所有客户端上传更新
                self.receiver.start(len(selected_client_threads))
                # 通知updater聚合权重
                self.mutex_sem.release()
                self.full_sem.release()
                time.sleep(0.01)

    def client_select(self, params):
        current_checked_client_tl = self.sync_client_manager.get_checked_in_client_thread_list()
        selected_client_threads = self.schedule.schedule(current_checked_client_tl, params)
        return selected_client_threads
