import time

from scheduler.BaseScheduler import BaseScheduler


# this scheduler schedules clients according to the number of aggregations
class LeoAsyncScheduler(BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.schedule_interval = config["schedule_interval"]
        self.schedule_delay = config["schedule_delay"]

    def run(self):
        last_s_time = -1
        while self.current_t.get_time() <= self.T:
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            current_time = self.current_t.get_time()
            schedule_time = self.schedule_t.get_time()
            # 每隔一段时间进行一次schedule
            if current_time % self.schedule_interval == 1 and current_time != last_s_time and current_time <= self.T:
                print("| current_time |", current_time % self.schedule_interval, "= 1", current_time, "!=", last_s_time)
                print("| queue.size |", self.queue_manager.size(), "<= ", self.schedule_delay)
                # 如果server已收到且未使用的client更新数小于schedule delay，则进行schedule。如果过多就不继续触发client进行新一轮的训练，等待updater完成调度后再schedule。
                if self.queue_manager.size() <= self.schedule_delay:
                    last_s_time = current_time
                    # Leo背景下，就不存在主动选了，而是指定相应的客户端
                    print("Begin client select")
                    selected_clients = self.client_select()
                    print("\nSchedulerThread select(", len(selected_clients), "clients):")
                    for client_id in selected_clients:
                        print(client_id, end=" | ")
                        # 将server的模型参数和时间戳发给client
                        self.send_weights(client_id, current_time, schedule_time)
                        # 启动一次client线程
                        self.selected_event_list[client_id].set()
                    print("\n-----------------------------------------------------------------Schedule complete")
                    self.schedule_t.time_add()
                else:
                    print("\n-----------------------------------------------------------------No need to Schedule")
            self.mutex_sem.release()
            self.full_sem.release()
            time.sleep(0.01)


# this scheduler schedules clients according to the nums of update which clients update
class AsyncSchedulerWithUpdate(LeoAsyncScheduler):
    def run(self):
        last_s_time = -1
        while self.current_t.get_time() <= self.T:
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            current_time = self.current_t.get_time()
            schedule_time = self.schedule_t.get_time()
            # 每隔一段时间进行一次schedule
            if self.global_var['queue_manager'].get.count % self.schedule_interval == 0 and current_time != last_s_time:
                print("| current_time |", current_time % self.schedule_interval, "= 0", current_time, "!=", last_s_time)
                print("| queue.size |", self.queue_manager.size(), "<= ", self.schedule_delay)
                # 如果server已收到且未使用的client更新数小于schedule delay，则进行schedule
                if self.queue_manager.size() <= self.schedule_delay:
                    last_s_time = current_time
                    print("Begin client select")
                    selected_clients = self.client_select()
                    print("\nSchedulerThread select(", len(selected_clients), "clients):")
                    for client_id in selected_clients:
                        print(client_id, end=" | ")
                        # 将server的模型参数和时间戳发给client
                        self.send_weights(client_id, current_time, schedule_time)
                        # 启动一次client线程
                        self.selected_event_list[client_id].set()
                    print("\n-----------------------------------------------------------------Schedule complete")
                    self.schedule_t.time_add()
                else:
                    print("\n-----------------------------------------------------------------No Schedule")
            self.mutex_sem.release()
            self.full_sem.release()
            time.sleep(0.01)

