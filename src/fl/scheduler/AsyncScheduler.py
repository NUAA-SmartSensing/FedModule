import time

from fl.scheduler import BaseScheduler


class AsyncScheduler(BaseScheduler.BaseScheduler):
    def __init__(self, server_thread_lock, client_manager,
                 queue, current_t, schedule_t, scheduler_config,
                 server_network, t, global_var):
        BaseScheduler.BaseScheduler.__init__(server_thread_lock, client_manager,
                                             current_t, schedule_t, scheduler_config,
                                             server_network, t, global_var)
        self.queue = queue
        self.schedule_interval = scheduler_config["params"]["schedule_interval"]

    def run(self):
        last_s_time = -1
        while self.current_t.get_time() <= self.T:
            current_time = self.current_t.get_time() - 1
            schedule_time = self.schedule_t.get_time()
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
                    for s_client_thread in selected_client_threads:
                        print(s_client_thread.get_client_id(), end=" | ")
                        # 将server的模型参数和时间戳发给client
                        s_client_thread.set_client_weight(self.server_weights)
                        s_client_thread.set_time_stamp(current_time)
                        s_client_thread.set_schedule_time_stamp(schedule_time)
                        # 启动一次client线程
                        s_client_thread.set_event()
                    print("\n-----------------------------------------------------------------Schedule complete")
                    self.schedule_t.time_add()
                else:
                    print("\n-----------------------------------------------------------------No Schedule")
                time.sleep(0.01)
            else:
                time.sleep(0.01)
