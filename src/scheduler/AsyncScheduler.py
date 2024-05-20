from scheduler.SyncScheduler import SyncScheduler


# this scheduler schedules clients according to the number of aggregations
class AsyncScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        super().__init__(server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.schedule_interval = config["schedule_interval"]
        self.schedule_delay = config["schedule_delay"]
        self.last_s_time = -1

    def schedule(self):
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        # Scheduling is performed periodically.
        if current_time % self.schedule_interval == 1 and current_time != self.last_s_time and current_time <= self.T:
            print("| current_time |", current_time % self.schedule_interval, "= 1", current_time, "!=", self.last_s_time)
            print("| queue.size |", self.queue_manager.size(), "<= ", self.schedule_delay)
            # scheduling according to the number of aggregations.
            if self.queue_manager.size() <= self.schedule_delay:
                self.last_s_time = current_time
                selected_client = self.client_select()
                self.notify_client(selected_client, current_time, schedule_time)
                self.schedule_t.time_add()
            else:
                print("\n-----------------------------------------------------------------No Schedule")


# this scheduler schedules clients according to the nums of update which clients update
class AsyncSchedulerWithUpdate(AsyncScheduler):
    def schedule(self):
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        # 每隔一段时间进行一次schedule
        if self.global_var['queue_manager'].get.count % self.schedule_interval == 0 and current_time != self.last_s_time:
            print("| current_time |", current_time % self.schedule_interval, "= 0", current_time, "!=", self.last_s_time)
            print("| queue.size |", self.queue_manager.size(), "<= ", self.schedule_delay)
            # scheduling according to the number of received updates
            if self.queue_manager.size() <= self.schedule_delay:
                self.last_s_time = current_time
                selected_client = self.client_select()
                self.notify_client(selected_client, current_time, schedule_time)
                self.schedule_t.time_add()
            else:
                print("\n-----------------------------------------------------------------No Schedule")

