import kmeans1d
from scheduler.SyncScheduler import SyncScheduler


class VDAScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        super().__init__(server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.time_window = None
        self.normal_client_num = 0
        self.clusters = []
        self.fixed_time_window = config["time_window"] if "time_window" in config else None

    def schedule(self):
        r"""
            schedule the clients
        """
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        if current_time > self.T:
            return
        selected_client = self.client_select()
        self.queue_manager.set_version(current_time, len(selected_client), self.server_weights)
        self.queue_manager.set_time_window(self.time_window)
        self.notify_client(selected_client, current_time, schedule_time)
        # Waiting for all clients to upload their updates.
        self.queue_manager.receive(self.normal_client_num)

    def client_select(self, *args, **kwargs):
        selected_clients = super().client_select(*args, **kwargs)
        stale_list = [self.global_var["client_staleness_list"][stale] for stale in selected_clients]
        if self.fixed_time_window:
            self.normal_client_num = sum(1 for i in stale_list if i <= self.fixed_time_window)
            self.time_window = self.fixed_time_window
        else:
            self.clusters, centroids = kmeans1d.cluster(stale_list, 2)
            if centroids[0] < centroids[1] and self.clusters.count(0) != 0:
                self.normal_client_num = self.clusters.count(0)
                self.time_window = max(stale for i, stale in enumerate(stale_list) if self.clusters[i] == 0)
            else:
                self.normal_client_num = self.clusters.count(1)
                self.time_window = max(stale for i, stale in enumerate(stale_list) if self.clusters[i] == 1)
        return selected_clients
