import kmeans1d

from core.handlers.Handler import Handler
from core.handlers.ServerHandler import ClientSelector
from scheduler.SyncScheduler import SyncScheduler


class VDAScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        super().__init__(server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.time_window = None
        self.normal_client_num = 0
        self.clusters = []
        self.fixed_time_window = config["time_window"] if "time_window" in config else None

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_after(PostClientSelection(), ClientSelector)


class PostClientSelection(Handler):
    def _handle(self, request):
        selected_clients = request.get('selected_clients')
        scheduler = request.get('scheduler')
        epoch = request.get('epoch')
        stale_list = [scheduler.global_var["client_staleness_list"][stale] for stale in selected_clients]
        if scheduler.fixed_time_window:
            scheduler.normal_client_num = sum(1 for i in stale_list if i <= scheduler.fixed_time_window)
            scheduler.time_window = scheduler.fixed_time_window
        else:
            scheduler.clusters, centroids = kmeans1d.cluster(stale_list, 2)
            if centroids[0] < centroids[1] and scheduler.clusters.count(0) != 0:
                scheduler.normal_client_num = scheduler.clusters.count(0)
                scheduler.time_window = max(stale for i, stale in enumerate(stale_list) if scheduler.clusters[i] == 0)
            else:
                scheduler.normal_client_num = scheduler.clusters.count(1)
                scheduler.time_window = max(stale for i, stale in enumerate(stale_list) if scheduler.clusters[i] == 1)
        scheduler.queue_manager.set_version(epoch, len(selected_clients), scheduler.server_weights)
        scheduler.queue_manager.set_time_window(scheduler.time_window)
        return request
