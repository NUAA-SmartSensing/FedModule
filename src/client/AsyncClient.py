from client import NormalClient


class AsyncClient(NormalClient.NormalClient):
    def __init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev, print_lock, global_var):
        NormalClient.NormalClient.__init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev,
                                           print_lock, global_var)
        self.schedule_t = None

    def set_schedule_time_stamp(self, schedule_t):
        self.schedule_t = schedule_t
