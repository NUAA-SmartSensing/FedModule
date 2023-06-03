from client import NormalClient


class SemiClient(NormalClient.NormalClient):
    def __init__(self, c_id, stop_event, delay, train_ds, config, dev):
        NormalClient.NormalClient.__init__(self, c_id, stop_event, delay, train_ds, config, dev)
        self.group_id = 0

    def set_group_id(self, group_id):
        self.group_id = group_id

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "group_id": self.group_id}
        self.queue_manager.put(update_dict)
