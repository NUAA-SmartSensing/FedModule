from client import NormalClient


class SemiClient(NormalClient.NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev):
        NormalClient.NormalClient.__init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev)
        self.group_id = 0

    def set_group_id(self, group_id):
        self.group_id = group_id

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "group_id": self.group_id}
        self.message_queue.put_into_uplink(update_dict)
