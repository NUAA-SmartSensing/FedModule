import time

from client import NormalClient
from utils.ModelTraining import train_one_epoch


class SemiClient(NormalClient.NormalClient):
    def __init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev, print_lock,  global_var):
        super().__init__(c_id, queue_manager, stop_event, delay, train_ds, client_config, dev, print_lock,  global_var)
        self.group_id = 0

    def set_group_id(self, group_id):
        self.group_id = group_id

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "group_id": self.group_id}
        self.queue_manager.put(update_dict)
