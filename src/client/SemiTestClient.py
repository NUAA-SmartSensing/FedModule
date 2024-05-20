from client.SemiClient import SemiClient
from client.TestClient import TestClient


class SemiTestClient(TestClient, SemiClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        TestClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        SemiClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "group_id": self.group_id,
                       "accuracy": self.accuracy_list[len(self.accuracy_list) - 1],
                       "loss": self.loss_list[len(self.loss_list) - 1]}
