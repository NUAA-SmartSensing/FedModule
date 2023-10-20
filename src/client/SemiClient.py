from time import sleep

from client.NormalClient import NormalClient


class SemiClient(NormalClient):
    def __init__(self, c_id, init_lock, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, init_lock, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = 0

    def run(self):
        while True:
            self.group_id = self.message_queue.get_from_downlink(self.client_id, "group_id")
            if self.group_id is not None:
                break
            sleep(0.01)
        NormalClient.NormalClient.run(self)

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "group_id": self.group_id}
        self.message_queue.put_into_uplink(update_dict)
