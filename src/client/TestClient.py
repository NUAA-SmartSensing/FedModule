from client.NormalClient import NormalClient
from client.mixin.Test import TestableMixin


class TestClient(NormalClient, TestableMixin):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        TestableMixin.__init__(self, config)
        self.global_config = None

    def train(self):
        data_sum, weights = super().train()
        self.run_test(self.client_id, self.time_stamp, self.model, self.loss_func, self.dev)
        return data_sum, weights

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp,
                       "accuracy": self.accuracy_list[len(self.accuracy_list) - 1],
                       "loss": self.loss_list[len(self.loss_list) - 1]}
        self.message_queue.put_into_uplink(update_dict)

    def init_client(self):
        super().init_client()
        self.global_config = self.message_queue.get_config_by_key('global_config')

    def finish_client(self):
        if self.global_config['save']:
            experiment = self.global_config['experiment']
            self.save_test_record(self.client_id, experiment)
