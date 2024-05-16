import torch

from client.TestClient import TestClient


class DLClient(TestClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        TestClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.init = False

    def receive_notify(self):
        if self.message_queue.get_from_downlink(self.client_id, 'received_weights'):
            self.message_queue.put_into_downlink(self.client_id, 'received_weights', False)
            self.weights_buffer = self.message_queue.get_from_downlink(self.client_id, 'weights_buffer')
            # 更新模型参数
            if self.init:
                if self.client_id in self.weights_buffer.keys():
                    for key, var in self.model.state_dict().items():
                        if self.training_params[key]:
                            if torch.cuda.is_available():
                                self.weights_buffer[self.client_id][key] = self.weights_buffer[self.client_id][key].to(
                                    self.dev)
                            self.weights_buffer[self.client_id][key] = self.config['alpha'] * var + (
                                    1 - self.config['alpha']) * self.weights_buffer[self.client_id][key]
                    self.model.load_state_dict(self.weights_buffer[self.client_id], strict=True)
            else:
                self.model.load_state_dict(self.weights_buffer['global'], strict=True)
                self.init = True
            self.received_weights = False
        if self.message_queue.get_from_downlink(self.client_id, 'received_time_stamp'):
            self.message_queue.put_into_downlink(self.client_id, 'received_time_stamp', False)
            self.time_stamp = self.message_queue.get_from_downlink(self.client_id, 'time_stamp_buffer')
            self.schedule_t = self.message_queue.get_from_downlink(self.client_id, 'schedule_time_stamp_buffer')
