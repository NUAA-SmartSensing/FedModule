from client.NormalClient import NormalClient


class BNClient(NormalClient):
    def receive_notify(self):
        if self.message_queue.get_from_downlink(self.client_id, 'received_weights'):
            if self.training_params is None:
                self.training_params = self.message_queue.get_training_params()
            self.message_queue.put_into_downlink(self.client_id, 'received_weights', False)
            weights_buffer = self.message_queue.get_from_downlink(self.client_id, 'weights_buffer')
            state_dict = self.model.state_dict()
            for k in weights_buffer:
                if self.training_params[k] and 'bn' not in k:
                    state_dict[k] = weights_buffer[k]
            self.model.load_state_dict(state_dict)
        if self.message_queue.get_from_downlink(self.client_id, 'received_time_stamp'):
            self.message_queue.put_into_downlink(self.client_id, 'received_time_stamp', False)
            self.time_stamp = self.message_queue.get_from_downlink(self.client_id, 'time_stamp_buffer')
            self.schedule_t = self.message_queue.get_from_downlink(self.client_id, 'schedule_time_stamp_buffer')

