import copy

import torch

from client.TestClient import TestClient
from client.mixin.ClientHandler import UpdateReceiver
from core.handlers.Handler import Handler


class DLClient(TestClient):
    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.exchange_handler(PersonalUpdateReceiver(), UpdateReceiver)


class PersonalUpdateReceiver(Handler):
    def __init__(self):
        super().__init__()
        self.is_init = False

    def _handle(self, request):
        client = request.get('client')
        weights_buffer = client.message_queue.get_from_downlink(client.client_id, 'weights')
        if self.is_init:
            if client.client_id in weights_buffer.keys():
                for key, var in client.model.state_dict().items():
                    if client.training_params[key]:
                        if torch.cuda.is_available():
                            weights_buffer[client.client_id][key] = weights_buffer[client.client_id][key].to(
                                client.dev)
                        weights_buffer[client.client_id][key] = client.config['alpha'] * var + (
                                1 - client.config['alpha']) * weights_buffer[client.client_id][key]
                client.model.load_state_dict(weights_buffer[client.client_id], strict=True)
        else:
            if 'global' in weights_buffer.keys():
                client.model.load_state_dict(weights_buffer['global'], strict=True)
            else:
                client.model.load_state_dict(weights_buffer, strict=True)
            self.is_init = True
        del weights_buffer
        client.time_stamp = client.message_queue.get_from_downlink(client.client_id, 'time_stamp')
        client.schedule_t = client.message_queue.get_from_downlink(client.client_id, 'schedule_time_stamp')
        return request
