import copy

from client.NormalClient import NormalClient
from client.mixin.ClientHandler import UpdateReceiver
from core.handlers.Handler import Handler


class BNClient(NormalClient):
    def create_handler_chain(self):
        super().create_handler_chain()
        self.init_chain.exchange_handler(BNUpdateReceiver(), UpdateReceiver)


class BNUpdateReceiver(Handler):
    def _handle(self, request):
        client = request.get('client')
        weights_buffer = client.message_queue.get_from_downlink(client.client_id, 'weights')
        state_dict = client.model.state_dict()
        for k in weights_buffer:
            if client.training_params[k] and 'bn' not in k:
                state_dict[k] = copy.deepcopy(weights_buffer[k])
        del weights_buffer
        client.model.load_state_dict(state_dict)
        client.time_stamp = client.message_queue.get_from_downlink(client.client_id, 'time_stamp')
        client.schedule_t = client.message_queue.get_from_downlink(client.client_id, 'schedule_time_stamp')
        return request
