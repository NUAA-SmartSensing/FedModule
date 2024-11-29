import copy
import time

from core.handlers.Handler import Handler


class UpdateSender(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.upload()
        return request


class UpdateReceiver(Handler):
    def _handle(self, request):
        client = request.get('client')
        weights_buffer = client.message_queue.get_from_downlink(client.client_id, 'weights')
        state_dict = client.model.state_dict()
        for k in weights_buffer:
            if client.training_params[k]:
                state_dict[k] = copy.deepcopy(weights_buffer[k])
        del weights_buffer
        client.model.load_state_dict(state_dict)
        client.time_stamp = client.message_queue.get_from_downlink(client.client_id, 'time_stamp')
        client.schedule_t = client.message_queue.get_from_downlink(client.client_id, 'schedule_time_stamp')
        if hasattr(client, 'receive_notify'):
            client.receive_notify()
        return request


class DelaySimulator(Handler):
    def _handle(self, request):
        client = request.get('client')
        if hasattr(client, 'delay_simulate'):
            client.delay_simulate()
        else:
            time.sleep(client.delay)
        return request

