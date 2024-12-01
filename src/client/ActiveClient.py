import time

from client.NormalClient import NormalClient
from client.mixin.ClientHandler import UpdateReceiver
from core.handlers.Handler import Handler


class ActiveClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = None
        self.acquire_model_delay = config['acquire_model_delay']

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(ActiveReceiver(), UpdateReceiver)

    def delay_simulate(self):
        time.sleep(self.acquire_model_delay + self.delay)


class ActiveReceiver(Handler):
    def _handle(self, request):
        client = request.get('client')
        while True:
            if client.stop_event.is_set():
                break
            latest_model, time_stamp = client.message_queue.get_latest_model()
            if latest_model is not None and time_stamp != client.time_stamp:
                break
            else:
                time.sleep(0.01)
        latest_model, time_stamp = client.message_queue.get_latest_model()
        client.model.load_state_dict(latest_model)
        client.time_stamp = time_stamp
        return request
