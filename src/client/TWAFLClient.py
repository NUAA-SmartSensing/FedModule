from time import sleep

from client.NormalClient import NormalClient
from client.mixin.ClientHandler import DelaySimulator
from core.handlers.Handler import Handler


class TWAFLClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.new_delay = config.get('new_delay', self.delay/2)

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.exchange_handler(TWAFLDelay(), DelaySimulator)


class TWAFLDelay(Handler):
    def _handle(self, request):
        client = request.get('client')
        flag = client.time_stamp % 15 in [11, 13, 14, 12, 0]
        if flag:
            sleep(client.delay)
        else:
            sleep(client.new_delay)
        return request
