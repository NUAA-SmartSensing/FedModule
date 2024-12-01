import random

from client.NormalClient import NormalClient
from core.handlers.Handler import Handler
from core.handlers.ModelTrainHandler import ClientTrainHandler


class ProxClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.max_epoch = self.epoch

    def create_handler_chain(self):
        super().create_handler_chain()
        self.init_chain.add_handler_before(RandomEpoch(), ClientTrainHandler)


class RandomEpoch(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.epoch = random.randint(1, client.max_epoch)
        return request
