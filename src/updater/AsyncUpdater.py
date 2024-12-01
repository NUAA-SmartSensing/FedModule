from core.handlers.Handler import Handler
from core.handlers.ServerHandler import ClientUpdateGetter
from numgenerator.NumGeneratorFactory import NumGeneratorFactory
from updater.SyncUpdater import SyncUpdater
from utils.GlobalVarGetter import GlobalVarGetter


class AsyncUpdater(SyncUpdater):
    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.exchange_handler(AsyncClientUpdateGetter(), ClientUpdateGetter)


class AsyncClientUpdateGetter(Handler):
    def __init__(self):
        super().__init__()
        config = GlobalVarGetter.get()['config']['server']['updater']
        self.num_generator = NumGeneratorFactory(config['num_generator']).create_num_generator()
        self.sum_delay = [0]

    def _handle(self, request):
        epoch = request.get('epoch')
        nums = self.num_generator.get_num()
        queue_manager = request['updater'].queue_manager
        queue_manager.receive(nums)
        update_list = []
        for i in range(nums):
            update_list.append(queue_manager.get())
            c_id = update_list[i]["client_id"]
            time_stamp = update_list[i]["time_stamp"]
            self.sum_delay[0] += (epoch - time_stamp)
            print("Updater received data from client", c_id, "| staleness =", time_stamp, "-",
                  epoch, "| queue size = ", queue_manager.size())
        request['update_list'] = update_list
        return request

    def run_once(self, request):
        updater = request.get('updater')
        updater.add_final_callback(lambda x: print("Average staleness:", x[0] / updater.T), self.sum_delay)
