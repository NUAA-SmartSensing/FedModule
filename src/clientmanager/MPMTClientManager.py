from multiprocessing import Process, Event

from clientmanager.NormalClientManager import NormalClientManager
from core.MessageQueue import MessageQueueFactory


def distribute_evenly(x, length):
    avg = x // length
    remainder = x % length
    result = [avg] * length
    for i in range(remainder):
        result[i] += 1
    return result


class MPMTClientManager(NormalClientManager):
    r"""
    The client manager of the multi-process with multi-thread mode.
    """

    def __init__(self, whole_config):
        super().__init__(whole_config)
        client_manager_config = whole_config["client_manager"]
        self.process_num = client_manager_config["process_num"]
        print(f"Process Nums: {self.process_num}")
        self.process_stop_event = [Event() for _ in range(self.process_num)]
        self.create_client_event = [Event() for _ in range(self.process_num)]
        self.init_event = Event()
        self.run_event = Event()
        self.stop_event = Event()
        self.create_client_event = [Event() for _ in range(self.process_num)]

        client_nums = distribute_evenly(self.client_num, self.process_num)
        self.process_pool = [MPMT(i, self.process_num, client_nums[i], self.client_class, self.init_event, self.run_event,
                                  self.stop_event, self.create_client_event[i], self.stop_event_list[i::self.process_num], self.selected_event_list[i::self.process_num],
                                  self.client_staleness_list[i::self.process_num], self.index_list[i::self.process_num], self.client_config, self.client_dev[i::self.process_num]) for i in range(self.process_num)]

    def start_all_clients(self):
        self.__init_clients()
        # start clients
        self.global_var['client_list'] = self.client_list
        self.global_var['client_id_list'] = self.client_id_list
        print("Starting clients")
        self.run_event.set()

    def __init_clients(self):
        for process in self.process_pool:
            process.start()
        self.init_event.set()
        self.client_list = list(range(self.client_num))
        self.client_id_list = list(range(self.client_num))

    def create_and_start_new_client(self, dev='cpu'):
        client_id = self.client_num
        self.create_client_event[client_id % self.process_num].set()
        self.client_id_list.append(client_id)
        self.client_num += 1

    def client_join(self):
        self.stop_event.set()
        for i in self.process_pool:
            i.join()

    def stop_all_clients(self):
        # stop all clients
        for i in self.client_id_list:
            self.stop_client_by_id(i)
        self.stop_event.set()
        for e in self.create_client_event:
            e.set()

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id].set()
        self.selected_event_list[client_id].set()


class MPMT(Process):
    def __init__(self, id, process_num, init_client_num, client_class, init_event, run_event, stop_event, create_client_event, stop_event_list, selected_event_list,
                 client_staleness_list, index_list, client_config, client_dev):
        super().__init__()
        self.id = id
        self.process_num = process_num
        self.client_num = init_client_num
        self.client_list = []
        self.create_client_event = create_client_event
        self.init_event = init_event
        self.run_event = run_event
        self.stop_event = stop_event
        self.message_queue = MessageQueueFactory.create_message_queue()

        # client params
        self.stop_event_list = stop_event_list
        self.selected_event_list = selected_event_list
        self.client_staleness_list = client_staleness_list
        self.index_list = index_list
        self.client_config = client_config
        self.client_dev = client_dev
        self.client_class = client_class

    def run(self):
        self.init_event.wait()
        self.init()
        self.run_event.wait()
        self.run_client()
        while True:
            self.create_client_event.wait()
            if self.stop_event.is_set():
                break
            self.create_client()
            self.create_client_event.clear()
        for i in self.client_list:
            i.join()

    def create_client(self):
        self.client_list.append(
            self.client_class(self.client_num*self.process_num+self.id, self.stop_event_list[self.client_num], self.selected_event_list[self.client_num], self.client_staleness_list[self.client_num],
                              self.index_list[self.client_num], self.client_config, self.client_dev[self.client_num]))
        self.client_num += 1
        self.client_list[-1].start()

    def init(self):
        for i in range(self.client_num):
            self.client_list.append(
                self.client_class(i*self.process_num+self.id, self.stop_event_list[i], self.selected_event_list[i], self.client_staleness_list[i],
                                  self.index_list[i], self.client_config, self.client_dev[i]))

    def run_client(self):
        for i in self.client_list:
            i.start()
