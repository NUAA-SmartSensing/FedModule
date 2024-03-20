import warnings
from multiprocessing import Event

from clientmanager.BaseClientManager import BaseClientManager
from core.Timeslice import TimeSliceRunner
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import MessageQueueFactory


class TimeSliceClientManager(BaseClientManager):
    def __init__(self, whole_config):
        super().__init__(whole_config)
        self.global_var = GlobalVarGetter.get()
        self.stop_event_list = [Event() for _ in whole_config["global"]["client_num"]]
        self.selected_event_list = [Event() for _ in whole_config["global"]["client_num"]]
        self.global_var['selected_event_list'] = self.selected_event_list
        self.create_client_event = Event()
        self.init_client_event = Event()
        self.join_event = Event()
        self.client_id_list = list(range(whole_config["global"]["client_num"]))
        client_dev = self.get_client_dev_list(whole_config["global"]["client_num"], whole_config["global"]["multi_gpu"])
        self.runner = TimeSliceRunner(self.init_client_event, self.create_client_event, self.join_event,
                                      self.stop_event_list, self.selected_event_list,
                                      {"index_list": whole_config["client_manager"]["index_list"],
                                       "client_config": whole_config["client"]["config"],
                                       "multi_gpu": whole_config["global"]["multi_gpu"],
                                       "client_staleness_list": whole_config["client_manager"]["staleness_list"],
                                       "client_num": whole_config["global"]["client_num"], "client_dev": client_dev})
        self.runner.start()

    def start_all_clients(self):
        self.init_client_event.set()

    def stop_all_clients(self):
        for e in self.stop_event_list:
            e.set()

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id].set()
        self.selected_event_list[client_id].set()

    def create_and_start_new_client(self, client_delay, dev='cpu'):
        mq = MessageQueueFactory.create_message_queue()
        mq.put_into_downlink(-1, "client_delay", client_delay)
        mq.put_into_downlink(-1, "dev", dev)
        self.create_client_event.set()
        self.client_id_list.append(len(self.client_id_list))
        self.client_list.append(len(self.client_list))

    def client_join(self):
        self.join_event.wait()
        print("All clients joined.")
        self.runner.join()
        print("client runner joined.")
        print("-----------------Client Manager joined.-----------------")

    def get_client_list(self):
        warnings.warn("This function is not implemented.")
        return self.client_list
