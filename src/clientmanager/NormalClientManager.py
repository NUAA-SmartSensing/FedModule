from clientmanager.BaseClientManager import BaseClientManager
from clientmanager.ClientFactroy import ClientFactory
from core.MessageQueue import EventFactory
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class NormalClientManager(BaseClientManager):
    def __init__(self, whole_config):
        super().__init__(whole_config)
        self.global_var = GlobalVarGetter.get()
        self.client_list = []  # client list
        self.client_id_list = []  # client id list

        config = whole_config["client_manager"]
        self.multi_gpu = whole_config["global"]["multi_gpu"]
        self.total_client_num = whole_config["global"]["client_num"]
        self.client_num = config.get("init_client_num", self.total_client_num)
        self.client_staleness_list = config["stale_list"]
        self.index_list = config["index_list"]  # each client's index list
        self.client_config = whole_config["client"]

        self.client_dev = self.get_client_dev_list(self.total_client_num, self.multi_gpu)
        self.client_factory = ModuleFindTool.find_class_by_path(
            config['client_factory']['path']) if 'client_factory' in config else ClientFactory
        self.stop_event_list = [EventFactory.create_Event() for _ in range(self.client_num)]
        self.selected_event_list = [EventFactory.create_Event() for _ in range(self.client_num)]
        self.global_var['selected_event_list'] = self.selected_event_list

    def start_all_clients(self):
        self.__init_clients()
        # start clients
        self.global_var['client_list'] = self.client_list
        self.global_var['client_id_list'] = self.client_id_list
        print("Starting clients")
        for i in self.client_id_list:
            self.client_list[i].start()

    def __init_clients(self):
        self.client_id_list = list(range(self.client_num))
        self.client_list = self.client_factory.create_clients(self.client_id_list, self.stop_event_list,
                                                              self.selected_event_list, self.client_staleness_list,
                                                              self.index_list,
                                                              self.client_config, self.client_dev,
                                                              self.global_var['config'])  # instance

    def get_client_list(self):
        return self.client_list

    def get_client_id_list(self):
        return self.client_id_list

    def stop_all_clients(self):
        # stop all clients
        for i in self.client_id_list:
            self.stop_client_by_id(i)

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id].set()
        self.selected_event_list[client_id].set()

    def create_and_start_new_client(self, dev='cpu'):
        client_id = len(self.client_id_list)
        self.client_list.append(
            self.client_factory.create_clients(client_id, self.stop_event_list[client_id],
                                               self.selected_event_list[client_id],
                                               self.client_staleness_list[client_id], self.index_list[client_id],
                                               self.client_config, self.client_dev[client_id]))  # instance
        self.client_id_list.append(client_id)
        self.client_list[client_id].start()
        self.client_num += 1

    def client_join(self):
        for i in self.client_list:
            i.join()
