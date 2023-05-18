import copy
import threading

from utils import ModuleFindTool


class BaseScheduler(threading.Thread):
    def __init__(self, server_thread_lock, client_manager,
                 current_t, schedule_t, scheduler_config,
                 server_network, t, global_var):
        threading.Thread.__init__(self)
        self.server_thread_lock = server_thread_lock
        self.client_manager = client_manager
        self.current_t = current_t
        self.schedule_t = schedule_t
        self.server_network = server_network
        self.server_weights = copy.deepcopy(self.server_network.state_dict())
        self.T = t
        schedule_class = ModuleFindTool.find_class_by_path(scheduler_config["scheduler_path"])
        self.schedule = schedule_class()
        self.config = scheduler_config
        self.global_var = global_var

    def run(self):
        pass

    def client_select(self, params):
        client_list = self.client_manager.get_client_thread_list()
        selected_client_threads = self.schedule.schedule(client_list, params)
        return selected_client_threads
