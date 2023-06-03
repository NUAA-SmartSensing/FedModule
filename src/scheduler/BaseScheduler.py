import copy
import threading
from abc import abstractmethod

from schedule.ScheduleCaller import ScheduleCaller
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class BaseScheduler(threading.Thread):
    def __init__(self, server_thread_lock, config):
        threading.Thread.__init__(self)
        self.server_thread_lock = server_thread_lock
        self.config = config

        self.global_var = GlobalVarGetter().get()
        self.client_manager = self.global_var['client_manager']
        self.current_t = self.global_var['current_t']
        self.schedule_t = self.global_var['schedule_t']
        self.server_network = self.global_var['server_network']
        self.T = self.global_var['T']
        self.queue_manager = self.global_var['queue_manager']
        self.print_lock = self.global_var['print_lock']

        self.server_weights = copy.deepcopy(self.server_network.state_dict())

        schedule_class = ModuleFindTool.find_class_by_path(config["schedule"]["path"])
        self.schedule_method = schedule_class(config["schedule"]["params"])
        self.schedule_caller = ScheduleCaller(self)

    @abstractmethod
    def run(self):
        pass

    def client_select(self, *args, **kwargs):
        client_list = self.client_manager.get_client_thread_list()
        selected_client_threads = self.schedule_caller.schedule(client_list)
        return selected_client_threads
