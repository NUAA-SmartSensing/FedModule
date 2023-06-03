import copy
import threading

import torch.cuda

from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class BaseClientManager:
    def __init__(self, stop_event, config):
        self.stop_event = stop_event
        self.config = config
        self.global_var = GlobalVarGetter().get()
        self.client_thread_list = None

        self.multi_gpu = self.global_var["global_config"]["multi_gpu"]
        self.clients_num = self.global_var["global_config"]["client_num"]
        self.batch_size = self.global_var["client_config"]["batch_size"]
        self.client_staleness_list = self.global_var["client_config"]["stale_list"]
        self.epoch = self.global_var["client_config"]["epochs"]
        self.client_config = self.global_var["client_config"]
        self.current_time = self.global_var["current_t"]
        self.schedule_t = self.global_var["schedule_t"]
        self.datasets = self.global_var["dataset"].get_train_dataset()
        self.queue_manager = self.global_var["queue_manager"]
        self.print_lock = self.global_var["print_lock"]
        self.init_weights = copy.deepcopy(self.global_var["server_network"].state_dict())

        self.thread_lock = threading.Lock()
        self.client_class = ModuleFindTool.find_class_by_path(self.global_var["client_config"]["path"])

    def start_all_clients(self):
        self.init_clients()
        # 启动clients
        self.global_var['client_list'] = self.client_thread_list
        print("Start clients:")
        for client_thread in self.client_thread_list:
            client_thread.start()

    def stop_all_clients(self):
        # 终止所有client线程
        self.stop_event.set()
        for client_threads in self.client_thread_list:
            client_threads.set_event()

    def init_clients(self):
        mode, dev_num, dev_total = self.get_running_mode()
        # 初始化clients
        self.client_thread_list = []
        for i in range(self.clients_num):
            if mode == 0:
                dev = f'cuda:{dev_num}'
                dev_num = (dev_num + 1) % dev_total
            elif mode == 1:
                dev = 'cuda'
            else:
                dev = 'cpu'
            client_delay = self.client_staleness_list[i]
            dataset = self.datasets[i]
            self.client_thread_list.append(self.client_class(i, self.stop_event, client_delay, dataset, self.client_config, dev))

    def set_client_thread_list(self, new_client_thread_list):
        self.thread_lock.acquire()
        self.client_thread_list = new_client_thread_list
        self.thread_lock.release()

    def get_running_mode(self):
        dev_num = 0
        dev_total = 0
        # 0: 多gpu，1：单gpu，2：cpu
        if torch.cuda.is_available():
            if self.multi_gpu:
                mode = 0
                dev_num = 0
                dev_total = torch.cuda.device_count()
            else:
                mode = 1
        else:
            mode = 2
        return mode, dev_num, dev_total

    def get_client_thread_list(self):
        self.thread_lock.acquire()
        client_thread_list = self.client_thread_list
        self.thread_lock.release()
        return client_thread_list

    def find_client_thread_by_c_id(self, c_id):
        self.thread_lock.acquire()
        target_client_thread = None
        for client_thread in self.client_thread_list:
            if client_thread.get_client_id() == c_id:
                target_client_thread = client_thread
        self.thread_lock.release()
        return target_client_thread
