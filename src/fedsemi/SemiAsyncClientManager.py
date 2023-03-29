import threading

import torch.cuda
from fedsemi import QueueManager
from utils import ModuleFindTool


class SemiAsyncClientManager:
    def __init__(self, init_weights, clients_num, multi_gpu, datasets, group_manager, current_time, stop_event,
                 client_config, manager_config, global_var):
        self.init_weights = init_weights
        self.group_manager = group_manager
        self.clients_num = clients_num
        self.batch_size = client_config["batch_size"]
        self.current_time = current_time
        self.stop_event = stop_event
        self.client_staleness_list = client_config["stale_list"]
        self.thread_lock = threading.Lock()
        self.epoch = client_config["epochs"]
        self.queue_manager = QueueManager.QueueManager([], current_time, manager_config["checker"])
        client_class = ModuleFindTool.find_class_by_path(manager_config["client_path"])
        self.global_var = global_var

        # 初始化clients
        # 0: 多gpu，1：单gpu，2：cpu
        if torch.cuda.is_available():
            if multi_gpu:
                mode = 0
                dev_num = 0
                dev_total = torch.cuda.device_count()
            else:
                mode = 1
        else:
            mode = 2
        self.client_thread_list = []
        for i in range(clients_num):
            if mode == 0:
                dev = f'cuda:{dev_num}'
                dev_num = (dev_num + 1) % dev_total
            elif mode == 1:
                dev = 'cuda'
            else:
                dev = 'cpu'
            client_delay = self.client_staleness_list[i]
            dataset = datasets[i]
            self.client_thread_list.append(
                client_class(i, self.queue_manager, self.stop_event, client_delay, dataset, client_config, dev, global_var))

        # 预分组
        self.group_manager.init(self.client_thread_list, self.client_staleness_list)

        # 启动 clients
        print("Start clients:")
        for client_thread in self.client_thread_list:
            client_thread.start()

    def stop_all_clients(self):
        # 终止所有client线程
        self.stop_event.set()
        for client_threads in self.client_thread_list:
            client_threads.set_event()

    def set_client_thread_list(self, new_client_thread_list):
        self.thread_lock.acquire()
        self.client_thread_list = new_client_thread_list
        self.thread_lock.release()

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

    def set_queue_list(self, queue_list):
        self.queue_manager.set_queue_list(queue_list)
