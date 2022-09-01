import threading

from utils import ModuleFindTool


class AsyncClientManager:
    def __init__(self, init_weights, clients_num, datasets, q, current_time, stop_event, client_config, manager_config):
        self.init_weights = init_weights
        self.queue = q
        self.clients_num = clients_num
        self.batch_size = client_config["batch_size"]
        self.current_time = current_time
        self.stop_event = stop_event
        self.client_staleness_list = client_config["stale_list"]
        self.thread_lock = threading.Lock()
        self.epoch = client_config["epochs"]

        client_class = ModuleFindTool.find_class_by_string("client", manager_config["client_file"], manager_config["client_name"])

        # 初始化clients
        self.client_thread_list = []
        for i in range(clients_num):
            client_delay = self.client_staleness_list[i]
            dataset = datasets[i]
            self.client_thread_list.append(
                client_class(i, self.queue, self.stop_event, client_delay, dataset, client_config))

        # 启动clients
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
