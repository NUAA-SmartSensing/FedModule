import random
import threading

from client import AsyncClient
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
        self.check_in_thread_lock = threading.Lock()
        self.epoch = client_config["epochs"]

        client_class = ModuleFindTool.find_class_by_string("client", manager_config["client_file"], manager_config["client_name"])

        # 初始化clients
        self.client_thread_list = []
        for i in range(clients_num):
            client_delay = self.client_staleness_list[i]
            dataset = datasets[i]
            self.client_thread_list.append(
                client_class(i, self.queue, self.stop_event, client_delay, dataset, client_config))

        self.checked_in_client_thread_list = []
        self.unchecked_in_client_thread_list = []
        self.checking_in_client_thread_id_list = []
        for i in range(clients_num):
            # if i < 20:
            if i % 100 < 20:  # 140:
                self.checked_in_client_thread_list.append(self.client_thread_list[i])
            else:
                self.unchecked_in_client_thread_list.append(self.client_thread_list[i])

        # 启动checked in clients
        print("Start checked in clients:")
        for client_thread in self.checked_in_client_thread_list:
            client_thread.start()

    def client_check_in(self, check_in_number):
        if len(self.unchecked_in_client_thread_list) > 0:
            self.thread_lock.acquire()
            if check_in_number >= len(self.unchecked_in_client_thread_list):
                print("| remain---------------------------------------------------------------------", check_in_number)
                check_in_number = len(self.unchecked_in_client_thread_list)
            check_in_clients = random.sample(self.unchecked_in_client_thread_list, check_in_number)

            # 去除已经在checking in的clients
            cc = 0
            while cc < len(check_in_clients):
                cc_id = int(check_in_clients[cc].get_client_id())
                if cc_id in self.checking_in_client_thread_id_list:
                    check_in_clients.remove(check_in_clients[cc])
                else:
                    cc += 1

            # 启动received_client_thread
            for c_i_client in check_in_clients:
                print("Start client", c_i_client.get_client_id())
                c_i_client.start()
                self.checking_in_client_thread_id_list.append(int(c_i_client.get_client_id()))
            self.thread_lock.release()
            self.thread_lock.acquire()
            # 更新checked_in_client_thread_list和checked_in_client_thread_list
            for c_i_client in check_in_clients:
                self.checked_in_client_thread_list.append(c_i_client)
                self.unchecked_in_client_thread_list.remove(c_i_client)
                print("[", len(self.checked_in_client_thread_list), len(self.unchecked_in_client_thread_list), "]")
            self.thread_lock.release()
        else:
            check_in_number = 0

        return check_in_number

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

    def set_checked_in_client_thread_list(self, new_checked_in_client_thread_list):
        self.thread_lock.acquire()
        self.checked_in_client_thread_list = new_checked_in_client_thread_list
        self.thread_lock.release()

    def get_checked_in_client_thread_list(self):
        self.thread_lock.acquire()
        checked_in_client_thread_list = self.checked_in_client_thread_list
        self.thread_lock.release()
        return checked_in_client_thread_list

    def get_unchecked_in_client_thread_list_len(self):
        self.thread_lock.acquire()
        unchecked_in_client_thread_list_len = len(self.unchecked_in_client_thread_list)
        self.thread_lock.release()
        return unchecked_in_client_thread_list_len

    def get_check_in_thread_lock(self):
        return self.check_in_thread_lock

